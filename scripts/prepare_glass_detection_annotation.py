import argparse
import csv
import math
import shutil
from pathlib import Path

import cv2
import numpy as np

from auto_crop_glass_defect import (
    BandEstimate,
    bright_band_mask,
    build_camera_references,
    camera_name,
    crop_x_from_component,
    defect_component,
    estimate_camera_bands,
    find_images,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a pseudo-labeled annotation batch for glass defect detection."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--samples-per-camera",
        type=int,
        default=60,
        help="Number of images to export per camera.",
    )
    parser.add_argument("--crop-width", type=int, default=1200)
    parser.add_argument("--crop-height", type=int, default=900)
    parser.add_argument("--sample-per-camera", type=int, default=80)
    parser.add_argument("--reference-per-camera", type=int, default=31)
    parser.add_argument(
        "--label-trust-threshold",
        type=float,
        default=0.55,
        help="Only suggestions above this trust score are written as YOLO labels unless overridden.",
    )
    parser.add_argument(
        "--write-low-trust-labels",
        action="store_true",
        help="Write YOLO labels even when the suggested box trust is low.",
    )
    return parser.parse_args()


def group_by_camera(image_paths: list[Path]) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = {}
    for path in image_paths:
        grouped.setdefault(camera_name(path), []).append(path)
    return grouped


def evenly_sample(paths: list[Path], count: int) -> list[Path]:
    """Pick a time-spread subset without clustering samples in one segment.

    Args:
        paths: Ordered image paths for one camera.
        count: Number of samples to keep.

    Returns:
        A roughly even subset that spans the original sequence.
    """

    if len(paths) <= count:
        return paths
    indices = np.linspace(0, len(paths) - 1, num=count, dtype=int)
    ordered_indices: list[int] = []
    seen: set[int] = set()
    for index in indices.tolist():
        if index not in seen:
            ordered_indices.append(index)
            seen.add(index)
    return [paths[index] for index in ordered_indices]


def split_train_val(paths: list[Path], val_ratio: float = 0.2) -> tuple[list[Path], list[Path]]:
    """Create a deterministic interleaved split that preserves camera coverage over time.

    Args:
        paths: Ordered image paths selected for annotation.
        val_ratio: Target fraction for the validation split.

    Returns:
        A `(train_paths, val_paths)` tuple.
    """

    if len(paths) <= 1:
        return paths, []

    # Interleave the split deterministically so nearby frames do not all land in the same subset.
    val_every = max(2, int(round(1.0 / max(val_ratio, 1e-6))))
    train_paths: list[Path] = []
    val_paths: list[Path] = []
    for idx, path in enumerate(paths):
        if idx % val_every == 0:
            val_paths.append(path)
        else:
            train_paths.append(path)
    if not train_paths:
        train_paths, val_paths = val_paths[1:], val_paths[:1]
    return train_paths, val_paths


def expand_bbox(
    bbox: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    scale: float = 1.35,
    min_size: int = 48,
) -> tuple[int, int, int, int]:
    """Loosen a tight heuristic component box into a more annotation-friendly proposal.

    Args:
        bbox: Input `xywh` box in full-image coordinates.
        image_width: Full image width.
        image_height: Full image height.
        scale: Multiplicative expansion factor for width and height.
        min_size: Minimum output size for each side.

    Returns:
        A clamped expanded `xywh` box in full-image coordinates.
    """

    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0
    new_w = max(min_size, int(round(w * scale)))
    new_h = max(min_size, int(round(h * scale)))
    left = int(round(cx - new_w / 2.0))
    top = int(round(cy - new_h / 2.0))
    left = max(0, min(left, image_width - new_w))
    top = max(0, min(top, image_height - new_h))
    return left, top, new_w, new_h


def bbox_to_yolo(
    bbox: tuple[int, int, int, int], image_width: int, image_height: int
) -> tuple[float, float, float, float]:
    """Convert an absolute XYWH box into normalized YOLO center-width-height format.

    Args:
        bbox: Input `xywh` box in full-image coordinates.
        image_width: Full image width.
        image_height: Full image height.

    Returns:
        Normalized `(x_center, y_center, width, height)` values.
    """

    x, y, w, h = bbox
    return (
        (x + w / 2.0) / image_width,
        (y + h / 2.0) / image_height,
        w / image_width,
        h / image_height,
    )


def overlay_annotation(
    image: np.ndarray,
    estimate: BandEstimate,
    crop_left: int,
    crop_width: int,
    component_bbox: tuple[int, int, int, int] | None,
    suggested_bbox: tuple[int, int, int, int] | None,
) -> np.ndarray:
    """Render the ROI, raw heuristic component, and expanded pseudo-label on the image.

    Args:
        image: Original color image in full-image coordinates.
        estimate: Per-camera bright-band and crop geometry.
        crop_left: Left edge of the suggested crop window.
        crop_width: Width of the suggested crop window.
        component_bbox: Tight heuristic component box in ROI-local coordinates.
        suggested_bbox: Expanded pseudo-label in full-image coordinates.

    Returns:
        A debug image with overlays used during manual review.
    """

    debug = image.copy()
    cv2.rectangle(
        debug,
        (crop_left, estimate.crop_top),
        (crop_left + crop_width, estimate.crop_bottom),
        (0, 255, 0),
        3,
    )
    for start, end in (
        (estimate.band1_start, estimate.band1_end),
        (estimate.band2_start, estimate.band2_end),
    ):
        cv2.rectangle(debug, (0, start), (debug.shape[1] - 1, end), (255, 128, 0), 2)
    if component_bbox is not None:
        x, y, w, h = component_bbox
        cv2.rectangle(
            debug,
            (x, y + estimate.crop_top),
            (x + w, y + h + estimate.crop_top),
            (0, 0, 255),
            2,
        )
    if suggested_bbox is not None:
        x, y, w, h = suggested_bbox
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 255), 2)
    return debug


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    image_paths = find_images(input_dir)
    if not image_paths:
        raise SystemExit(f"No .jpg files found under {input_dir}")

    estimates = estimate_camera_bands(image_paths, args.crop_height, args.sample_per_camera)
    references = build_camera_references(image_paths, estimates, args.reference_per_camera)
    grouped = group_by_camera([path for path in image_paths if camera_name(path) in estimates])

    export_paths: list[Path] = []
    for cam, paths in sorted(grouped.items()):
        # Sample evenly per camera to keep the batch representative across viewpoints and time.
        export_paths.extend(evenly_sample(paths, args.samples_per_camera))

    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    overlays_dir = output_dir / "overlays"
    for directory in (images_dir, labels_dir, overlays_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    for index, path in enumerate(export_paths, start=1):
        cam = camera_name(path)
        estimate = estimates[cam]
        reference_crop = references.get(cam)

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray_crop = gray[estimate.crop_top : estimate.crop_bottom, :]
        band_mask = bright_band_mask(gray_crop, estimate)
        component_bbox, score_map, component_meta = defect_component(
            gray_crop, band_mask, reference_crop
        )
        crop_left = crop_x_from_component(
            component_bbox,
            score_map,
            args.crop_width,
            gray.shape[1],
            component_meta,
        )

        suggested_bbox = None
        if component_bbox is not None:
            x, y, w, h = component_bbox
            # Convert the tight ROI-local component into a looser full-image box for manual correction.
            suggested_bbox = expand_bbox(
                (x, y + estimate.crop_top, w, h),
                image_width=gray.shape[1],
                image_height=gray.shape[0],
            )

        image_out = images_dir / path.name
        label_out = labels_dir / f"{path.stem}.txt"
        overlay_out = overlays_dir / path.name
        shutil.copy2(path, image_out)

        label_written = False
        if suggested_bbox is not None:
            trust_score = component_meta["trust_score"] if component_meta is not None else 0.0
            # Low-trust suggestions are intentionally left blank so reviewers are not nudged by bad boxes.
            if args.write_low_trust_labels or trust_score >= args.label_trust_threshold:
                x_center, y_center, box_w, box_h = bbox_to_yolo(
                    suggested_bbox,
                    image_width=gray.shape[1],
                    image_height=gray.shape[0],
                )
                label_out.write_text(
                    f"0 {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n",
                    encoding="utf-8",
                )
                label_written = True
            else:
                label_out.write_text("", encoding="utf-8")
        else:
            label_out.write_text("", encoding="utf-8")

        overlay = overlay_annotation(
            bgr,
            estimate,
            crop_left,
            args.crop_width,
            component_bbox,
            suggested_bbox,
        )
        cv2.imwrite(str(overlay_out), overlay)

        manifest_rows.append(
            {
                "file_name": path.name,
                "camera": cam,
                "image_path": str(image_out),
                "overlay_path": str(overlay_out),
                "label_path": str(label_out),
                "crop_left": crop_left,
                "crop_top": estimate.crop_top,
                "suggested_bbox_xywh": ""
                if suggested_bbox is None
                else ",".join(map(str, suggested_bbox)),
                "component_bbox_xywh": ""
                if component_bbox is None
                else ",".join(
                    map(
                        str,
                        (
                            component_bbox[0],
                            component_bbox[1] + estimate.crop_top,
                            component_bbox[2],
                            component_bbox[3],
                        ),
                    )
                ),
                "component_angle_degrees": ""
                if component_meta is None
                else round(component_meta["angle_degrees"], 2),
                "component_trust_score": ""
                if component_meta is None
                else round(component_meta["trust_score"], 4),
                "label_written": int(label_written),
                "review_code": "",
                "review_note": "",
            }
        )

        if index % 100 == 0:
            print(f"prepared {index}/{len(export_paths)} images")

    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    train_paths, val_paths = split_train_val(export_paths, val_ratio=0.2)
    (output_dir / "train.txt").write_text(
        "\n".join((images_dir / path.name).as_posix() for path in train_paths) + "\n",
        encoding="utf-8",
    )
    (output_dir / "val.txt").write_text(
        "\n".join((images_dir / path.name).as_posix() for path in val_paths) + "\n",
        encoding="utf-8",
    )

    (output_dir / "dataset.yaml").write_text(
        "\n".join(
            [
                f"path: {output_dir.as_posix()}",
                "train: train.txt",
                "val: val.txt",
                "names:",
                "  0: glass_defect",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (output_dir / "README.txt").write_text(
        "\n".join(
            [
                "Annotation batch for single-class defect detection.",
                "",
                "Folders:",
                "- images/: original images",
                "- labels/: YOLO-format suggestions",
                "- overlays/: original images with bright bands, crop window and suggested boxes",
                "",
                "Colors in overlays:",
                "- blue: estimated bright bands",
                "- green: suggested 1200x900 crop",
                "- red: raw component chosen by the heuristic",
                "- yellow: expanded pseudo-label suggested for annotation",
                "",
                "Suggested workflow:",
                "1. Open images + labels in your annotation tool",
                "2. Keep or adjust yellow suggested boxes",
                "3. Delete boxes on false positives and add boxes on missed defects",
                "4. Fill manifest.csv review_code/review_note if you want to audit suggestion quality",
                "",
                "review_code suggestion:",
                "1 = suggestion good",
                "0 = suggestion wrong",
                "2 = partly useful but needs adjustment",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"prepared {len(manifest_rows)} annotation images under {output_dir}")


if __name__ == "__main__":
    main()
