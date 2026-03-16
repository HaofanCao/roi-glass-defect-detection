import argparse
import csv
import json
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision.transforms import functional as F

from auto_crop_glass_defect import (
    BandEstimate,
    bright_band_mask,
    build_camera_references,
    camera_name,
    crop_x_from_component,
    defect_component,
    find_images,
)
from train_glass_defect_detector import make_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the trained glass defect detector and crop images to 1200x900."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crop-width", type=int, default=1200)
    parser.add_argument("--crop-height", type=int, default=900)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--reference-per-camera", type=int, default=31)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--debug-overlays", action="store_true")
    return parser.parse_args()


def load_estimates(checkpoint_dir: Path) -> dict[str, BandEstimate]:
    """Load the per-camera ROI geometry saved alongside the trained checkpoint.

    Args:
        checkpoint_dir: Directory containing the trained model artifacts.

    Returns:
        A mapping from camera name to its stored ROI estimate.
    """

    # Inference reuses the training-time ROI geometry instead of re-estimating it from scratch.
    data = json.loads((checkpoint_dir / "camera_band_estimates.json").read_text(encoding="utf-8"))
    return {cam: BandEstimate(**estimate) for cam, estimate in data.items()}


def select_best_prediction(output: dict[str, torch.Tensor], score_threshold: float):
    """Return the top-scoring detection that survives thresholding, or None.

    Args:
        output: Raw detector output for one image.
        score_threshold: Minimum score required for a prediction to survive.

    Returns:
        A dictionary with the best prediction fields, or `None` if nothing survives.
    """

    if output["boxes"].numel() == 0:
        return None
    keep = output["scores"] >= score_threshold
    if int(keep.sum().item()) == 0:
        return None
    # Use the highest-score prediction after thresholding because the pipeline only needs one crop center.
    boxes = output["boxes"][keep].cpu()
    scores = output["scores"][keep].cpu()
    labels = output["labels"][keep].cpu()
    best_idx = int(torch.argmax(scores).item())
    return {
        "box_xyxy": boxes[best_idx].tolist(),
        "score": float(scores[best_idx].item()),
        "label": int(labels[best_idx].item()),
    }


def overlay_debug(
    image: cv2.Mat,
    estimate: BandEstimate,
    crop_left: int,
    crop_width: int,
    model_bbox_full: tuple[int, int, int, int] | None,
    heuristic_bbox_full: tuple[int, int, int, int] | None,
) -> cv2.Mat:
    """Render the final crop, bright bands, and both model/heuristic candidate boxes.

    Args:
        image: Original color image in full-image coordinates.
        estimate: Per-camera bright-band and crop geometry.
        crop_left: Left edge of the final crop window.
        crop_width: Width of the final crop window.
        model_bbox_full: Detector prediction in full-image `xywh` format, if available.
        heuristic_bbox_full: Heuristic fallback box in full-image `xywh` format, if available.

    Returns:
        A debug image showing the crop region and both candidate sources.
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
    if heuristic_bbox_full is not None:
        x, y, w, h = heuristic_bbox_full
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 0, 255), 2)
    if model_bbox_full is not None:
        x, y, w, h = model_bbox_full
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 255), 2)
    return debug


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    checkpoint_dir = checkpoint_path.parent
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    config = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    estimates = load_estimates(checkpoint_dir)
    image_paths = find_images(input_dir)
    if args.limit > 0:
        image_paths = image_paths[: args.limit]

    # References are only needed for the heuristic fallback path when the detector stays below threshold.
    references = build_camera_references(image_paths, estimates, args.reference_per_camera)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = make_model(min_size=int(config["min_size"]), max_size=int(config["max_size"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    images_dir = output_dir / "images"
    debug_dir = output_dir / "debug"
    images_dir.mkdir(parents=True, exist_ok=True)
    if args.debug_overlays:
        debug_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows: list[dict[str, object]] = []
    for index, image_path in enumerate(image_paths, start=1):
        cam = camera_name(image_path)
        estimate = estimates.get(cam)
        if estimate is None:
            continue

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        roi = bgr[estimate.crop_top : estimate.crop_bottom, :, :]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        image_tensor = F.to_tensor(Image.fromarray(roi_rgb)).to(device)

        with torch.no_grad():
            output = model([image_tensor])[0]
        best_pred = select_best_prediction(output, args.score_threshold)

        model_bbox_full = None
        heuristic_bbox_full = None
        heuristic_trust = None
        crop_source = "model"

        # Prefer the detector when it returns a confident box; otherwise fall back to the heuristic scorer.
        if best_pred is not None:
            x1, y1, x2, y2 = best_pred["box_xyxy"]
            model_bbox_full = (
                int(round(x1)),
                int(round(y1 + estimate.crop_top)),
                int(round(x2 - x1)),
                int(round(y2 - y1)),
            )
            center_x = (x1 + x2) / 2.0
            crop_left = int(round(center_x - args.crop_width / 2.0))
            crop_left = max(0, min(crop_left, gray.shape[1] - args.crop_width))
        else:
            crop_source = "heuristic"
            gray_crop = gray[estimate.crop_top : estimate.crop_bottom, :]
            band_mask = bright_band_mask(gray_crop, estimate)
            heuristic_bbox, score_map, component_meta = defect_component(
                gray_crop, band_mask, references.get(cam)
            )
            crop_left = crop_x_from_component(
                heuristic_bbox,
                score_map,
                args.crop_width,
                gray.shape[1],
                component_meta,
            )
            if heuristic_bbox is not None:
                x, y, w, h = heuristic_bbox
                heuristic_bbox_full = (x, y + estimate.crop_top, w, h)
            heuristic_trust = (
                None if component_meta is None else float(component_meta["trust_score"])
            )

        crop = bgr[
            estimate.crop_top : estimate.crop_bottom,
            crop_left : crop_left + args.crop_width,
        ]
        cv2.imwrite(str(images_dir / image_path.name), crop)

        # Store enough metadata to audit which path produced each crop and why.
        metadata_rows.append(
            {
                "file_name": image_path.name,
                "camera": cam,
                "crop_left": crop_left,
                "crop_top": estimate.crop_top,
                "crop_width": args.crop_width,
                "crop_height": args.crop_height,
                "source": crop_source,
                "model_score": "" if best_pred is None else round(best_pred["score"], 4),
                "model_bbox_xywh": ""
                if model_bbox_full is None
                else ",".join(map(str, model_bbox_full)),
                "heuristic_bbox_xywh": ""
                if heuristic_bbox_full is None
                else ",".join(map(str, heuristic_bbox_full)),
                "heuristic_trust": ""
                if heuristic_trust is None
                else round(heuristic_trust, 4),
            }
        )

        if args.debug_overlays:
            debug = overlay_debug(
                bgr,
                estimate,
                crop_left,
                args.crop_width,
                model_bbox_full,
                heuristic_bbox_full,
            )
            cv2.imwrite(str(debug_dir / image_path.name), debug)

        if index % 200 == 0:
            print(f"processed {index}/{len(image_paths)} images")

    with (output_dir / "crop_metadata.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(metadata_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"saved {len(metadata_rows)} cropped images to {images_dir}")


if __name__ == "__main__":
    main()
