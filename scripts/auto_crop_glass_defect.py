import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


CAMERA_RE = re.compile(r"^(cam-?\d+)_", re.IGNORECASE)


@dataclass
class BandEstimate:
    """Per-camera bright-band geometry and the derived fixed vertical crop window.

    Attributes:
        camera: Camera name parsed from the filename prefix.
        image_height: Full-frame image height for this camera.
        image_width: Full-frame image width for this camera.
        band1_start: Top row of the first bright band in full-image coordinates.
        band1_end: Bottom row of the first bright band in full-image coordinates.
        band2_start: Top row of the second bright band in full-image coordinates.
        band2_end: Bottom row of the second bright band in full-image coordinates.
        crop_top: Top row of the fixed vertical crop window.
        crop_bottom: Bottom row of the fixed vertical crop window.
        sample_count: Number of frames that contributed valid band detections.
    """

    camera: str
    image_height: int
    image_width: int
    band1_start: int
    band1_end: int
    band2_start: int
    band2_end: int
    crop_top: int
    crop_bottom: int
    sample_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-crop glass defect images to a fixed 1200x900 window."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory that contains the raw .jpg images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for cropped images and metadata.",
    )
    parser.add_argument(
        "--crop-width",
        type=int,
        default=1200,
        help="Crop width in pixels.",
    )
    parser.add_argument(
        "--crop-height",
        type=int,
        default=900,
        help="Crop height in pixels.",
    )
    parser.add_argument(
        "--sample-per-camera",
        type=int,
        default=80,
        help="Sample count used to estimate each camera's bright-band ROI.",
    )
    parser.add_argument(
        "--reference-per-camera",
        type=int,
        default=31,
        help="Sample count used to build each camera's median reference crop. Set 0 to disable.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional image limit for debugging.",
    )
    parser.add_argument(
        "--debug-overlays",
        action="store_true",
        help="Save crop boxes and detected components as debug overlays.",
    )
    return parser.parse_args()


def find_images(input_dir: Path) -> list[Path]:
    files = [p for p in input_dir.rglob("*.jpg") if p.is_file()]
    return sorted(files)


def camera_name(path: Path) -> str:
    match = CAMERA_RE.match(path.name)
    return match.group(1) if match else "unknown"


def longest_segments(mask: np.ndarray, min_len: int = 20) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(mask.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            if idx - start >= min_len:
                segments.append((start, idx - 1))
            start = None
    if start is not None and len(mask) - start >= min_len:
        segments.append((start, len(mask) - 1))
    segments.sort(key=lambda item: item[1] - item[0], reverse=True)
    return segments


def detect_bands(gray: np.ndarray) -> list[tuple[int, int]]:
    """Locate the two dominant bright horizontal bands in a grayscale frame.

    Args:
        gray: Grayscale input image in full-image coordinates.

    Returns:
        A list of up to two `(start_row, end_row)` segments sorted from top to bottom.
    """

    row_mean = gray.mean(axis=1)
    threshold = row_mean.mean() + 0.55 * row_mean.std()
    mask = row_mean > threshold
    segments = longest_segments(mask, min_len=max(20, gray.shape[0] // 80))
    if len(segments) >= 2:
        return sorted(segments[:2])

    smooth = cv2.GaussianBlur(row_mean.astype(np.float32), (1, 0), sigmaX=0, sigmaY=15)
    fallback_threshold = np.percentile(smooth, 75)
    segments = longest_segments(smooth > fallback_threshold, min_len=max(20, gray.shape[0] // 80))
    return sorted(segments[:2])


def estimate_camera_bands(
    image_paths: list[Path], crop_height: int, sample_per_camera: int
) -> dict[str, BandEstimate]:
    """Estimate one stable bright-band ROI per camera from a small image sample.

    Args:
        image_paths: Full list of raw image paths.
        crop_height: Target crop height used to derive the vertical ROI.
        sample_per_camera: Number of early frames per camera used for estimation.

    Returns:
        A mapping from camera name to its median bright-band estimate.
    """

    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in image_paths:
        grouped[camera_name(path)].append(path)

    estimates: dict[str, BandEstimate] = {}
    for cam, paths in grouped.items():
        starts_1: list[int] = []
        ends_1: list[int] = []
        starts_2: list[int] = []
        ends_2: list[int] = []
        image_height = None
        image_width = None

        for path in paths[:sample_per_camera]:
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            image_height, image_width = gray.shape[:2]
            bands = detect_bands(gray)
            if len(bands) != 2:
                continue
            (s1, e1), (s2, e2) = bands
            starts_1.append(s1)
            ends_1.append(e1)
            starts_2.append(s2)
            ends_2.append(e2)

        if not starts_1 or image_height is None or image_width is None:
            continue

        band1_start = int(np.median(starts_1))
        band1_end = int(np.median(ends_1))
        band2_start = int(np.median(starts_2))
        band2_end = int(np.median(ends_2))
        center_y = (band1_start + band2_end) / 2.0
        crop_top = int(round(center_y - crop_height / 2.0))
        crop_top = max(0, min(crop_top, image_height - crop_height))
        crop_bottom = crop_top + crop_height

        estimates[cam] = BandEstimate(
            camera=cam,
            image_height=image_height,
            image_width=image_width,
            band1_start=band1_start,
            band1_end=band1_end,
            band2_start=band2_start,
            band2_end=band2_end,
            crop_top=crop_top,
            crop_bottom=crop_bottom,
            sample_count=len(starts_1),
        )

    return estimates


def build_camera_references(
    image_paths: list[Path], estimates: dict[str, BandEstimate], reference_per_camera: int
) -> dict[str, np.ndarray]:
    """Build median ROI reference images that suppress camera-stationary structures.

    Args:
        image_paths: Full list of raw image paths.
        estimates: Per-camera bright-band and crop geometry.
        reference_per_camera: Number of frames to sample per camera when building the median.

    Returns:
        A mapping from camera name to a float32 median ROI image.
    """

    if reference_per_camera <= 0:
        return {}

    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in image_paths:
        cam = camera_name(path)
        if cam in estimates:
            grouped[cam].append(path)

    references: dict[str, np.ndarray] = {}
    for cam, paths in grouped.items():
        estimate = estimates[cam]
        if reference_per_camera >= len(paths):
            sampled_paths = paths
        else:
            indices = np.linspace(0, len(paths) - 1, num=reference_per_camera, dtype=int)
            sampled_paths = [paths[index] for index in sorted(set(indices.tolist()))]

        stack: list[np.ndarray] = []
        for path in sampled_paths:
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            # The per-camera median reference suppresses fixed fixtures and keeps transient defects salient.
            stack.append(gray[estimate.crop_top : estimate.crop_bottom, :])

        if stack:
            references[cam] = np.median(np.stack(stack, axis=0), axis=0).astype(np.float32)

    return references


def bright_band_mask(gray: np.ndarray, estimate: BandEstimate, trim: int = 12) -> np.ndarray:
    """Create a mask for the two bright bands inside the ROI-local coordinate frame.

    Args:
        gray: ROI image whose shape defines the output mask.
        estimate: Per-camera band estimate in full-image coordinates.
        trim: Margin removed from each band edge to reduce boundary artifacts.

    Returns:
        A uint8 mask with the bright-band interiors set to 255.
    """

    mask = np.zeros_like(gray, dtype=np.uint8)
    for start, end in (
        (estimate.band1_start, estimate.band1_end),
        (estimate.band2_start, estimate.band2_end),
    ):
        start = max(0, start - estimate.crop_top + trim)
        end = min(gray.shape[0] - 1, end - estimate.crop_top - trim)
        if end > start:
            mask[start : end + 1, :] = 255
    return mask


def bright_band_distance_weight(bright_mask: np.ndarray) -> np.ndarray:
    band_binary = (bright_mask > 0).astype(np.uint8)
    if not np.any(band_binary):
        return np.ones(bright_mask.shape, dtype=np.float32)

    # Favor responses that stay inside the bright bands instead of hugging their top/bottom edges.
    distance = cv2.distanceTransform(band_binary, cv2.DIST_L2, 3)
    positive = distance[band_binary > 0]
    scale = float(np.percentile(positive, 95)) if positive.size else 1.0
    scale = max(scale, 1.0)
    normalized = np.clip(distance / scale, 0.0, 1.0)
    return (0.35 + 0.65 * normalized).astype(np.float32)


def component_angle_degrees(xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.size < 5:
        return 90.0

    points = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    vx, vy, _, _ = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
    angle = np.degrees(np.arctan2(float(vy[0]), float(vx[0])))
    return abs(float(angle))


def defect_component(
    gray_crop: np.ndarray, bright_mask: np.ndarray, reference_crop: np.ndarray | None = None
) -> tuple[tuple[int, int, int, int] | None, np.ndarray, dict[str, float] | None]:
    """Find the most defect-like connected component inside the ROI.

    Args:
        gray_crop: Grayscale ROI image in ROI-local coordinates.
        bright_mask: ROI-local mask limiting the search to the bright bands.
        reference_crop: Optional per-camera median ROI used to suppress static fixtures.

    Returns:
        A tuple of `(bbox, score_map, metadata)` where `bbox` is the best component in
        ROI-local `xywh` format, `score_map` is the dense heuristic response map, and
        `metadata` stores shape and trust information used by later crop selection.
    """

    roi_u8 = gray_crop if gray_crop.dtype == np.uint8 else gray_crop.astype(np.uint8)
    roi = roi_u8.astype(np.float32)
    local_background = cv2.GaussianBlur(roi, (0, 0), sigmaX=13, sigmaY=13)
    local_dark = np.clip(local_background - roi, 0, None)
    blackhat = cv2.morphologyEx(
        roi_u8,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
    ).astype(np.float32)
    reference_dark = (
        np.clip(reference_crop.astype(np.float32) - roi, 0, None)
        if reference_crop is not None
        else np.zeros_like(roi)
    )

    grad_x = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    gradient = cv2.GaussianBlur(gradient, (0, 0), sigmaX=1.2, sigmaY=1.2)

    interior_weight = bright_band_distance_weight(bright_mask)
    band_mask = (bright_mask > 0).astype(np.float32)
    # Blend reference subtraction, local dark contrast, and edges so both chips and crack traces can score.
    score = (
        0.45 * reference_dark
        + 0.30 * blackhat
        + 0.15 * local_dark
        + 0.10 * gradient
    )
    score *= band_mask
    score *= interior_weight
    positive = score[bright_mask > 0]
    if positive.size == 0:
        return None, score, None

    threshold = max(np.percentile(positive, 99.2), positive.mean() + 1.8 * positive.std())
    binary = np.zeros_like(score, dtype=np.uint8)
    binary[score >= threshold] = 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary = cv2.dilate(binary, np.ones((5, 5), np.uint8), iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    best_bbox = None
    best_score = -1.0
    best_meta = None

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < 12:
            continue
        if w > int(0.92 * gray_crop.shape[1]) and h < 80:
            continue

        component_mask = labels == label
        ys, xs = np.where(component_mask)
        if xs.size == 0:
            continue

        raw_score = float(score[component_mask].sum())
        aspect_ratio = w / max(h, 1)
        angle = component_angle_degrees(xs, ys)
        # Roller edges and band boundaries are often long and nearly horizontal, so penalize that shape.
        orientation_weight = max(0.25, float(np.sin(np.deg2rad(max(angle, 1.0)))))
        height_weight = float(np.clip(h / 120.0, 0.45, 1.25))
        boundary_weight = float(np.clip(interior_weight[component_mask].mean(), 0.35, 1.0))
        fill_ratio = area / max(w * h, 1)
        fill_weight = 0.80 + 0.40 * fill_ratio

        if angle < 8 and h < 90:
            orientation_weight *= 0.28
        elif angle < 15 and aspect_ratio > 6:
            orientation_weight *= 0.45

        if aspect_ratio > 24 and h < 80:
            continue

        component_score = raw_score * orientation_weight * height_weight * boundary_weight * fill_weight
        trust_score = orientation_weight * height_weight * boundary_weight

        if component_score > best_score:
            best_score = component_score
            best_bbox = (x, y, w, h)
            best_meta = {
                "angle_degrees": angle,
                "aspect_ratio": aspect_ratio,
                "height": float(h),
                "raw_score": raw_score,
                "trust_score": trust_score,
            }

    return best_bbox, score, best_meta


def crop_x_from_component(
    bbox: tuple[int, int, int, int] | None,
    score_map: np.ndarray,
    crop_width: int,
    image_width: int,
    component_meta: dict[str, float] | None = None,
) -> int:
    """Choose the horizontal crop origin from the best component or score-map centroid.

    Args:
        bbox: Best heuristic component in ROI-local `xywh` format, if one exists.
        score_map: Dense heuristic response map for the ROI.
        crop_width: Desired crop width in pixels.
        image_width: Full image width in pixels.
        component_meta: Optional trust metadata for the best component.

    Returns:
        The clamped left edge of the final horizontal crop in full-image coordinates.
    """

    use_component_center = (
        bbox is not None
        and component_meta is not None
        and component_meta.get("trust_score", 0.0) >= 0.33
    )
    if use_component_center:
        x, _, w, _ = bbox
        center_x = x + w / 2.0
    else:
        # When the best component is weak, use the full score map center of mass as a softer fallback.
        col_score = score_map.sum(axis=0)
        if float(col_score.sum()) <= 0:
            center_x = image_width / 2.0
        else:
            coords = np.arange(len(col_score), dtype=np.float32)
            center_x = float((coords * col_score).sum() / col_score.sum())

    crop_left = int(round(center_x - crop_width / 2.0))
    crop_left = max(0, min(crop_left, image_width - crop_width))
    return crop_left


def overlay_debug(
    original_bgr: np.ndarray,
    estimate: BandEstimate,
    crop_left: int,
    crop_width: int,
    bbox: tuple[int, int, int, int] | None,
) -> np.ndarray:
    debug = original_bgr.copy()
    cv2.rectangle(
        debug,
        (crop_left, estimate.crop_top),
        (crop_left + crop_width, estimate.crop_bottom),
        (0, 255, 0),
        4,
    )
    for start, end in (
        (estimate.band1_start, estimate.band1_end),
        (estimate.band2_start, estimate.band2_end),
    ):
        cv2.rectangle(debug, (0, start), (debug.shape[1] - 1, end), (255, 128, 0), 2)

    if bbox is not None:
        x, y, w, h = bbox
        cv2.rectangle(
            debug,
            (x, y + estimate.crop_top),
            (x + w, y + h + estimate.crop_top),
            (0, 0, 255),
            2,
        )
    return debug


def ensure_crop_shape(image: np.ndarray, crop_width: int, crop_height: int) -> np.ndarray:
    """Guarantee a fixed output size even when boundary clipping shrinks the raw slice.

    Args:
        image: Cropped image patch.
        crop_width: Desired output width.
        crop_height: Desired output height.

    Returns:
        The original crop if it already matches the target size, otherwise a resized copy.
    """

    if image.shape[1] == crop_width and image.shape[0] == crop_height:
        return image
    return cv2.resize(image, (crop_width, crop_height), interpolation=cv2.INTER_AREA)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    crop_width = args.crop_width
    crop_height = args.crop_height

    image_paths = find_images(input_dir)
    if args.limit > 0:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise SystemExit(f"No .jpg files found under {input_dir}")

    # Camera geometry is stable, so estimate one vertical ROI per camera and reuse it for the full run.
    estimates = estimate_camera_bands(image_paths, crop_height, args.sample_per_camera)
    if not estimates:
        raise SystemExit("Failed to estimate bright-band ROI for any camera.")
    references = build_camera_references(image_paths, estimates, args.reference_per_camera)

    crop_dir = output_dir / "images"
    debug_dir = output_dir / "debug"
    crop_dir.mkdir(parents=True, exist_ok=True)
    if args.debug_overlays:
        debug_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows: list[dict[str, object]] = []
    for index, path in enumerate(image_paths, start=1):
        cam = camera_name(path)
        estimate = estimates.get(cam)
        if estimate is None:
            continue

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray_crop = gray[estimate.crop_top : estimate.crop_bottom, :]
        bright_mask = bright_band_mask(gray_crop, estimate)
        bbox, score_map, component_meta = defect_component(
            gray_crop, bright_mask, references.get(cam)
        )
        crop_left = crop_x_from_component(
            bbox, score_map, crop_width, gray.shape[1], component_meta
        )

        crop = bgr[
            estimate.crop_top : estimate.crop_bottom,
            crop_left : crop_left + crop_width,
        ]
        crop = ensure_crop_shape(crop, crop_width, crop_height)

        out_path = crop_dir / path.name
        cv2.imwrite(str(out_path), crop)

        metadata_rows.append(
            {
                "file_name": path.name,
                "camera": cam,
                "crop_left": crop_left,
                "crop_top": estimate.crop_top,
                "crop_width": crop_width,
                "crop_height": crop_height,
                "component_bbox": "" if bbox is None else ",".join(map(str, bbox)),
                "component_angle_degrees": ""
                if component_meta is None
                else round(component_meta["angle_degrees"], 2),
                "component_trust_score": ""
                if component_meta is None
                else round(component_meta["trust_score"], 4),
                "score_sum": float(score_map.sum()),
            }
        )

        if args.debug_overlays:
            debug = overlay_debug(bgr, estimate, crop_left, crop_width, bbox)
            cv2.imwrite(str(debug_dir / path.name), debug)

        if index % 500 == 0:
            print(f"processed {index}/{len(image_paths)} images")

    with (output_dir / "crop_metadata.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(metadata_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_rows)

    with (output_dir / "camera_band_estimates.json").open("w", encoding="utf-8") as fh:
        json.dump({cam: asdict(estimate) for cam, estimate in estimates.items()}, fh, indent=2)

    print(f"saved {len(metadata_rows)} cropped images to {crop_dir}")


if __name__ == "__main__":
    main()
