import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.transforms import functional as F

from auto_crop_glass_defect import camera_name, estimate_camera_bands, find_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a single-class detector for glass defects on the fixed-band ROI."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--sample-per-camera", type=int, default=80)
    parser.add_argument("--crop-height", type=int, default=900)
    parser.add_argument("--min-size", type=int, default=800)
    parser.add_argument("--max-size", type=int, default=2048)
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260315)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Optional checkpoint path to resume from.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def read_split_file(path: Path) -> list[Path]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [Path(line) for line in lines if line.strip()]


def yolo_to_xyxy(
    line: str, image_width: int, image_height: int
) -> tuple[float, float, float, float] | None:
    """Parse one YOLO label line into absolute full-image XYXY coordinates.

    Args:
        line: A single YOLO label line.
        image_width: Full image width.
        image_height: Full image height.

    Returns:
        An absolute `xyxy` box, or `None` if the line is malformed.
    """

    parts = line.split()
    if len(parts) != 5:
        return None
    _, x_center, y_center, width, height = map(float, parts)
    box_w = width * image_width
    box_h = height * image_height
    center_x = x_center * image_width
    center_y = y_center * image_height
    x1 = center_x - box_w / 2.0
    y1 = center_y - box_h / 2.0
    x2 = center_x + box_w / 2.0
    y2 = center_y + box_h / 2.0
    return x1, y1, x2, y2


def clip_box_to_roi(
    box: tuple[float, float, float, float], crop_top: int, crop_height: int, image_width: int
) -> tuple[float, float, float, float] | None:
    """Clip a full-image box to the fixed vertical ROI and shift it to ROI-local coordinates.

    Args:
        box: Input `xyxy` box in full-image coordinates.
        crop_top: Top row of the fixed ROI in full-image coordinates.
        crop_height: Height of the fixed ROI.
        image_width: Full image width.

    Returns:
        An ROI-local `xyxy` box, or `None` if clipping removes the box.
    """

    x1, y1, x2, y2 = box
    roi_y1 = float(crop_top)
    roi_y2 = float(crop_top + crop_height)
    clipped_x1 = max(0.0, min(float(image_width), x1))
    clipped_x2 = max(0.0, min(float(image_width), x2))
    clipped_y1 = max(roi_y1, min(roi_y2, y1))
    clipped_y2 = max(roi_y1, min(roi_y2, y2))
    if clipped_x2 - clipped_x1 < 2 or clipped_y2 - clipped_y1 < 2:
        return None
    # Shift labels into ROI-local coordinates because the detector never sees the full-height image.
    return clipped_x1, clipped_y1 - crop_top, clipped_x2, clipped_y2 - crop_top


class GlassROIDataset(Dataset):
    """Torch dataset that serves ROI-cropped images and ROI-local defect boxes.

    Args:
        image_paths: Images assigned to this dataset split.
        label_dir: Directory containing YOLO label files.
        estimates: Per-camera ROI geometry used to crop each image before training.
    """

    def __init__(
        self,
        image_paths: list[Path],
        label_dir: Path,
        estimates: dict[str, object],
    ) -> None:
        self.image_paths = image_paths
        self.label_dir = label_dir
        self.estimates = estimates

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        cam = camera_name(image_path)
        estimate = self.estimates[cam]

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Training runs on the stable bright-band ROI only, not on the full 2448x2048 frame.
        roi = image[estimate.crop_top : estimate.crop_bottom, :, :]
        roi_height, roi_width = roi.shape[:2]

        boxes = []
        label_path = self.label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                box = yolo_to_xyxy(line, image.shape[1], image.shape[0])
                if box is None:
                    continue
                clipped = clip_box_to_roi(
                    box,
                    crop_top=estimate.crop_top,
                    crop_height=estimate.crop_bottom - estimate.crop_top,
                    image_width=image.shape[1],
                )
                if clipped is not None:
                    boxes.append(clipped)

        boxes_tensor = (
            torch.as_tensor(boxes, dtype=torch.float32)
            if boxes
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        labels_tensor = torch.ones((boxes_tensor.shape[0],), dtype=torch.int64)
        area_tensor = (
            (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])
            if boxes_tensor.numel()
            else torch.zeros((0,), dtype=torch.float32)
        )
        iscrowd_tensor = torch.zeros((boxes_tensor.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": area_tensor,
            "iscrowd": iscrowd_tensor,
            "file_name": image_path.name,
            "camera": cam,
        }

        image_tensor = F.to_tensor(Image.fromarray(roi))
        return image_tensor, target


def make_model(min_size: int, max_size: int) -> torch.nn.Module:
    """Construct the single-class Faster R-CNN detector used by this project.

    Args:
        min_size: Minimum image size argument passed to the detector.
        max_size: Maximum image size argument passed to the detector.

    Returns:
        A Faster R-CNN model configured for one foreground defect class.
    """

    # num_classes includes background, so a single defect class becomes 2 here.
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=2,
        min_size=min_size,
        max_size=max_size,
    )
    return model


@torch.no_grad()
def evaluate_map50(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    score_threshold: float,
) -> dict[str, float]:
    """Compute a lightweight IoU@0.5 precision/recall/F1 summary for validation.

    Args:
        model: Detector to evaluate.
        dataloader: Validation dataloader.
        device: Torch device used for inference.
        score_threshold: Minimum score required for a prediction to count.

    Returns:
        A dictionary containing precision, recall, F1, and TP/FP/FN counts.
    """

    # This is a lightweight project metric, not a full COCO-style evaluation pipeline.
    model.eval()
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for images, targets in dataloader:
        images = [image.to(device) for image in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            gt_boxes = target["boxes"].to(device)
            pred_scores = output["scores"]
            keep = pred_scores >= score_threshold
            pred_boxes = output["boxes"][keep]

            if pred_boxes.numel() == 0 and gt_boxes.numel() == 0:
                continue
            if pred_boxes.numel() == 0:
                total_fn += gt_boxes.shape[0]
                continue
            if gt_boxes.numel() == 0:
                total_fp += pred_boxes.shape[0]
                continue

            matched_gt = set()
            for pred_box in pred_boxes:
                ious = box_iou(pred_box.unsqueeze(0), gt_boxes).squeeze(0)
                best_iou, best_idx = torch.max(ious, dim=0)
                best_idx_value = int(best_idx.item())
                if best_iou.item() >= 0.5 and best_idx_value not in matched_gt:
                    matched_gt.add(best_idx_value)
                    total_tp += 1
                else:
                    total_fp += 1
            total_fn += gt_boxes.shape[0] - len(matched_gt)

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(total_tp),
        "fp": float(total_fp),
        "fn": float(total_fn),
    }


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    """Run one training epoch and return averaged torchvision loss components.

    Args:
        model: Detector being optimized.
        dataloader: Training dataloader.
        optimizer: Optimizer used for parameter updates.
        scaler: Optional AMP gradient scaler.
        device: Torch device used for training.
        epoch: One-based epoch index used for logging.

    Returns:
        A dictionary of averaged loss components for the completed epoch.
    """

    model.train()
    running = {"loss": 0.0, "loss_classifier": 0.0, "loss_box_reg": 0.0, "loss_objectness": 0.0, "loss_rpn_box_reg": 0.0}
    num_batches = 0

    for step, (images, targets) in enumerate(dataloader, start=1):
        images = [image.to(device) for image in images]
        targets = [
            {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in target.items()}
            for target in targets
        ]

        optimizer.zero_grad(set_to_none=True)
        use_amp = scaler is not None
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())
        if not torch.isfinite(losses):
            # Bad batches are skipped so a single numerical issue does not abort a long run.
            print(f"epoch {epoch} step {step}: non-finite loss, batch skipped")
            optimizer.zero_grad(set_to_none=True)
            continue

        if use_amp:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        num_batches += 1
        running["loss"] += float(losses.item())
        for key in ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"):
            running[key] += float(loss_dict[key].item())

        if step % 20 == 0:
            print(
                f"epoch {epoch} step {step}/{len(dataloader)} "
                f"loss={running['loss'] / num_batches:.4f}"
            )

    return {key: value / max(num_batches, 1) for key, value in running.items()}


@torch.no_grad()
def compute_val_loss(
    model: torch.nn.Module, dataloader: DataLoader, device: torch.device
) -> dict[str, float]:
    """Measure validation losses using the detector's training-mode loss path.

    Args:
        model: Detector being evaluated.
        dataloader: Validation dataloader.
        device: Torch device used for evaluation.

    Returns:
        A dictionary of averaged validation loss components.
    """

    # Torchvision detection models emit losses only in train mode, so validation temporarily switches back.
    model.train()
    running = {"loss": 0.0, "loss_classifier": 0.0, "loss_box_reg": 0.0, "loss_objectness": 0.0, "loss_rpn_box_reg": 0.0}
    num_batches = 0

    for images, targets in dataloader:
        images = [image.to(device) for image in images]
        targets = [
            {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in target.items()}
            for target in targets
        ]
        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())
        if not torch.isfinite(losses):
            continue
        num_batches += 1
        running["loss"] += float(losses.item())
        for key in ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"):
            running[key] += float(loss_dict[key].item())

    model.eval()
    return {key: value / max(num_batches, 1) for key, value in running.items()}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    dataset_dir = args.dataset_dir.resolve()
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_paths = read_split_file(dataset_dir / "train.txt")
    val_paths = read_split_file(dataset_dir / "val.txt")
    all_raw_images = find_images(raw_dir)
    # Persist the same camera ROI estimates used in training so inference crops the identical vertical region.
    estimates = estimate_camera_bands(all_raw_images, args.crop_height, args.sample_per_camera)

    train_dataset = GlassROIDataset(train_paths, dataset_dir / "labels", estimates)
    val_dataset = GlassROIDataset(val_paths, dataset_dir / "labels", estimates)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_model(min_size=args.min_size, max_size=args.max_size).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == "cuda"))

    start_epoch = 1
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1

    config = {
        "dataset_dir": str(dataset_dir),
        "raw_dir": str(raw_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "sample_per_camera": args.sample_per_camera,
        "crop_height": args.crop_height,
        "min_size": args.min_size,
        "max_size": args.max_size,
        "score_threshold": args.score_threshold,
        "seed": args.seed,
        "device": str(device),
        "resume": str(args.resume) if args.resume is not None else None,
        "train_images": len(train_dataset),
        "val_images": len(val_dataset),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (output_dir / "camera_band_estimates.json").write_text(
        json.dumps({cam: asdict(estimate) for cam, estimate in estimates.items()}, indent=2),
        encoding="utf-8",
    )

    best_f1 = -1.0
    history_rows: list[dict[str, float | int]] = []

    history_path = output_dir / "history.csv"
    if args.resume is not None and history_path.exists():
        with history_path.open("r", newline="", encoding="utf-8") as fh:
            history_rows = list(csv.DictReader(fh))
            history_rows = [
                {k: (float(v) if k != "epoch" else int(float(v))) for k, v in row.items()}
                for row in history_rows
            ]
        if history_rows:
            best_f1 = max(float(row["f1"]) for row in history_rows)

    start_time = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch)
        val_loss_metrics = compute_val_loss(model, val_loader, device)
        eval_metrics = evaluate_map50(model, val_loader, device, args.score_threshold)

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_loss_metrics.items()},
            **eval_metrics,
        }
        history_rows.append(row)
        scheduler.step()

        print(
            f"epoch {epoch}: train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_loss_metrics['loss']:.4f} "
            f"precision={eval_metrics['precision']:.4f} "
            f"recall={eval_metrics['recall']:.4f} "
            f"f1={eval_metrics['f1']:.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": config,
        }
        torch.save(checkpoint, output_dir / "last.pt")

        if eval_metrics["f1"] > best_f1:
            best_f1 = eval_metrics["f1"]
            torch.save(checkpoint, output_dir / "best.pt")

        with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(history_rows[0].keys()))
            writer.writeheader()
            writer.writerows(history_rows)

    elapsed = time.time() - start_time
    summary = {
        "best_f1": best_f1,
        "elapsed_seconds": elapsed,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"training finished in {elapsed / 60.0:.2f} minutes, best_f1={best_f1:.4f}")


if __name__ == "__main__":
    main()
