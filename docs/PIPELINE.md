# Pipeline

Run commands from the repository root.

## Directory map

- `data/raw_input/raw_data_glass_defect/`: expected location of the full raw dataset
- `data/annotation_batch_v1/`: reviewed annotation batch used for training
- `models/glass_detector_v1/`: released checkpoint and training artifacts
- `outputs/formal_run_v1/`: bundled example output and metadata
- `outputs/formal_run_v1_generated/`: suggested location for newly generated formal-run outputs

## 1. Place raw data

Put the full raw image folder at:

```text
data/raw_input/raw_data_glass_defect/
```

The scripts expect `.jpg` images and the original dataset naming style such as:

```text
cam-1_ts1763100100504.jpg
cam1_ts1763536491547.jpg
cam2_ts1764986728565.jpg
```

## 2. Heuristic baseline crop

```powershell
python scripts/auto_crop_glass_defect.py `
  --input-dir .\data\raw_input\raw_data_glass_defect `
  --output-dir .\outputs\heuristic_baseline `
  --debug-overlays
```

## 3. Prepare annotation batch

```powershell
python scripts/prepare_glass_detection_annotation.py `
  --input-dir .\data\raw_input\raw_data_glass_defect `
  --output-dir .\data\annotation_batch_v1 `
  --samples-per-camera 60
```

## 4. Manual correction

Correct YOLO labels under:

```text
data/annotation_batch_v1/labels/
```

Reference files:
- `data/annotation_batch_v1/images/`
- `data/annotation_batch_v1/overlays/`
- `data/annotation_batch_v1/manifest.csv`

## 5. Train detector

```powershell
python scripts/train_glass_defect_detector.py `
  --dataset-dir .\data\annotation_batch_v1 `
  --raw-dir .\data\raw_input\raw_data_glass_defect `
  --output-dir .\models\glass_detector_v1 `
  --epochs 20 `
  --batch-size 2 `
  --num-workers 0
```

## 6. Final formal crop

```powershell
python scripts/crop_with_glass_detector.py `
  --checkpoint .\models\glass_detector_v1\best.pt `
  --input-dir .\data\raw_input\raw_data_glass_defect `
  --output-dir .\outputs\formal_run_v1_generated
```

## 7. Inspect results

- Released training history: `models/glass_detector_v1/history.csv`
- Released formal-run metadata: `outputs/formal_run_v1/crop_metadata.csv`
- Released sample crops: `outputs/formal_run_v1/sample_crops/`
- Newly generated outputs: `outputs/formal_run_v1_generated/`

For a quick repository-level structure reference, see `docs/PROJECT_LAYOUT.md`.
