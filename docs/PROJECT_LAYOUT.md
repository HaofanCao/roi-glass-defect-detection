# Project Layout

This document is a quick structural reference for the open-source bundle.

## Top-level layout

```text
glass_defect_detection/
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- .gitattributes
|-- docs/
|-- scripts/
|-- data/
|-- models/
|-- outputs/
`-- examples/
```

## Directory guide

### `docs/`

Repository-facing documentation and visual assets.

- `docs/assets/pipeline-overview.svg`: the main README pipeline figure
- `docs/assets/origin_example.jpg`: raw-image example used in the README walkthrough
- `docs/assets/detect_example.jpg`: localization overlay example used in the README walkthrough
- `docs/assets/final-crop_example.jpg`: final crop example used in the README walkthrough
- `docs/PIPELINE.md`: command-oriented end-to-end workflow
- `docs/PROJECT_LAYOUT.md`: this structure reference

### `scripts/`

Core executable pipeline code.

- `scripts/auto_crop_glass_defect.py`: heuristic baseline cropper and ROI estimator
- `scripts/prepare_glass_detection_annotation.py`: annotation-batch exporter with heuristic suggestions
- `scripts/train_glass_defect_detector.py`: ROI detector training entry point
- `scripts/crop_with_glass_detector.py`: formal detector-first cropper with heuristic fallback

### `data/`

Input-facing project data and annotation assets.

- `data/raw_input/README.md`: placeholder note for the raw dataset location
- `data/raw_input/raw_data_glass_defect/`: expected location of the full raw dataset, not bundled in the repository
- `data/annotation_batch_v1/`: reviewed annotation batch used for detector training

Inside `data/annotation_batch_v1/`:

- `README.txt`: original batch note exported during project preparation
- `PORTABILITY_NOTE.md`: release note about path portability and historical metadata
- `dataset.yaml`: dataset config used for training
- `train.txt`, `val.txt`: split files normalized for the open-source layout
- `manifest.csv`: annotation-batch metadata and review notes
- `images/`: sampled raw images for annotation
- `labels/`: corrected single-class YOLO-format labels
- `overlays/`: visual annotation aids

### `models/`

Released detector checkpoint and training artifacts.

- `models/glass_detector_v1/best.pt`: trained detector checkpoint
- `models/glass_detector_v1/config.json`: saved training configuration
- `models/glass_detector_v1/camera_band_estimates.json`: per-camera ROI geometry reused at inference time
- `models/glass_detector_v1/history.csv`: epoch-level training log
- `models/glass_detector_v1/summary.json`: compact run summary

### `outputs/`

Included example outputs from the formal crop stage.

- `outputs/formal_run_v1/README.md`: notes about the bundled output folder
- `outputs/formal_run_v1/crop_metadata.csv`: metadata from the released formal run
- `outputs/formal_run_v1/sample_crops/`: small sample of final `1200x900` crops

### `examples/`

Very small raw-image examples for documentation and quick inspection.

- `examples/raw_samples/`: representative raw frames only, not the full dataset

## What is intentionally excluded

This repository does not bundle:

- the full raw production dataset
- the full generated crop dataset
- intermediate smoke-test folders
- earlier audit rounds and temporary experiment outputs
