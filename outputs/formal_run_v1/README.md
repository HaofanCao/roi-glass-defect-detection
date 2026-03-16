# Formal Run Output

This folder contains the released example artifacts from one formal crop run.

## Included files

- `crop_metadata.csv`: metadata exported during the released formal run
- `sample_crops/`: a small subset of final `1200x900` crops for quick inspection

## Relation to generated outputs

This repository does not bundle the full generated crop dataset. The recommended location for a new end-to-end run is:

```text
outputs/formal_run_v1_generated/
```

That generated folder is separate from `outputs/formal_run_v1/`, which is kept as a compact reference snapshot for the repository.

Generate the full formal output set with:

```powershell
python scripts/crop_with_glass_detector.py `
  --checkpoint .\models\glass_detector_v1\best.pt `
  --input-dir .\data\raw_input\raw_data_glass_defect `
  --output-dir .\outputs\formal_run_v1_generated
```
