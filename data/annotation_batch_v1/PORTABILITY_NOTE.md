# Portability Note

This annotation batch was copied from the original working directory into the open-source repository.

The following files were normalized for the released layout:

- `dataset.yaml`
- `train.txt`
- `val.txt`

These files now point to paths that are valid inside this repository.

`manifest.csv` is different. It is preserved mainly as historical review metadata, so it may still contain absolute paths from the original workspace. Treat those path fields as reference-only metadata rather than portable inputs.
