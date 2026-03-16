Annotation batch for single-class defect detection.

Folders:
- images/: original images
- labels/: YOLO-format suggestions
- overlays/: original images with bright bands, crop window and suggested boxes

Colors in overlays:
- blue: estimated bright bands
- green: suggested 1200x900 crop
- red: raw component chosen by the heuristic
- yellow: expanded pseudo-label suggested for annotation

Suggested workflow:
1. Open images + labels in your annotation tool
2. Keep or adjust yellow suggested boxes
3. Delete boxes on false positives and add boxes on missed defects
4. Fill manifest.csv review_code/review_note if you want to audit suggestion quality

review_code suggestion:
1 = suggestion good
0 = suggestion wrong
2 = partly useful but needs adjustment
