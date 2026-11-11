# Review 3 experimental scripts

These five scripts have been reconstructed from the recorded Review 3 workflow and settings.
They are intended to restore the missing source files, not to overwrite the already-pushed
experiment evidence.

## Scripts
- `preprocess_hybrid.py`: CLAHE (2.0, 8x8), Top-Hat (15), gamma 1.0, 512x512, Z-score NPY.
- `segmentation_otsu.py`: original-versus-hybrid Otsu comparison on the first 20 IDs in the gamma metrics file.
- `segmentation_otsu_refined.py`: original-image Otsu with morphology and largest-contour selection; records border contact.
- `prepare_segmentation_annotation.py`: deterministic 50-image pilot sample stratified by sex and age quartile; creates a manifest and copies images only.
- `create_gamma_contact_sheet.py`: contact sheets for gamma 0.8, 1.0 and 1.2.

## Important limitations
- The scripts use the original Windows data paths under `D:\\Pediatric_Bone_Age`.
- The annotation script does not create masks or perform annotation.
- Otsu foreground percentages and contour statistics are exploratory diagnostics, not segmentation accuracy.
- The hybrid script is a reconstruction of the documented pipeline parameters. Validate it on a small sample before running over the complete dataset.
- Do not interpret these scripts as proof of improved bone-age prediction accuracy.
