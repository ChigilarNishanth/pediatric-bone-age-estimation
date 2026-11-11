# Review 3 – Preprocessing and Segmentation Experiments

## 1. Primary preprocessing approach

CLAHE remains the primary preprocessing approach for the project.
Top-Hat filtering, gamma correction and Z-score normalization were
explored as experimental alternatives. No improvement in bone-age
prediction accuracy is claimed from visual comparisons alone.

## 2. Gamma comparison

A 20-image sample was evaluated with gamma values 0.8, 1.0 and 1.2.

Recorded average standard deviations:
- Gamma 0.8: 0.841190
- Gamma 1.0: 0.842582
- Gamma 1.2: 0.843378

These statistics describe image intensity variation. They do not
establish which gamma value produces lower bone-age prediction error.

## 3. Original versus hybrid Otsu comparison

Otsu thresholding was examined on 20 original and 20 hybrid images.

Recorded average foreground percentages:
- Original images: 31.96%
- Hybrid images: 18.96%

Foreground percentage is not a segmentation accuracy metric.
Visual inspection showed incomplete hand regions and background or
positioning-plate interference in some masks.

## 4. Refined Otsu experiment

The refined Otsu experiment processed 20 images successfully.
- Failed images: 0
- Average foreground percentage: 33.24%
- Selected contours touching the image boundary: 16 of 20

Visual inspection found plausible hand regions in some images, but
also plate-dominated or boundary-interfered masks in others.
Therefore, this is an experimental baseline, not validated final
hand segmentation.

## 5. Annotation preparation

A pilot annotation manifest was prepared for 50 images:
- Male=False: 26
- Male=True: 24
- Four age-group counts: 13, 13, 12 and 12
- Images copied: 50
- Missing images: 0

The manifest is a preparation artifact. It does not mean that
ground-truth segmentation masks have been annotated or that U-Net
has been trained.

## 6. Current conclusions and limitations

- CLAHE remains the primary preprocessing approach.
- Gamma, Top-Hat and Z-score remain exploratory alternatives.
- Otsu and refined Otsu are experimental segmentation baselines.
- No validated U-Net or YOLOv8 segmentation result is claimed.
- No model-level MAE comparison between raw, CLAHE and hybrid
  inputs is included in these artifacts.
- A fair ResNet-50 validation comparison is required before
  claiming that one preprocessing method reduces prediction error.

## 7. Evidence included

- Gamma comparison metrics and contact sheets.
- Original-versus-hybrid Otsu metrics.
- Refined Otsu metrics, failures report and contact sheets.
- Hybrid processing logs.
- 50-image segmentation annotation manifest.
