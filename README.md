# Automated Pediatric Bone Age and Biological Sex Estimation

**Project ID:** PW_27_COP_01  
**Institution:** PES University, Department of Computer Science & Engineering  

## Review 2 Scope (Semester 5)
- Ingestion and Exploratory Data Analysis (EDA) of the RSNA Pediatric Bone Age Dataset (14,236 radiographs).
- Contrast-Limited Adaptive Histogram Equalization (CLAHE) preprocessing pipeline using OpenCV (`clipLimit=2.0`, `tileGridSize=(8,8)`, resolution `512x512`).
- Single-task ResNet-50 PyTorch baseline model for continuous age regression targeting MAE < 12.0 months.

## Repository Structure
- `src/`: Modular Python scripts for dataset splits, preprocessing, baseline model, and training loop.
- `notebooks/`: Exploratory Data Analysis (EDA) and CLAHE parameter experiments.
- `docs/`: IEEE System Requirements Specification (SRS) document draft.
- `checkpoints/`: Model weights (`.pth` files, git-ignored).
- `data/`: RSNA dataset storage (git-ignored).
