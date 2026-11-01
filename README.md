# Automated Pediatric Bone Age and Biological Sex Estimation

> **Capstone Project ID:** `PW_27_COP_01`
> **Course Code:** UE24CS320A Capstone Phase-1 (Semester V, B.Tech CSE, 2024–2028)
> **Department:** Department of Computer Science & Engineering, PES University
> **Project Guide:** Prof. Prakash C O

---

## 📌 Team Credentials
- **Nishanth T Chigilar** (`PES1UG24CS302`) — *Deep Learning Lead*
- **Pavan Kumar HK** (`PES1UG24CS319`) — *Data & XAI Lead*
- **Mahesh Kumar B** (`PES1UG24CS255`) — *Systems & UI Lead*
- **Dhanush K S** (`PES1UG24CS148`) — *Documentation Lead*

---

## 🎯 Project Overview & Review 2 Milestone Status
This project presents an end-to-end deep learning framework designed to assess pediatric skeletal maturity from postero-anterior (PA) hand-wrist radiographs while estimating continuous bone age (in months) and biological sex.

In **Review 2**, the baseline single-task **ResNet-50 continuous regression model** achieved a **Best Validation MAE of 10.08 months** at Epoch 4, successfully surpassing the Review 2 benchmark requirement ($\text{MAE} < 12.0\text{ months}$).

```text
=================================================================
 PEDIATRIC BONE AGE - RESNET-50 BASELINE RESULTS (REVIEW 2)
=================================================================
Review 2 Metric Target     : Validation MAE < 12.0 Months
Achieved Validation MAE    : 10.08 Months (PASSED at Epoch 04)
Saved Checkpoint           : checkpoints/resnet50_baseline_best.pth
Image Preprocessing        : OpenCV CLAHE (clipLimit=2.0, tileGridSize=(8,8))
Input Tensor Resolution    : 512 x 512 RGB (Aspect-Ratio Padded)
Execution Hardware         : NVIDIA Tesla T4 GPU (FP32 Precision)
=================================================================
