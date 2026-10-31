# System Requirements Specification (SRS)
## Automated Pediatric Bone Age and Biological Sex Estimation
**Project ID:** PW_27_COP_01  
**Department:** Computer Science & Engineering, PES University  
**Document Version:** 1.0 (Review 2 Phase I Draft)  

---

### 1. Introduction
#### 1.1 Purpose
This document specifies the software requirements for an automated pediatric bone age assessment system using multi-task deep learning backbones.

#### 1.2 Scope
The system processes postero-anterior (PA) hand-wrist radiographs to simultaneously predict continuous skeletal bone age (in months) and biological sex, with anatomical Grad-CAM explainability and automated diagnostic PDF generation.

---

### 2. General Description
#### 2.1 Product Perspective
An end-to-end clinical research decision-support aid connecting image ingestion, preprocessing, deep feature extraction, CPU inference, and diagnostic presentation.

#### 2.2 System Functions
1. **Radiograph Ingestion & Enhancement:** OpenCV CLAHE contrast tuning ($clipLimit=2.0$, $tileGridSize=(8,8)$) and scaling to $512 \times 512$[cite: 8, 10, 23].
2. **Bone Age Estimation:** Regression modeling targeting baseline $MAE < 12.0$ months (Phase I ResNet-50)[cite: 1, 10, 34] and target $MAE < 8.5$ months (Phase II Dual-Head)[cite: 1, 34].
3. **Biological Sex Classification:** Target accuracy $> 92.0\%$[cite: 1, 34].
4. **Explainable AI (XAI):** Grad-CAM spatial heatmaps overlaid on carpal/epiphyseal growth regions[cite: 1, 9, 31].
5. **Report Generation:** PDF export summarizing predictions and heatmaps[cite: 1, 9, 22].

---

### 3. Specific Requirements
#### 3.1 Functional Requirements
- **FR-1:** System shall parse RSNA PNG radiographs and metadata CSV files[cite: 8, 23].
- **FR-2:** Preprocessing script shall convert 8-bit grayscale images to normalized 3-channel $512 \times 512 \times 3$ RGB tensors[cite: 8, 22, 23].
- **FR-3:** ResNet-50 baseline model shall output single continuous floating-point prediction for bone age in months[cite: 1, 8, 10].

#### 3.2 Non-Functional Requirements
- **NFR-1 (Inference Latency):** Final ONNX INT8 CPU model response time $< 300\text{ ms}$ per case[cite: 1, 6, 11, 34].
- **NFR-2 (Accuracy Target):** ResNet-50 baseline validation $MAE < 12.0\text{ months}$[cite: 1, 10, 34].
- **NFR-3 (Data Security):** De-identified open-access RSNA dataset utilization with legal medical research disclaimer stamps[cite: 8, 15, 23].
