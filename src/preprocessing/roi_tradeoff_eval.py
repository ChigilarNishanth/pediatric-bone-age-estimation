import pandas as pd

def generate_tradeoff_matrix():
    matrix_data = [
        {
            "Method": "Otsu Thresholding + Morphology",
            "Target Objective": "Fast Hand Background Isolation",
            "Boundary Precision": "Moderate",
            "Annotation Cost": "Zero (Unsupervised)",
            "Compute Latency": "< 5 ms per frame",
            "TW3 ROI Target": "Full Hand Outline Only",
            "Limitation": "Fails under low fingertip contrast or uneven beam exposure."
        },
        {
            "Method": "U-Net Semantic Masking",
            "Target Objective": "Pixel-Level Hand Contour Segmenting",
            "Boundary Precision": "High",
            "Annotation Cost": "High (Requires binary masks)",
            "Compute Latency": "~ 45 ms per frame",
            "TW3 ROI Target": "Full Isolated Hand Geometry",
            "Limitation": "High compute cost; requires full pixel annotations for training."
        },
        {
            "Method": "YOLOv8 ROI Bounding Boxes",
            "Target Objective": "TW3 Epiphyseal & Carpal Localization",
            "Boundary Precision": "Very High (Bounding Box)",
            "Annotation Cost": "Moderate (BBox labels)",
            "Compute Latency": "~ 20 ms per frame",
            "TW3 ROI Target": "20 Tanner-Whitehouse Growth Zones",
            "Limitation": "Requires localized bounding box labels over carpal/distal zones."
        }
    ]
    df = pd.DataFrame(matrix_data)
    out_csv = "checkpoints/preprocessing_roi_tradeoff_matrix.csv"
    df.to_csv(out_csv, index=False)
    print("\n--- Review 3 Preprocessing & ROI Trade-Off Comparison Matrix ---")
    print(df[["Method", "Boundary Precision", "Compute Latency", "Limitation"]].to_string(index=False))
    print(f"\nSaved trade-off matrix to '{out_csv}'.")

if __name__ == "__main__":
    generate_tradeoff_matrix()
