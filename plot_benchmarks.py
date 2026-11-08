import os
import json
import matplotlib.pyplot as plt

def generate_benchmark_plots(variants, log_dir="checkpoints"):
    plt.figure(figsize=(10, 4.5))

    plt.subplot(1, 2, 1)
    for variant in variants:
        json_path = os.path.join(log_dir, f"history_{variant}.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
            plt.plot(range(1, len(data["val_mae"]) + 1), data["val_mae"], label=f"{variant.upper()}")
    
    plt.axhline(y=12.0, color='r', linestyle='--', label='Target Limit (<12.0m)')
    plt.title("Validation MAE Convergence")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (Months)")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    final_maes = []
    labels = []
    for variant in variants:
        json_path = os.path.join(log_dir, f"history_{variant}.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
            final_maes.append(min(data["val_mae"]))
            labels.append(variant.upper())

    bars = plt.bar(labels, final_maes, color=['#7f7f7f', '#1f77b4', '#2ca02c', '#d62728'])
    plt.axhline(y=12.0, color='r', linestyle='--', label='Target Limit')
    plt.title("Best MAE Across Preprocessing Pipelines")
    plt.ylabel("MAE (Months)")
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{yval:.2f}m", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("review3_variant_benchmark_comparison.png", dpi=300)
    print("Benchmark visualization plot successfully saved to 'review3_variant_benchmark_comparison.png'.")

if __name__ == "__main__":
    generate_benchmark_plots(['raw', 'clahe', 'hybrid', 'unet_masked'])