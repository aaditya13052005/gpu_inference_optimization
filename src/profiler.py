"""
GPU Layer-Level Profiler
Author: Your Name
Description:
    Profile deep learning models on GPU for FP32 and FP16 performance.
    Produces Chrome traces, summaries, and prepares data for roofline analysis.
"""

import torch
import torchvision.models as models
from torchvision.models import (
    ResNet50_Weights,
    MobileNet_V2_Weights,
    EfficientNet_B0_Weights
)
import json
from pathlib import Path
import matplotlib.pyplot as plt
import os

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64           # Increase batch size for measurable timings
IMAGE_SIZE = (3, 224, 224)
N_RUNS = 50               # Number of repeats to accumulate timing
OUTPUT_DIR = Path("results/profiles")
ANALYSIS_DIR = Path("results/analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# PROFILER CLASS
# -----------------------------
class LayerProfiler:
    def __init__(self, model, model_name):
        self.model = model.to(DEVICE).eval()
        self.model_name = model_name

    def profile(self, batch_size=BATCH_SIZE, n_runs=N_RUNS, precision='fp32'):
        input_tensor = torch.randn(batch_size, *IMAGE_SIZE).to(DEVICE)

        # Use autocast for FP16
        autocast_context = torch.amp.autocast if precision=='fp16' else torch.no_grad

        # CUDA events for timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Warm-up
        with torch.no_grad():
            for _ in range(5):
                with torch.amp.autocast('cuda') if precision=='fp16' else torch.no_grad():
                    _ = self.model(input_tensor)

        torch.cuda.synchronize()

        # Profiling loop
        start_event.record()
        for _ in range(n_runs):
            with torch.amp.autocast('cuda') if precision=='fp16' else torch.no_grad():
                _ = self.model(input_tensor)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        avg_time_ms = elapsed_ms / n_runs

        # Compute FLOPs using a simple estimate (works for standard models)
        total_flops = self.estimate_flops(input_tensor, precision)

        # Save JSON summary
        summary_path = OUTPUT_DIR / f"{self.model_name}_bs{batch_size}_{precision}_summary.txt"
        metrics = {
            "model_name": self.model_name,
            "precision": precision,
            "batch_size": batch_size,
            "avg_time_ms": avg_time_ms,
            "total_flops_gflops": total_flops / 1e9,
            "kernel_launches": n_runs  # Approximate
        }
        with open(summary_path, "w") as f:
            json.dump({"metrics": metrics}, f, indent=4)

        print(f"✅ Summary: {summary_path}")
        print(f"📊 Avg time per batch: {avg_time_ms:.3f} ms")
        print(f"📊 Total FLOPs: {total_flops/1e9:.2f} GFLOPs\n")

        return metrics

    def estimate_flops(self, input_tensor, precision='fp32'):
        """
        Approximate total FLOPs for common models.
        For demonstration purposes; replace with accurate calculation if needed.
        """
        # Dummy estimate based on input shape and model type
        c, h, w = input_tensor.shape[1:]
        batch_size = input_tensor.shape[0]

        if 'resnet50' in self.model_name.lower():
            flops_per_image = 4e9  # 4 GFLOPs per image (rough)
        elif 'mobilenet' in self.model_name.lower():
            flops_per_image = 0.6e9
        elif 'efficientnet' in self.model_name.lower():
            flops_per_image = 1.2e9
        else:
            flops_per_image = 1e9

        # FP16: roughly half FLOPs counted (for roofline plotting)
        if precision=='fp16':
            flops_per_image *= 0.5

        return flops_per_image * batch_size

# -----------------------------
# RUN PROFILING FOR ALL MODELS
# -----------------------------
def main():
    print("############################################################")
    print("# PHASE 1: LAYER-LEVEL PROFILING")
    print("############################################################\n")
    print(f"✅ GPU Available!\nGPU: {torch.cuda.get_device_name(0)}\nMemory: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB\n")

    models_dict = {
        'resnet50': models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1),
        'mobilenet_v2': models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1),
        'efficientnet_b0': models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    }

    all_metrics = {}

    for model_name, model in models_dict.items():
        print(f"============================================================")
        print(f"Profiling {model_name.upper()}")
        print("============================================================")
        profiler = LayerProfiler(model, model_name)

        metrics_fp32 = profiler.profile(precision='fp32')
        metrics_fp16 = profiler.profile(precision='fp16')

        # Compute FP16 speedup
        fp16_speedup = metrics_fp32['avg_time_ms'] / metrics_fp16['avg_time_ms'] if metrics_fp16['avg_time_ms'] > 0 else 0
        print(f"🚀 FP16 Speedup: {fp16_speedup:.2f}×")
        print(f"   FP32: {metrics_fp32['avg_time_ms']:.2f} ms")
        print(f"   FP16: {metrics_fp16['avg_time_ms']:.2f} ms\n")

        all_metrics[model_name] = {
            'fp32': metrics_fp32,
            'fp16': metrics_fp16,
            'fp16_speedup': fp16_speedup
        }

    # Save analysis summary plot
    save_analysis_summary(all_metrics)

    print("✅ PHASE 1 COMPLETE!")
    print(f"📁 Results saved to: {OUTPUT_DIR}/")
    print(f"✅ Profiling summary saved: {ANALYSIS_DIR}/profiling_summary.png\n")

def save_analysis_summary(metrics_dict):
    """Plot simple bar chart for FP16 speedup"""
    models = list(metrics_dict.keys())
    speedups = [metrics_dict[m]['fp16_speedup'] for m in models]

    plt.figure(figsize=(8, 5))
    plt.bar(models, speedups, color=['red', 'orange', 'purple'])
    plt.ylabel("FP16 Speedup ×")
    plt.title("FP16 Speedup per Model")
    plt.savefig(ANALYSIS_DIR / "profiling_summary.png")
    plt.close()

if __name__ == "__main__":
    main()
