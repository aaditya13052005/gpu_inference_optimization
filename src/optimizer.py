"""
Inference Optimization Experiments
(SAFE for torchvision models like ResNet)
"""

import torch
import numpy as np
import copy


class InferenceOptimizer:
    """
    Test various optimization strategies for inference
    """

    def __init__(self, model, device='cuda'):
        self.device = device
        self.model = model.to(device).eval()

        # ✅ Snapshot initial weights ONCE
        self._base_state = copy.deepcopy(self.model.state_dict())

    # ----------------------------------------------------
    def _reset_model(self):
        """Restore model to original weights"""
        self.model.load_state_dict(self._base_state)
        self.model.to(self.device).eval()

    # ----------------------------------------------------
    def benchmark(self, input_tensor, num_iterations=100, warmup=10):
        """Accurate benchmarking with warmup"""

        with torch.no_grad():
            for _ in range(warmup):
                _ = self.model(input_tensor)

        if self.device == 'cuda':
            torch.cuda.synchronize()

        latencies = []
        with torch.no_grad():
            for _ in range(num_iterations):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = self.model(input_tensor)
                end.record()
                torch.cuda.synchronize()
                latencies.append(start.elapsed_time(end))

        return {
            'mean_ms': float(np.mean(latencies)),
            'median_ms': float(np.median(latencies)),
            'p95_ms': float(np.percentile(latencies, 95)),
            'std_ms': float(np.std(latencies))
        }

    # ----------------------------------------------------
    def test_torchscript(self, input_tensor):
        print("  Testing TorchScript...")
        self._reset_model()

        baseline = self.benchmark(input_tensor)

        try:
            scripted = torch.jit.trace(self.model, input_tensor)
            self.model = scripted
            optimized = self.benchmark(input_tensor)
            speedup = baseline['mean_ms'] / optimized['mean_ms']
        except Exception as e:
            print(f"    ⚠️ Failed: {e}")
            optimized = baseline
            speedup = 1.0

        return {
            'method': 'TorchScript',
            'baseline': baseline,
            'optimized': optimized,
            'speedup': speedup
        }

    # ----------------------------------------------------
    def test_channels_last(self, input_tensor):
        print("  Testing channels_last...")
        self._reset_model()

        baseline = self.benchmark(input_tensor)

        try:
            self.model = self.model.to(memory_format=torch.channels_last)
            input_cl = input_tensor.to(memory_format=torch.channels_last)
            optimized = self.benchmark(input_cl)
            speedup = baseline['mean_ms'] / optimized['mean_ms']
        except Exception as e:
            print(f"    ⚠️ Failed: {e}")
            optimized = baseline
            speedup = 1.0

        return {
            'method': 'channels_last',
            'baseline': baseline,
            'optimized': optimized,
            'speedup': speedup
        }

    # ----------------------------------------------------
    def test_cudnn_benchmark(self, input_tensor):
        print("  Testing cuDNN benchmark...")
        self._reset_model()

        torch.backends.cudnn.benchmark = False
        baseline = self.benchmark(input_tensor)

        torch.backends.cudnn.benchmark = True
        with torch.no_grad():
            for _ in range(20):
                _ = self.model(input_tensor)

        optimized = self.benchmark(input_tensor)
        speedup = baseline['mean_ms'] / optimized['mean_ms']

        return {
            'method': 'cudnn_benchmark',
            'baseline': baseline,
            'optimized': optimized,
            'speedup': speedup
        }

    # ----------------------------------------------------
    def test_fp16(self, input_tensor):
        print("  Testing FP16...")
        self._reset_model()

        baseline = self.benchmark(input_tensor.float())

        with torch.cuda.amp.autocast():
            optimized = self.benchmark(input_tensor.half())

        speedup = baseline['mean_ms'] / optimized['mean_ms']

        return {
            'method': 'fp16',
            'baseline': baseline,
            'optimized': optimized,
            'speedup': speedup
        }

    # ----------------------------------------------------
    def run_all_optimizations(self, input_tensor):
        results = []

        for fn in [
            self.test_torchscript,
            self.test_channels_last,
            self.test_cudnn_benchmark,
            self.test_fp16,
        ]:
            result = fn(input_tensor)
            results.append(result)

            print(
                f"    ✅ {result['method']}: "
                f"{result['speedup']:.3f}× "
                f"({(result['speedup'] - 1) * 100:.1f}%)"
            )

        return results
    import matplotlib.pyplot as plt

def plot_optimization_results(df):
    plt.figure(figsize=(12, 6))
    
    for model_name in df['model'].unique():
        subset = df[df['model'] == model_name]
        plt.plot(
            subset['optimization'],
            subset['speedup'],
            marker='o',
            label=model_name.upper()
        )
    
    plt.axhline(1.0, linestyle='--', linewidth=1)
    plt.ylabel('Speedup (×)')
    plt.xlabel('Optimization Method')
    plt.title('Inference Optimization Speedups')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

