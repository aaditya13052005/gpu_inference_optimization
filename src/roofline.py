"""
Roofline Model Analysis for GPU Performance
"""

import numpy as np
import matplotlib.pyplot as plt
import py3nvml.py3nvml as nvml
from configs.gpu_specs import GPU_SPECS


class RooflineAnalyzer:
    """
    Roofline model for analyzing compute vs memory bottlenecks
    """
    
    def __init__(self):
        # Initialize NVML
        nvml.nvmlInit()
        self.handle = nvml.nvmlDeviceGetHandleByIndex(0)
        
        # Get GPU name
        gpu_name = nvml.nvmlDeviceGetName(self.handle)
        self.gpu_name = gpu_name.decode() if isinstance(gpu_name, bytes) else gpu_name
        
        # Get specs
        self.gpu_specs = self._get_gpu_specs()
    
    def _get_gpu_specs(self):
        """Match GPU to known specs database"""
        
        # Try to match
        for known_gpu, specs in GPU_SPECS.items():
            if known_gpu in self.gpu_name:
                print(f"✅ Matched GPU: {known_gpu}")
                print(f"   Peak FP32: {specs['peak_fp32_tflops']} TFLOPS")
                print(f"   Peak FP16 (Tensor): {specs['peak_fp16_tensor_tflops']} TFLOPS")
                print(f"   Memory BW: {specs['memory_bandwidth_gbs']} GB/s")
                return specs
        
        # Not found
        print(f"⚠️  GPU '{self.gpu_name}' not in database!")
        print("Using generic values - results will be approximate")
        
        return {
            'name': self.gpu_name,
            'peak_fp32_tflops': 10.0,
            'peak_fp16_tflops': 20.0,
            'peak_fp16_tensor_tflops': 40.0,
            'memory_bandwidth_gbs': 500
        }
    
    def calculate_arithmetic_intensity(self, flops, memory_bytes):
        """
        AI = FLOPs / Bytes Transferred
        
        High AI (>50): Compute-bound
        Low AI (<10): Memory-bound
        """
        return flops / memory_bytes if memory_bytes > 0 else 0
    
    def calculate_achieved_tflops(self, flops, time_seconds):
        """Calculate achieved TFLOPS from profiling data"""
        return (flops / time_seconds) / 1e12
    
    def analyze_bottleneck(self, arithmetic_intensity, achieved_tflops, precision='fp32'):
        """Determine if compute-bound or memory-bound"""
        
        bandwidth = self.gpu_specs['memory_bandwidth_gbs']
        peak_key = f'peak_{precision}_tensor_tflops' if precision == 'fp16' else f'peak_{precision}_tflops'
        peak_compute = self.gpu_specs[peak_key]
        
        # Ridge point (where roofs intersect)
        ridge_point = peak_compute * 1000 / bandwidth
        
        if arithmetic_intensity < ridge_point:
            bottleneck = "MEMORY-BOUND"
            theoretical_max = arithmetic_intensity * bandwidth / 1000
            efficiency = achieved_tflops / theoretical_max * 100
        else:
            bottleneck = "COMPUTE-BOUND"
            efficiency = achieved_tflops / peak_compute * 100
        
        return {
            'bottleneck': bottleneck,
            'efficiency_pct': efficiency,
            'ridge_point': ridge_point,
            'headroom_tflops': peak_compute - achieved_tflops
        }
    
    def plot_roofline(self, model_data, save_path='results/roofline/roofline_model.png'):
        """
        Create roofline plot
        
        model_data format:
        {
            'model_name': {
                'arithmetic_intensity': float,
                'achieved_tflops_fp32': float,
                'achieved_tflops_fp16': float
            }
        }
        """
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # X-axis: Arithmetic Intensity
        ai_range = np.logspace(-1, 3, 1000)
        
        # FP32 Roofline
        peak_fp32 = self.gpu_specs['peak_fp32_tflops']
        bandwidth = self.gpu_specs['memory_bandwidth_gbs']
        
        memory_bound_fp32 = ai_range * bandwidth / 1000
        compute_bound_fp32 = np.full_like(ai_range, peak_fp32)
        roofline_fp32 = np.minimum(memory_bound_fp32, compute_bound_fp32)
        
        ax.loglog(ai_range, roofline_fp32, 'b-', linewidth=3, 
                 label='FP32 Roofline', alpha=0.7)
        
        # FP16 Roofline (Tensor Cores)
        peak_fp16_tensor = self.gpu_specs['peak_fp16_tensor_tflops']
        memory_bound_fp16 = ai_range * bandwidth / 1000
        compute_bound_fp16 = np.full_like(ai_range, peak_fp16_tensor)
        roofline_fp16 = np.minimum(memory_bound_fp16, compute_bound_fp16)
        
        ax.loglog(ai_range, roofline_fp16, 'g-', linewidth=3,
                 label='FP16 Roofline (Tensor Cores)', alpha=0.7)
        
        # Plot model points
        colors = {'resnet50': 'red', 'mobilenet_v2': 'orange', 'efficientnet_b0': 'purple'}
        
        for model_name, data in model_data.items():
            ai = data['arithmetic_intensity']
            color = colors.get(model_name, 'blue')
            
            # FP32 point
            if 'achieved_tflops_fp32' in data:
                ax.scatter(ai, data['achieved_tflops_fp32'],
                          s=200, marker='o', color=color,
                          edgecolor='black', linewidth=2,
                          label=f'{model_name} (FP32)', zorder=5)
            
            # FP16 point
            if 'achieved_tflops_fp16' in data:
                ax.scatter(ai, data['achieved_tflops_fp16'],
                          s=200, marker='s', color=color,
                          edgecolor='black', linewidth=2,
                          label=f'{model_name} (FP16)', zorder=5)
            
            # Labels
            ax.text(ai * 1.3, data.get('achieved_tflops_fp32', 0), 
                   f'{model_name}', fontsize=9, ha='left')
        
        # Styling
        ax.axhline(y=peak_fp32, color='blue', linestyle='--', alpha=0.3)
        ax.axhline(y=peak_fp16_tensor, color='green', linestyle='--', alpha=0.3)
        
        ax.set_xlabel('Arithmetic Intensity (FLOPs/Byte)', 
                     fontsize=14, fontweight='bold')
        ax.set_ylabel('Performance (TFLOPS)', fontsize=14, fontweight='bold')
        ax.set_title(f'Roofline Model - {self.gpu_name}', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='lower right', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3, which='both')
        
        ax.set_xlim(0.5, 500)
        ax.set_ylim(0.1, peak_fp16_tensor * 1.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Roofline plot saved: {save_path}")