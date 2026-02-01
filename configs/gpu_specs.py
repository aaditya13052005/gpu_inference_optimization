"""
GPU Specifications Database
"""

GPU_SPECS = {
    'Tesla T4': {
        'peak_fp32_tflops': 8.1,
        'peak_fp16_tflops': 65,
        'peak_fp16_tensor_tflops': 65,
        'memory_bandwidth_gbs': 300
    },
    'Tesla V100-SXM2-16GB': {
        'peak_fp32_tflops': 15.7,
        'peak_fp16_tflops': 31.4,
        'peak_fp16_tensor_tflops': 125,
        'memory_bandwidth_gbs': 900
    },
    'Tesla P100-PCIE-16GB': {
        'peak_fp32_tflops': 10.6,
        'peak_fp16_tflops': 21.2,
        'peak_fp16_tensor_tflops': 21.2,
        'memory_bandwidth_gbs': 732
    },
    'Tesla K80': {
        'peak_fp32_tflops': 8.73,
        'peak_fp16_tflops': 8.73,
        'peak_fp16_tensor_tflops': 8.73,
        'memory_bandwidth_gbs': 480
    },
    'Tesla A100-SXM4-40GB': {
        'peak_fp32_tflops': 19.5,
        'peak_fp16_tflops': 77.9,
        'peak_fp16_tensor_tflops': 312,
        'memory_bandwidth_gbs': 1555
    },
    'NVIDIA L4': {
        'peak_fp32_tflops': 30.3,
        'peak_fp16_tflops': 60.6,
        'peak_fp16_tensor_tflops': 121,
        'memory_bandwidth_gbs': 300
    }
}