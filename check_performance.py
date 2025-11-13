import psutil
import torch
import GPUtil

print("=== System Info ===")
print(f"CPU Usage: {psutil.cpu_percent()}%")
print(f"RAM Usage: {psutil.virtual_memory().percent}%")

print("\n=== PyTorch GPU Info ===")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

print("\n=== GPU Utilization ===")
try:
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU {gpu.id}: {gpu.name}")
        print(f"  Load: {gpu.load*100:.1f}%")
        print(f"  Memory: {gpu.memoryUsed}/{gpu.memoryTotal} MB ({gpu.memoryUtil*100:.1f}%)")
except:
    print("GPUtil not available, install: pip install gputil")
