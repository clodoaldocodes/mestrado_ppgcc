import psutil
import tensorflow as tf

# CPU and RAM resources
cpu_info = psutil.cpu_percent(interval=1, percpu=True)  # CPU usage per core
ram_info = psutil.virtual_memory()  # RAM usage

print("CPU Resources:")
for i, cpu in enumerate(cpu_info):
    print(f"Core {i + 1}: {cpu}%")

print("\nRAM Usage:")
print(f"Total: {ram_info.total / (1024 ** 3):.2f} GB")
print(f"Available: {ram_info.available / (1024 ** 3):.2f} GB")
print(f"Used: {ram_info.used / (1024 ** 3):.2f} GB")
print(f"Usage Percentage: {ram_info.percent}%")

# GPU resources (if available)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print("\nGPU Resources:")
    for i, gpu in enumerate(gpus):
        print(f"GPU {i + 1}: {gpu.name}")
        memory_info = tf.config.experimental.get_memory_info(gpu)
        print(f"Total Memory: {memory_info.total / (1024 ** 3):.2f} GB")
else:
    print("\nNo GPU available on this system.")
