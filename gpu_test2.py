import subprocess

# Obtém informações sobre a GPU
try:
    gpu_info = subprocess.check_output(['nvidia-smi'], stderr=subprocess.STDOUT).decode()
    print(gpu_info)
except subprocess.CalledProcessError as e:
    print('No GPU found. Please make sure NVIDIA GPU and nvidia-smi are properly installed.')
