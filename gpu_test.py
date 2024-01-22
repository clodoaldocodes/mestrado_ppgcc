import tensorflow as tf

# Listar as GPUs físicas disponíveis
physical_devices = tf.config.experimental.list_physical_devices('GPU')

if len(physical_devices) == 0:
    print("Nenhuma GPU disponível")
else:
    # Obter o nome da GPU em uso (se houver)
    current_gpu = tf.test.gpu_device_name()
    if current_gpu:
        print(current_gpu)
    else:
        print("Nenhuma GPU em uso")
