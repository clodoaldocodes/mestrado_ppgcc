import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib
import time
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)

path = "/home/clodoaldo/mestrado_ppgcc/archive"
# Definir o tamanho do lote e o número de épocas
batch_size = 16  # Tamanho do lote
num_epochs = 50  # Número máximo de épocas

# Carregar o conjunto de dados MNIST
train_ds =  datagen.flow_from_directory(path, subset="training", seed=123, batch_size=batch_size)
val_ds =  datagen.flow_from_directory(path, subset="validation", seed=123, batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Definir a arquitetura da CNN
model = models.Sequential([
    tf.keras.layers.Rescaling(1./255),
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(np.size(class_names)[0])
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Obter a hora atual
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")

# Tempo de cada uma das épocas
time_plot = []

# Salvar todas as métricas em um arquivo de texto
metrics_path = os.path.join(os.getcwd(), f'metricas_treinamento_{current_time}.txt')
with open(metrics_path, 'w') as file:
    file.write(f"Início do treinamento: {current_time}\n")
    file.write("Época\tAcurácia Treino\tAcurácia Validação\tErro Treino\tErro Validação\tTempo de Época\n")

    for epoch in range(num_epochs):
        start_epoch_time = time.time()  # Medir o tempo de início da época

        # Treinar o modelo por uma época
        history = model.fit(train_ds, batch_size=batch_size, epochs=1, 
                            validation_data=(val_ds))

        end_epoch_time = time.time()  # Medir o tempo de final da época

        file.write(f"{epoch + 1}\t{history.history['acc'][0]}\t{history.history['val_acc'][0]}\t"
                   f"{history.history['loss'][0]}\t{history.history['val_loss'][0]}\t"
                   f"{end_epoch_time - start_epoch_time:.2f} segundos\n")
        time_plot.append(end_epoch_time - start_epoch_time)

# Avaliar o modelo
test_loss, test_acc = model.evaluate(val_ds, verbose=2)
print(f"Acurácia no conjunto de teste: {test_acc * 100:.2f}%")

# Plotar a curva de aprendizado
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['acc'], label='Training Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.title(f'Learning Curve - {current_time}')  # Add the current time to the title
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Salvar a figura com a hora atual no nome do arquivo
image_path = os.path.join(os.getcwd(), f'curva_aprendizado_{current_time}.png')
plt.savefig(image_path)

# Plotar o tempo de treinamento de cada época
plt.figure(figsize=(12, 4))
plt.plot(range(len(time_plot)), time_plot)
plt.xlabel('Época')
plt.ylabel('Tempo')
plt.title(f'Curva de Aprendizado - {current_time}')  # Adicionar a hora atual ao título

# Salvar a figura com a hora atual no nome do arquivo
image_path = os.path.join(os.getcwd(), f'tempo_treinamento{current_time}.png')
plt.savefig(image_path)