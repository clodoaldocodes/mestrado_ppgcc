#%%
local_path = "/home/clodoaldo/mestrado_ppgcc/archive/Test/Test/"

from PIL import Image
import os
import datetime
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import datetime
import sys
import functionsToUse

#%%
# Caminho para a imagem
caminho_imagem = "/home/clodoaldo/mestrado_ppgcc/archive/Test/Test/Healthy/8bd96ae02c2b9cb6.jpg"  # Substitua pelo caminho da sua imagem

# Verificar se o arquivo existe
if os.path.exists(caminho_imagem):
    # Abrir a imagem
    with Image.open(caminho_imagem) as img:
        # Obter o tamanho da imagem
        largura, altura = img.size

        print(f"Tamanho da imagem: {largura} x {altura}")
else:
    print(f"A imagem não foi encontrada no caminho: {caminho_imagem}")

largura, altura = int(largura/6), int(altura/6)
print(f"Tamanho da imagem reduzida: {largura} x {altura}")

#%%
batch_size_nb = 2

# Configurar o gerador de dados de treinamento
train_generator = functionsToUse.CustomDataGenerator(
    local_path,
    image_size=(largura, altura),
    batch_size=batch_size_nb,
    label_mode='int',
    subset="training",
    seed=123,
    validation_split=0.2
)

# Configurar o gerador de dados de validação
validation_generator = functionsToUse.CustomDataGenerator(
    local_path,
    image_size=(largura, altura),
    batch_size=batch_size_nb,
    label_mode='int',
    subset="validation",
    seed=123,
    validation_split=0.2
)

class_names = train_generator.get_class_names()
num_classes = len(class_names)

#%%
all_image_paths, all_image_labels = train_generator.load_image_paths_and_labels()
#x_train, y_train = train_generator.process_images(all_image_paths, all_image_labels)
#x_train, y_train = train_generator.process_images(all_image_paths, all_image_labels)
x_train, y_train = train_generator.process_images_optimized(all_image_paths, all_image_labels)

#%%
x_train = tf.concat(x_train, axis=0)
y_train = tf.concat(y_train, axis=0)

all_image_paths, all_image_labels = validation_generator.load_image_paths_and_labels()
x_test, y_test = train_generator.process_images(all_image_paths, all_image_labels)

#%%
# Concatenar todas as listas em tensores
x_test = tf.concat(x_test, axis=0)
y_test = tf.concat(y_test, axis=0)

format_my = "%Y-%m-%d_%H:%M:%S"
datetime_now = datetime.datetime.now()
datetime_now = datetime_now.strftime(format_my)

outpath = "/home/clodoaldo/mestrado_ppgcc/resultados" + datetime_now + "/"

isExist = os.path.exists(outpath)
if not isExist:
  os.makedirs(outpath)

# Criar o modelo baseado na InceptionV3
base_model = tf.keras.applications.InceptionV3(
    include_top=False,
    weights='imagenet',
    input_shape=(largura, altura, 3)
)

modelId = "InceptionV3_"

# Congelar as camadas pré-treinadas
for layer in base_model.layers:
    layer.trainable = True

# Adicionar camadas personalizadas para a sua tarefa específica
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

#%%
# Compilar o modelo
model.compile(optimizer='SGD',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Inicializar uma lista para armazenar os tempos de execução de cada época
epoch_runtimes = []

# Treinar o modelo
for epoch in range(1000):
    # Registrar o tempo de início da época
    start_time = time.time()

    # Treinar o modelo para uma época
    history = model.fit(
        x_train, y_train,
        validation_data=[x_test, y_test],
        epochs=1,  # Treinar por uma única época
        batch_size=batch_size_nb,
        verbose=1
    )

    # Registrar o tempo de término da época
    end_time = time.time()

    # Calcular o tempo de execução da época
    epoch_runtime = end_time - start_time

    # Adicionar o tempo de execução à lista
    epoch_runtimes.append(epoch_runtime)

    # Imprimir o tempo de execução da época
    print(f"Época {epoch + 1} - Tempo de execução: {epoch_runtime:.2f} segundos")

    # Verificar critério de parada antecipada
    if len(epoch_runtimes) > 5 and all(epoch_runtimes[-5] < rt for rt in epoch_runtimes[-4:]):
        print("Parando antecipadamente devido à convergência")
        break

filename_txt_times = modelId + "times_" + datetime_now + ".txt"
outpath_filename = outpath + filename_txt_times

# Salvar tempos de execução em um arquivo de texto
with open(outpath_filename, 'w') as f:
    for epoch, runtime in enumerate(epoch_runtimes, start=1):
        f.write(f"Época {epoch} - Tempo de execução: {runtime:.2f} segundos\n")

# Plotar curvas de treinamento
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.savefig(outpath + modelId + "training_curve_" + datetime_now + ".png", dpi=600)
plt.show()

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc=0)
plt.savefig(outpath + modelId + "loss_curve_" + datetime_now + ".png", dpi=600)
plt.show()

filename = modelId + "model_" + datetime_now + ".h5"
model.save(outpath + filename)

print(f"Modelo salvo em: {outpath}")
print(f"Nome do arquivo: {filename}")

#%%
# Carregar o modelo treinado
model = tf.keras.models.load_model(outpath + filename)

# Tempo de predição no final
start_time_prediction = time.time()

# Fazer previsões no conjunto de teste
predictions = model.predict(x_test)

# Registrar o tempo de término da predição
end_time_prediction = time.time()

# Calcular o tempo de execução da predição
prediction_runtime = end_time_prediction - start_time_prediction

# Imprimir o tempo de execução da predição
print(f"Tempo de predição: {prediction_runtime:.2f} segundos")
outpath_filename = outpath + filename_txt_times

# Salvar o tempo de predição em um arquivo de texto
with open(outpath_filename, 'a') as f:
    f.write(f"Tempo de predição: {prediction_runtime:.2f} segundos\n")

# Converter as previsões em rótulos de classe
predicted_labels = np.argmax(predictions, axis=1)

# Calcular a matriz de confusão
conf_matrix = confusion_matrix(y_test, predicted_labels)

# Calcular as porcentagens para cada classe
row_sums = conf_matrix.sum(axis=1, keepdims=True)
conf_matrix_percent = conf_matrix / row_sums.astype(float) * 100

# Formatar manualmente os valores para adicionar o símbolo de porcentagem
annot_values = np.asarray([f"{val:.1f}%" for val in conf_matrix_percent.flatten()]).reshape(conf_matrix.shape)

sns.heatmap(conf_matrix_percent, annot=annot_values, fmt="", cmap="Blues", cbar=False,
            xticklabels=class_names,
            yticklabels=class_names)
plt.title("Matriz de Confusão com Porcentagens")
plt.xlabel("Valores Previstos")
plt.ylabel("Valores Reais")
plt.savefig(outpath + modelId + "matrix_conf_pc_" + datetime_now + ".png", dpi=600)
plt.show()

# Imprimir relatório de classificação
report = classification_report(y_test, predicted_labels, target_names=class_names)
print("\nRelatório de Classificação:\n", report)

filename = modelId + "relatory_" + datetime_now + ".txt"
outpath_filename = outpath + filename

# Redirecionar a saída para um arquivo
with open(outpath_filename, 'w') as f:
    sys.stdout = f  # Redirecionar a saída para o arquivo

    # Imprimir relatório de classificação
    report = classification_report(y_test, predicted_labels, target_names=class_names)
    print("\nRelatório de Classificação:\n", report)

# Restaurar a saída padrão
sys.stdout = sys.__stdout__
# %%
