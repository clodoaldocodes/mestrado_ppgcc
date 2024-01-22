import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.applications import MobileNet
from keras_efficientnets import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import time

number_epochs = 2
#number_epochs = 10000

# Carregar os dados
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
num_classes = 10

# Lista de modelos
models = []

# Função para criar e treinar um modelo
def train_model(model_fn, model_name):
    input_layer = Input(shape=(32, 32, 3))
    model = model_fn(input_layer)

    model.compile(optimizer=SGD(learning_rate=1e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Adicione a função EarlyStopping para parar o treinamento quando não há ganho de informação
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print(f"Iniciando treinamento do modelo: {model_name}")  # Adicione a impressão do nome do modelo
    start_time = time.time()
    history = model.fit(x_train, y_train, epochs=number_epochs, batch_size=128,
                        validation_data=(x_test, y_test), verbose=1,
                        callbacks=[early_stopping])  # Callback EarlyStopping

    end_time = time.time()

    # Avaliar o modelo nos dados de teste
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

    models.append({
        'Model': model_name,
        'Test Loss': test_loss,
        'Test Accuracy': test_accuracy,
        'Training Time (s)': end_time - start_time,
        'Epochs Trained': len(history.history['loss'])
    })

# LeNet
def create_lenet(input_layer):
    x = Conv2D(6, (5, 5), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(16, (5, 5), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(120, activation='relu')(x)
    x = Dense(84, activation='relu')(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=input_layer, outputs=output_layer)

train_model(create_lenet, 'LeNet')

# AlexNet (uma implementação simplificada)
def create_alexnet(input_layer):
    x = Conv2D(96, (11, 11), strides=(4, 4), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D((2, 2))(x)  # Reduza o tamanho do pool para (2, 2)
    x = Conv2D(256, (5, 5), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)  # Reduza o tamanho do pool para (2, 2)
    x = Conv2D(384, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(384, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)  # Reduza o tamanho do pool para (2, 2)
    x = Flatten()(x)
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=input_layer, outputs=output_layer)

train_model(create_alexnet, 'AlexNet')

# VGG16
def create_vgg16(input_layer):
    base_model = tf.keras.applications.VGG16(input_shape=(32, 32, 3), include_top=False, weights=None)
    x = Flatten()(base_model(input_layer))
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=input_layer, outputs=output_layer)

train_model(create_vgg16, 'VGG16')

# VGG16
def create_vgg19(input_layer):
    base_model = tf.keras.applications.VGG19(input_shape=(32, 32, 3), include_top=False, weights=None)
    x = Flatten()(base_model(input_layer))
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=input_layer, outputs=output_layer)

train_model(create_vgg19, 'VGG19')

# ResNet-50
def create_resnet50(input_layer):
    base_model = tf.keras.applications.ResNet50(input_shape=(32, 32, 3), include_top=False, weights=None)
    x = Flatten()(base_model(input_layer))
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=input_layer, outputs=output_layer)

train_model(create_resnet50, 'ResNet-50')

# Inception-V3 (GoogleNet)
# Custom Inception-like model for 32x32 input size
def create_inceptionv1(input_layer):
    x = Conv2D(64, (1, 1), activation='relu')(input_layer)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = Conv2D(192, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = Conv2D(256, (1, 1), activation='relu')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(384, (1, 1), activation='relu')(x)
    x = Conv2D(384, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(256, (1, 1), activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=input_layer, outputs=output_layer)


train_model(create_inceptionv1, 'Inception-V3')

# EfficientNet
def create_efficientnet(input_layer):
    base_model = EfficientNetB0(input_shape=(32, 32, 3), include_top=False, weights=None)
    x = Flatten()(base_model(input_layer))
    x = Dense(512, activation='relu')(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=input_layer, outputs=output_layer)

# Train EfficientNet
train_model(create_efficientnet, 'EfficientNet')

# MobileNet
def create_mobilenet(input_layer):
    base_model = MobileNet(input_shape=(32, 32, 3), include_top=False, weights=None)
    x = Flatten()(base_model(input_layer))
    x = Dense(512, activation='relu')(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=input_layer, outputs=output_layer)

# Train MobileNet
train_model(create_mobilenet, 'MobileNet')

# MobileNetV3Large
def create_mobilenetv3large(input_layer):
    base_model = tf.keras.applications.MobileNetV3Large(input_shape=(32, 32, 3), include_top=False, weights=None)
    x = Flatten()(base_model(input_layer))
    x = Dense(512, activation='relu')(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=input_layer, outputs=output_layer)

# Train MobileNetV3Large
train_model(create_mobilenetv3large, 'MobileNetV3Large')

# Criar um DataFrame com os resultados
results_df = pd.DataFrame(models)
print(results_df)

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Create a figure with subplots
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 5))

# Plot Test Loss
sns.barplot(x='Test Loss', y='Model', data=results_df, ax=axes[0])
axes[0].set_title('Test Loss')
axes[0].set_xlim(0, 2)

# Plot Test Accuracy (formatted as percentage)
def format_percentage(x, pos):
    return f'{x*100:.1f}%'

sns.barplot(x='Test Accuracy', y='Model', data=results_df, ax=axes[1])
axes[1].set_title('Test Accuracy')
axes[1].set_xlim(0, 1)
axes[1].xaxis.set_major_formatter(FuncFormatter(format_percentage))  # Format as percentage

# Plot Training Time
sns.barplot(x='Training Time (s)', y='Model', data=results_df, ax=axes[2])
axes[2].set_title('Training Time (s)')

# Plot Number of Epochs
sns.barplot(x='Epochs Trained', y='Model', data=results_df, ax=axes[3])
axes[3].set_title('Number of Epochs')

# Adjust the layout
plt.tight_layout()

# Show the plot
plt.show()