import os
import numpy as np
from skimage import exposure
from skimage import io
from tqdm import tqdm

# Diretório de entrada e saída
input_dir = "C:/Users/cdsfj/Desktop/DOCUMENTOS/non_HLB/png/"
output_dir = "C:/Users/cdsfj/Desktop/DOCUMENTOS/non_HLB/png_corrected/"

# Certifique-se de que o diretório de saída exista, caso contrário, crie-o
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Lista todos os arquivos de imagem no diretório de entrada
image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Iterar sobre todos os arquivos de imagem
for filename in tqdm(image_files, desc="Processing images", unit="image"):
    # Caminho completo do arquivo de entrada
    input_path = os.path.join(input_dir, filename)
    
    # Ler a imagem
    img = io.imread(input_path)
    
    # Calcular os percentis
    p2, p98 = np.percentile(img, (2, 98))
    
    # Aplicar rescale de intensidade
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    
    # Caminho completo do arquivo de saída
    output_path = os.path.join(output_dir, filename)
    
    # Salvar a imagem
    io.imsave(output_path, img_rescale)