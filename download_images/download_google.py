import os
import requests
from bs4 import BeautifulSoup

def download_images(query, num_images, directory):
    # URL do Google Images
    url = f"https://www.google.com/search?q={query}&tbm=isch"
    
    # Fazendo a requisição GET para a página do Google Images
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Lista para armazenar os links das imagens
    image_links = []
    
    # Encontrando as tags 'img' e extraindo os links das imagens
    for img in soup.find_all('img'):
        image_links.append(img.get('src'))
        
    # Criação do diretório para salvar as imagens
    save_directory = os.path.join(directory, query)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    # Download das imagens e salvando no diretório
    for i, link in enumerate(image_links[:num_images]):
        try:
            img_data = requests.get(link).content
            with open(os.path.join(save_directory, f"image_{i+1}.jpg"), 'wb') as f:
                f.write(img_data)
            print(f"Imagem {i+1} baixada com sucesso.")
        except Exception as e:
            print(f"Erro ao baixar imagem {i+1}: {e}")

# Exemplo de uso
search_query = "laranja folha"
num_images_to_download = 50
download_directory = r"C:/Users/cdsfj/Desktop/DOCUMENTOS/non_HLB/train_google"
download_images(search_query, num_images_to_download, download_directory)
