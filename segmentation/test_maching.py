import cv2
import os
import matplotlib.pyplot as plt
import uuid

def calculate_iou(image1_path, image2_path):
    # Lê as imagens
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    # Redimensiona a imagem 2 para ter o mesmo tamanho da imagem 1
    img2_resized = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    # Converte para escala de cinza
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)

    # Calcula o índice de Jaccard
    intersection = cv2.bitwise_and(gray1, gray2)
    union = cv2.bitwise_or(gray1, gray2)
    iou = cv2.countNonZero(intersection) / cv2.countNonZero(union)

    return iou, img1, img2_resized

# Caminhos das imagens
output_path = "A:/ACOLITE_CLODOALDO/results_correspondence/"
image1_path = "A:/ACOLITE_CLODOALDO/runs/segment/predict11/crops/folha/LET06652.ARW_stack_q492.jpg"
const = 0.90
uiid_aux = str(uuid.uuid4())

iou_max = 0
for i in range(2,99):
    image2_path = "A:/ACOLITE_CLODOALDO/runs/segment/predict11/crops/folha/LET06652.ARW_stack_q6" + str(i) + ".jpg"
    print(image2_path)

    # Calcula o IOU
    iou, img1, img2_resized = calculate_iou(image1_path, image2_path)

    # Verifica se o IOU é maior que 0.70
    if iou > const:
        if iou > iou_max:
            iou_max = iou
            img_matching = img2_resized
            path_matching = image2_path

    else:
        print(f"O IOU entre as duas imagens não é maior que {const}.")

    if i == 98 and iou_max > 0:
        print(f"O IOU entre as duas imagens é maior que {const}.")
        print(f"Usando as imagem com iou de {iou_max:.2f}: {os.path.split(image1_path)[-1]} e {os.path.split(path_matching)[-1]}")
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        plt.title("Imagem 1")

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(img_matching, cv2.COLOR_BGR2RGB))
        plt.title("Imagem 2")

        plt.suptitle(f"IOU: {iou_max:.2f}", fontsize=16)
        plt.savefig(output_path + "image_with_iou" + uiid_aux + ".png", dpi=600)
        plt.show()
