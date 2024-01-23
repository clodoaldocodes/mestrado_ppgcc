from ultralytics import YOLO
import cv2
import os
import torch
import numpy as np
from ultralytics.utils.plotting import Annotator 


path_type = "gray_850" 
# q6 - 450 - 560 - 685  (rgb_q6)
# q4 - 430 - 550 - 650  (rgb_q4)
# gray_650
# gray_850

img_path = "/home/clodoaldo/Documentos/non_HLB/" + path_type + "/"
output_path = "/home/clodoaldo/Documentos/GitHub/mestrado_ppgcc/yolo_test/output/"
confidence_number = 0.3

#name_model = "yolov8n"
name_model = "yolov8x"
model = YOLO(name_model + ".pt")

output_path = output_path + name_model + "/"

file_names = [f for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f))]

selected_names = file_names[::10]

classes = {0: u'__background__',
 1: u'person',
 2: u'bicycle',
 3: u'car',
 4: u'motorcycle',
 5: u'airplane',
 6: u'bus',
 7: u'train',
 8: u'truck',
 9: u'boat',
 10: u'traffic light',
 11: u'fire hydrant',
 12: u'stop sign',
 13: u'parking meter',
 14: u'bench',
 15: u'bird',
 16: u'cat',
 17: u'dog',
 18: u'horse',
 19: u'sheep',
 20: u'cow',
 21: u'elephant',
 22: u'bear',
 23: u'zebra',
 24: u'giraffe',
 25: u'backpack',
 26: u'umbrella',
 27: u'handbag',
 28: u'tie',
 29: u'suitcase',
 30: u'frisbee',
 31: u'skis',
 32: u'snowboard',
 33: u'sports ball',
 34: u'kite',
 35: u'baseball bat',
 36: u'baseball glove',
 37: u'skateboard',
 38: u'surfboard',
 39: u'tennis racket',
 40: u'bottle',
 41: u'wine glass',
 42: u'cup',
 43: u'fork',
 44: u'knife',
 45: u'spoon',
 46: u'bowl',
 47: u'banana',
 48: u'apple',
 49: u'sandwich',
 50: u'orange',
 51: u'broccoli',
 52: u'carrot',
 53: u'hot dog',
 54: u'pizza',
 55: u'donut',
 56: u'cake',
 57: u'chair',
 58: u'couch',
 59: u'potted plant',
 60: u'bed',
 61: u'dining table',
 62: u'toilet',
 63: u'tv',
 64: u'laptop',
 65: u'mouse',
 66: u'remote',
 67: u'keyboard',
 68: u'cell phone',
 69: u'microwave',
 70: u'oven',
 71: u'toaster',
 72: u'sink',
 73: u'refrigerator',
 74: u'book',
 75: u'clock',
 76: u'vase',
 77: u'scissors',
 78: u'teddy bear',
 79: u'hair drier',
 80: u'toothbrush'}

if not os.path.exists(output_path):
    os.makedirs(output_path)

for i in range(np.size(selected_names)):
    results = model.predict(img_path + selected_names[i])

    # Carregar a imagem
    img = cv2.imread(img_path+  selected_names[i])

    for result in results:
        # Obter informações sobre os boxes e probabilidades
        boxes = result.boxes.xyxy
        confidences = result.boxes.conf
        class_labels = result.boxes.cls

        # Iterar sobre os boxes
        for box, confidence, class_label in zip(boxes, confidences, class_labels):
            # Verificar se a confiança é maior que 0.5
            if confidence > confidence_number:
                # Extrair coordenadas do box
                x_min, y_min, x_max, y_max = map(int, box)

                # Desenhar o box na imagem
                color = (0, 255, 0)  # Cor verde
                thickness = 2
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, thickness)

                # Adicionar rótulo com a classe e confiança
                class_name = classes[int(class_label)+1]
                label = f"Class: {class_name} - {int(class_label)+1}, Confidence: {confidence:.2f}"
                cv2.putText(img, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    
    output_path_end = output_path + "output_" + str(i) + "_cf_" + str(confidence_number) + "_" + name_model + "_" + path_type + ".png"
    # Salvar a imagem com os boxes desenhados
    cv2.imwrite(output_path_end, img)
    print(f"Imagem salva em: {output_path_end}")