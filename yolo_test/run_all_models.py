from ultralytics import YOLO
import cv2
import os
import torch
import numpy as np
from ultralytics.utils.plotting import Annotator 
import classes_name
import time

# q6 - 450 - 560 - 685  (rgb_q6)
# q4 - 430 - 550 - 650  (rgb_q4)
# gray_650
# gray_850
#name_model = "yolov8n"
#name_model = "yolov8x"
classes = classes_name.classes

for iPath in range(0,4):
    for iModel in range(0,2):
        if iPath == 0:
            path_type = "gray_650" 
        if iPath == 1:
            path_type = "gray_850"
        if iPath == 2:
            path_type = "rgb_q6"
        else:
            path_type = "rgb_q4"

        if iModel == 0:
            name_model = "yolov8n"
        else:
            name_model = "yolov8x"

        #print(f"Using {path_type} and {name_model}\n -----")

        time_to_save = []
        img_path = "C:/Users/cdsfj/Desktop/DOCUMENTOS/non_HLB/" + path_type + "/"
        output_path = "C:/Users/cdsfj/Desktop/DOCUMENTOS/non_HLB/output/"
        confidence_number = 0.3
        model = YOLO(name_model + ".pt")

        output_path = output_path + name_model + "/"
        txt_path = output_path + "output_cf_" + str(confidence_number) + "_" + name_model + "_" + path_type + ".txt"

        file_names = [f for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f))]

        #selected_names = file_names[::30]
        selected_names = file_names

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open(txt_path, "w") as f:
            f.write("Information about the current run\n")
            f.write(f"Model used: {name_model}\n")
            f.write(f"Image used: {path_type}\n\n")
            
            for i in range(np.size(selected_names)):
                f.write(f"Begin image: {i} ---------------------------------------------------\n\n")
                
                start_time = time.time()
                results = model.predict(img_path + selected_names[i])
                end_time = time.time()
                inference_time = end_time - start_time

                # Carregar a imagem
                img = cv2.imread(img_path+  selected_names[i])

                for result in results:
                    # Obter informações sobre os boxes e probabilidades
                    boxes = result.boxes.xyxy
                    confidences = result.boxes.conf
                    class_labels = result.boxes.cls

                    k = 0
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
                            label = f"#{k} - Class: {class_name} - {int(class_label)+1}, Confidence: {confidence:.2f}"
                            cv2.putText(img, label, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
                            f.write(f"{label}\n")
                            k = k + 1

                time_to_save.append(inference_time)
                output_path_end = output_path + "output_" + str(i) + "_cf_" + str(confidence_number) + "_" + name_model + "_" + path_type + ".png"
                
                f.write(f"Inference Time: {inference_time:.4f} seconds\n")
                f.write(f"Finished image: {i} ---------------------------------------------------\n\n")
                # Salvar a imagem com os boxes desenhados
                cv2.imwrite(output_path_end, img)
                print(f"Imagem salva em: {output_path_end}")
            
            f.write(f"Max inference time: {np.max(inference_time)} seconds\n")
            f.write(f"Min inference time: {np.min(inference_time)} seconds\n")
            f.write(f"Avg inference time: {np.mean(inference_time)} seconds\n")
            f.write(f"Median inference time: {np.median(inference_time)} seconds\n")
            f.write(f"95th percentile inference time: {np.percentile(inference_time, 95)} seconds\n")
            f.write(f"99th percentile inference time: {np.percentile(inference_time, 99)} seconds\n")