from PIL import Image
import os
from ultralytics import YOLO
from ultralytics import YOLO, settings

model = YOLO("A:/ACOLITE_CLODOALDO/runs/segment/train4/weights/best.pt")

# Directory where the images are located
#directory = "A:\ACOLITE_CLODOALDO\dataset_complete.v3i.yolov8-obb-20240219T004720Z-001\dataset_complete.v3i.yolov8-obb/train\images"
directory = "A:/ACOLITE_CLODOALDO/correspondence/"

# List to store the paths of .jpg images
jpg_images = []

# Iterate through all files in the directory
for file in os.listdir(directory):
    # Check if the file is a .jpg image
    if file.endswith('.jpg') or file.endswith('.png'):
        # Add the full path of the file to the list
        full_path = os.path.join(directory, file)
        jpg_images.append(full_path)

results_img = model.predict(source=jpg_images,
                        conf=0.50,
                        iou=0.9,  # Non-Maximum Supression (NMS)
                        imgsz=640,
                        #show=True,
                        save=True,
                        save_txt=True,  # Save bbox coordenation
                        save_conf=True,  # save_txt must be True
                        save_crop=True,
                        show_conf=True,
                        show_labels=True,
                        show_boxes=True,
                        # project='runs/detect',
                        stream=False  # Do inference now (False) or after (True)
                        )

for result in results_img:
    # detection
    #result.boxes.xyxy   # box with xyxy format, (N, 4)
    #result.boxes.xywh   # box with xywh format, (N, 4)
    #result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
    #result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
    #result.boxes.conf   # confidence score, (N, 1)
    #result.boxes.cls    # cls, (N, 1)

    # classification
    #result.probs     # cls prob, (num_class, )

    c = result.boxes.xywh.tolist()[0] # To get the coordinates.
    x, y, w, h = c[0], c[1], c[2], c[3] # x, y are the center coordinates.
    c = [x, y, w, h]

    x1 = int(c[0] - c[2] / 2)  # Canto superior esquerdo (x)
    y1 = int(c[1] - c[3] / 2)  # Canto superior esquerdo (y)
    x2 = int(c[0] + c[2] / 2)  # Canto inferior direito (x)
    y2 = int(c[1] + c[3] / 2)  # Canto inferior direito (y)
