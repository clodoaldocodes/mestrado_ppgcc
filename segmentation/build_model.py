from ultralytics import YOLO
from ultralytics import YOLO, settings

settings.update({'runs_dir': './runs'})
model_name = "yolov8x-seg"

model = YOLO(model_name)  # yolov8n_custom.yaml
# filename = 'A:\ACOLITE_CLODOALDO\dataset_complete.v3i.yolov8-obb-20240219T004720Z-001\dataset_complete.v3i.yolov8-obb\data.yaml'
# filename = '/content/drive/MyDrive/dataset_complete.v2i.yolov8-obb/data.yaml'
# filename = '/content/drive/MyDrive/dataset_complete.v1i.yolov8-obb/data.yaml'
filename = "C:/Users/cdsfj\Desktop/DOCUMENTOS/test_segmentation.v2i.yolov8-obbdata.yaml"

model.train(data=filename,
            epochs=1000,
            patience=10,
            batch=-1,  # number of images per batch (-1 for AutoBatch)
            imgsz=640,
            workers=8,
            pretrained=False,
            resume=False,  # resume training from last checkpoint
            single_cls=False,  # Whether all classes will be the same (just one class)
            box=10,#-5,  # More recall, better IoU, less precission,
            cls=1.5,  # Bbox class better
            dfl=1.5,  # Distribution Focal Loss. Better bbox boundaries
            val=True,
            verbose=True,
            plots=True,
            seed=42,
            )