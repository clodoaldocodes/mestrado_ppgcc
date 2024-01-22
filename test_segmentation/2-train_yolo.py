import gc
import yaml
import time
import torch
import os, shutil
from tqdm import tqdm
from pathlib import Path
from ultralytics import YOLO
import variable_direc

# #### Configurations #### #
# Full path
path = variable_direc.path
path_labels = path + '/tmp/dataset/yolo'
path_images = path + '/tmp/dataset/images'

def train_yolo(path_yaml, path_save_run):
    model = YOLO('yolov8n')
    model.train(data=path_yaml,
                epochs=15,
                patience=4,
                batch=32,
                imgsz=640,
                workers=8,
                pretrained=True,
                project=path_save_run,
                val=True,
                exist_ok=True,
                verbose=False,
    )
    model.data = None
    del model
# #### End configurations #### #

# If you want to continue from a specific folder, set "is_continue" to True and within the loop inform one folder before starting processing
is_continue = False
for idx, folder_train in enumerate(os.listdir(path_labels)):
    print(f'##### Strating training {folder_train} ({idx + 1}-{len(os.listdir(path_labels))}) #####')
    for folder_especific_train in os.listdir(f'{path_labels}/{folder_train}'):
        # If "is_continue" is True (beforer running the script), it will skip the folders before and equal the if condition (useful if you stopped running and want to continue)
        if is_continue:
            if folder_train == 'yolo-0_10-90-95-50' and folder_especific_train == 'bbox_smaller':
                is_continue = False
            continue
        # By default we do not store the image folders with annotations, but the YOLO requires it. Then copy the required image folders to the same folder as annotation (temporary, the images will be removed later)
        for folder in ['train', 'val', 'test']:
            for filename in os.listdir(f'{path_labels}/{folder_train}/{folder_especific_train}/{folder}/labels'):
                Path(f'{path_labels}/{folder_train}/{folder_especific_train}/{folder}/images').mkdir(parents=True, exist_ok=True)
                shutil.copy(f'{path_images}/{folder}/{filename[:-3]}png', f'{path_labels}/{folder_train}/{folder_especific_train}/{folder}/images')

        # Remove result folder if exits
        shutil.rmtree(f'{path_labels}/{folder_train}/{folder_especific_train}/runs', ignore_errors=True)

        tic = time.time()
        train_yolo(f'{path_labels}/{folder_train}/{folder_especific_train}/dataset.yaml', f'{path_labels}/{folder_train}/{folder_especific_train}/runs')
        toc = time.time() - tic

        # Save total time to train in a new yalm
        with open(f'{path_labels}/{folder_train}/{folder_especific_train}/dataset.yaml', 'r') as file:
            dict_yaml = yaml.safe_load(file)
        with open(f'{path_labels}/{folder_train}/{folder_especific_train}/dataset_final.yaml', 'w') as outfile:
            dict_yaml['total_training_time'] = toc
            yaml.dump(dict_yaml, outfile)

        # Remove copied image folders to train the model
        for folder in ['train', 'val', 'test']:
            shutil.rmtree(f'{path_labels}/{folder_train}/{folder_especific_train}/{folder}/images')

        # Clean GPU em RAM memory used during the training. If we do not clean up, some trash will remain in our memory and after some training it will explode!!
        torch.cuda.empty_cache()
        gc.collect()