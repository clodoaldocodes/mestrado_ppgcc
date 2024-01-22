import os
import yaml
import json
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
import variable_direc

np.random.seed(42)

# #### Configurations #### #
path = variable_direc.path
path_dataset = path + 'tmp/dataset'
path_images = path + 'tmp/dataset/images'
path_labels_json = path + 'tmp/dataset/labels_json'
path_output_original = path + 'tmp/dataset/yolo'
classes = {
    'rect': 0,
    'circle': 1
}

datasets_to_create = {  # Each dictionary key is a new folder with the parameters data
    # #### 100% dataset #### #
    'yolo-0_5-100-100-100': {  # Margin, not change class, not miss annotation, size dataset
        'bbox_low_high': (0, 5),  # Random value between low and high to draw bbox bigger or smaller than the real object
        'chance_not_change_class': 1.0,  # Range between 0 and 1; 1.0 == never change class; 0.0 == always change class
        'chance_not_miss_annotation': 1.0, # Range between 0 and 1; 1.0 == never miss annotation; 0.0 == always miss annotation
        'create_official': True, # Create folder with no annotation mistakes (create folder with other parameters and a new one ignoring them)
        'limit_img': {  # Range between 0 and 1; Images per folder. 1.0 == all images available; 0.5 == 50% images available
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },
    'yolo-0_10-100-100-100': {
        'bbox_low_high': (0, 10),
        'chance_not_change_class': 1.0,
        'chance_not_miss_annotation': 1.0,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },
    'yolo-0_15-100-100-100': {
        'bbox_low_high': (0, 15),
        'chance_not_change_class': 1.0,
        'chance_not_miss_annotation': 1.0,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },

    'yolo-0_5-95-100-100': {
        'bbox_low_high': (0, 5),
        'chance_not_change_class': 0.95,
        'chance_not_miss_annotation': 1.0,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },
    'yolo-0_10-95-100-100': {
        'bbox_low_high': (0, 10),
        'chance_not_change_class': 0.95,
        'chance_not_miss_annotation': 1.0,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },
    'yolo-0_15-95-100-100': {
        'bbox_low_high': (0, 15),
        'chance_not_change_class': 0.95,
        'chance_not_miss_annotation': 1.0,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },

    'yolo-0_5-90-100-100': {
        'bbox_low_high': (0, 5),
        'chance_not_change_class': 0.90,
        'chance_not_miss_annotation': 1.0,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },
    'yolo-0_10-90-100-100': {
        'bbox_low_high': (0, 10),
        'chance_not_change_class': 0.90,
        'chance_not_miss_annotation': 1.0,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },
    'yolo-0_15-90-100-100': {
        'bbox_low_high': (0, 15),
        'chance_not_change_class': 0.90,
        'chance_not_miss_annotation': 1.0,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },

    'yolo-0_5-85-100-100': {
        'bbox_low_high': (0, 5),
        'chance_not_change_class': 0.85,
        'chance_not_miss_annotation': 1.0,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },
    'yolo-0_10-85-100-100': {
        'bbox_low_high': (0, 10),
        'chance_not_change_class': 0.85,
        'chance_not_miss_annotation': 1.0,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },
    'yolo-0_15-85-100-100': {
        'bbox_low_high': (0, 15),
        'chance_not_change_class': 0.85,
        'chance_not_miss_annotation': 1.0,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },

    'yolo-0_5-100-95-100': {
        'bbox_low_high': (0, 5),
        'chance_not_change_class': 1.0,
        'chance_not_miss_annotation': 0.95,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },
    'yolo-0_10-100-95-100': {
        'bbox_low_high': (0, 10),
        'chance_not_change_class': 1.0,
        'chance_not_miss_annotation': 0.95,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },
    'yolo-0_15-100-95-100': {
        'bbox_low_high': (0, 15),
        'chance_not_change_class': 1.0,
        'chance_not_miss_annotation': 0.95,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },

    'yolo-0_5-95-95-100': {
        'bbox_low_high': (0, 5),
        'chance_not_change_class': 0.95,
        'chance_not_miss_annotation': 0.95,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },
    'yolo-0_10-95-95-100': {
        'bbox_low_high': (0, 10),
        'chance_not_change_class': 0.95,
        'chance_not_miss_annotation': 0.95,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },
    'yolo-0_15-95-95-100': {
        'bbox_low_high': (0, 15),
        'chance_not_change_class': 0.95,
        'chance_not_miss_annotation': 0.95,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },

    'yolo-0_5-90-95-100': {
        'bbox_low_high': (0, 5),
        'chance_not_change_class': 0.90,
        'chance_not_miss_annotation': 0.95,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },
    'yolo-0_10-90-95-100': {
        'bbox_low_high': (0, 10),
        'chance_not_change_class': 0.90,
        'chance_not_miss_annotation': 0.95,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },
    'yolo-0_15-90-95-100': {
        'bbox_low_high': (0, 15),
        'chance_not_change_class': 0.90,
        'chance_not_miss_annotation': 0.95,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },

    'yolo-0_5-85-95-100': {
        'bbox_low_high': (0, 5),
        'chance_not_change_class': 0.85,
        'chance_not_miss_annotation': 0.95,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },
    'yolo-0_10-85-95-100': {
        'bbox_low_high': (0, 10),
        'chance_not_change_class': 0.85,
        'chance_not_miss_annotation': 0.95,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },
    'yolo-0_15-85-95-100': {
        'bbox_low_high': (0, 15),
        'chance_not_change_class': 0.85,
        'chance_not_miss_annotation': 0.95,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },

    'yolo-0_5-100-90-100': {
        'bbox_low_high': (0, 5),
        'chance_not_change_class': 1.0,
        'chance_not_miss_annotation': 0.90,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },
    'yolo-0_10-100-90-100': {
        'bbox_low_high': (0, 10),
        'chance_not_change_class': 1.0,
        'chance_not_miss_annotation': 0.90,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },
    'yolo-0_15-100-90-100': {
        'bbox_low_high': (0, 15),
        'chance_not_change_class': 1.0,
        'chance_not_miss_annotation': 0.90,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },

    'yolo-0_5-95-90-100': {
        'bbox_low_high': (0, 5),
        'chance_not_change_class': 0.95,
        'chance_not_miss_annotation': 0.90,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },
    'yolo-0_10-95-90-100': {
        'bbox_low_high': (0, 10),
        'chance_not_change_class': 0.95,
        'chance_not_miss_annotation': 0.90,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },
    'yolo-0_15-95-90-100': {
        'bbox_low_high': (0, 15),
        'chance_not_change_class': 0.95,
        'chance_not_miss_annotation': 0.90,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },

    'yolo-0_5-90-90-100': {
        'bbox_low_high': (0, 5),
        'chance_not_change_class': 0.90,
        'chance_not_miss_annotation': 0.90,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },
    'yolo-0_10-90-90-100': {
        'bbox_low_high': (0, 10),
        'chance_not_change_class': 0.90,
        'chance_not_miss_annotation': 0.90,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },
    'yolo-0_15-90-90-100': {
        'bbox_low_high': (0, 15),
        'chance_not_change_class': 0.90,
        'chance_not_miss_annotation': 0.90,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },

    'yolo-0_5-85-90-100': {
        'bbox_low_high': (0, 5),
        'chance_not_change_class': 0.85,
        'chance_not_miss_annotation': 0.90,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },
    'yolo-0_10-85-90-100': {
        'bbox_low_high': (0, 10),
        'chance_not_change_class': 0.85,
        'chance_not_miss_annotation': 0.90,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },
    'yolo-0_15-85-90-100': {
        'bbox_low_high': (0, 15),
        'chance_not_change_class': 0.85,
        'chance_not_miss_annotation': 0.90,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },

    'yolo-0_5-100-85-100': {
        'bbox_low_high': (0, 5),
        'chance_not_change_class': 1.0,
        'chance_not_miss_annotation': 0.85,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },
    'yolo-0_10-100-85-100': {
        'bbox_low_high': (0, 10),
        'chance_not_change_class': 1.0,
        'chance_not_miss_annotation': 0.85,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },
    'yolo-0_15-100-85-100': {
        'bbox_low_high': (0, 15),
        'chance_not_change_class': 1.0,
        'chance_not_miss_annotation': 0.85,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },

    'yolo-0_5-95-85-100': {
        'bbox_low_high': (0, 5),
        'chance_not_change_class': 0.95,
        'chance_not_miss_annotation': 0.85,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },
    'yolo-0_10-95-85-100': {
        'bbox_low_high': (0, 10),
        'chance_not_change_class': 0.95,
        'chance_not_miss_annotation': 0.85,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },
    'yolo-0_15-95-85-100': {
        'bbox_low_high': (0, 15),
        'chance_not_change_class': 0.95,
        'chance_not_miss_annotation': 0.85,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },

    'yolo-0_5-90-85-100': {
        'bbox_low_high': (0, 5),
        'chance_not_change_class': 0.90,
        'chance_not_miss_annotation': 0.85,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },
    'yolo-0_10-90-85-100': {
        'bbox_low_high': (0, 10),
        'chance_not_change_class': 0.90,
        'chance_not_miss_annotation': 0.85,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },
    'yolo-0_15-90-85-100': {
        'bbox_low_high': (0, 15),
        'chance_not_change_class': 0.90,
        'chance_not_miss_annotation': 0.85,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },

    'yolo-0_5-85-85-100': {
        'bbox_low_high': (0, 5),
        'chance_not_change_class': 0.85,
        'chance_not_miss_annotation': 0.85,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },
    'yolo-0_10-85-85-100': {
        'bbox_low_high': (0, 10),
        'chance_not_change_class': 0.85,
        'chance_not_miss_annotation': 0.85,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },
    'yolo-0_15-85-85-100': {
        'bbox_low_high': (0, 15),
        'chance_not_change_class': 0.85,
        'chance_not_miss_annotation': 0.85,
        'create_official': False,
        'limit_img': {
            'train': 1.0,
            'val': 1.0,
            'test': 1.0,
        },
    },
}

# Repeat all "datasets_to_create" configurations but with less data
datasets_to_create_temp = deepcopy(datasets_to_create)
for folder_name, params in datasets_to_create_temp.items():
    new_param = deepcopy(params)
    new_param['limit_img']['train'] = 0.5
    new_param['limit_img']['val'] = 0.5
    new_param['limit_img']['test'] = 0.5

    new_param2 = deepcopy(params)
    new_param2['limit_img']['train'] = 0.1
    new_param2['limit_img']['val'] = 0.1
    new_param2['limit_img']['test'] = 0.1

    datasets_to_create[f'{folder_name[:-3]}50'] = deepcopy(new_param)
    datasets_to_create[f'{folder_name[:-3]}10'] = deepcopy(new_param2)
# #### End configurations #### #

def calculate_iou(bb1, bb2):
    """
    Calculate bbox IoU of two objects (receive in YOLO format)
    Adapted from https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
    """
    bb1['x1'] = bb1['x'] / image_w
    bb1['y1'] = bb1['y'] / image_h
    bb1['x2'] = bb1['x'] + bb1['w'] / image_w
    bb1['y2'] = bb1['y'] + bb1['h'] / image_h

    bb2['x1'] = bb2['x'] / image_w
    bb2['y1'] = bb2['y'] / image_h
    bb2['x2'] = bb2['x'] + bb2['w'] / image_w
    bb2['y2'] = bb2['y'] + bb2['h'] / image_h

    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

    return iou


def change_bbox(x, y, shape, image_w, image_h, bbox_type):
    """
    Add mistakes in annotation (bbox larger/smaller, wrong class, missing object annotation)
    """
    wrong_bbox_type = 'circle' if bbox_type == 'rect' else 'rect'
    left = np.random.randint(bbox_low_high[0], bbox_low_high[1])
    up = np.random.randint(bbox_low_high[0], bbox_low_high[1])
    right = np.random.randint(bbox_low_high[0], bbox_low_high[1])
    down = np.random.randint(bbox_low_high[0], bbox_low_high[1])

    left2 = np.random.randint(bbox_low_high[0], bbox_low_high[1])
    up2 = np.random.randint(bbox_low_high[0], bbox_low_high[1])
    right2 = np.random.randint(bbox_low_high[0], bbox_low_high[1])
    down2 = np.random.randint(bbox_low_high[0], bbox_low_high[1])

    cls = bbox_type if np.random.rand() <= chance_not_change_class else wrong_bbox_type


    # Bounding box correct with wrong classe
    bbox_official = {
        'object': bbox_type,
        'x': max(min(x, image_w), 0),
        'y': max(min(y, image_h), 0),
        'w': max(min(shape[0], image_w), 1),
        'h': max(min(shape[1], image_h), 1),
        'dif_cls': False,
        'iou': 1.0
    }


    if np.random.rand() <= chance_not_miss_annotation:
        # Bounding box larger than the object
        bbox_larger = {
            'object': cls,
            'x': max(min(x - left, image_w), 0),
            'y': max(min(y - up, image_h), 0),
            'w': max(min(shape[0] + left + right, image_w), 1),
            'h': max(min(shape[1] + up + down, image_h), 1),
            'dif_cls': False if cls == bbox_type else True,
        }
        bbox_larger['iou'] = calculate_iou(bbox_official, bbox_larger)

        # Bounding box smaller than the object
        bbox_smaller = {
            'object': cls,
            'x': max(min(x + right, image_w), 0),
            'y': max(min(y + down, image_h), 0),
            'w': max(min(shape[0] - right - left, image_w), 1),
            'h': max(min(shape[1] - down - up, image_h), 1),
            'dif_cls': False if cls == bbox_type else True,
        }
        bbox_smaller['iou'] = calculate_iou(bbox_official, bbox_smaller)

        # Bounding box sometimes larger and sometimes smaller than the object
        bbox_larger_smaller = {
            'object': cls,
            'x': max(min(x + right - left, image_w), 0),
            'y': max(min(y + down - up, image_h), 0),
            'w': max(min(shape[0] + right2 - left2, image_w), 1),
            'h': max(min(shape[1] + down2 - up2, image_h), 1),
            'dif_cls': False if cls == bbox_type else True,
        }
        bbox_larger_smaller['iou'] = calculate_iou(bbox_official, bbox_larger_smaller)
    else:
        bbox_larger = None
        bbox_smaller = None
        bbox_larger_smaller = None

    if create_official is False:
        bbox_official = None

    return {
        'bbox_official': bbox_official,
        'bbox_larger': bbox_larger,
        'bbox_smaller': bbox_smaller,
        'bbox_larger_smaller': bbox_larger_smaller,
    }


def convert_bbox_to_yolo(bbox, image_w, image_h):
    # When bbox is bigger than the image
    ww = bbox['w']
    hh = bbox['h']
    diff = (bbox['w'] + bbox['x']) - image_w
    if diff > 0:
        ww = bbox['w'] - diff

    diff = (bbox['h'] + bbox['y']) - image_w
    if diff > 0:
        hh = bbox['h'] - diff

    # object, x, y, w, h
    box_normalized_xywh = (f'{classes[bbox["object"]]} '
                           f'{min((bbox["x"] + ww / 2) / image_w, 1.0)}'
                           f' {min((bbox["y"] + hh / 2) / image_h, 1.0)}'
                           f' {ww / image_w}'
                           f' {hh / image_h}')
    return box_normalized_xywh


for folder_name, params in tqdm(datasets_to_create.items()):
    path_output = f'{path_output_original}/{folder_name}'

    bbox_low_high = params['bbox_low_high']
    chance_not_change_class = params['chance_not_change_class']
    chance_not_miss_annotation = params['chance_not_miss_annotation']
    limit_img = params['limit_img']
    create_official = params['create_official']

    info_dataset = {}
    for split_folder in os.listdir(path_images):
        info_dataset[split_folder] = {}

        count_images = 0
        total_images_folder = len(os.listdir(f'{path_images}/{split_folder}'))
        for filename in os.listdir(f'{path_images}/{split_folder}'):
            if count_images >= total_images_folder * limit_img[split_folder]:
                break

            count_images += 1
            img = Image.open(f'{path_images}/{split_folder}/{filename}')
            image_w, image_h = img.size
            with open(f'{path_labels_json}/{split_folder}/{filename[:-3]}json') as file:
                json_data = json.load(file)
            bboxs = []
            for obj in json_data:
                bboxs.append(change_bbox(obj['x'], obj['y'], [obj['w'], obj['h']], image_w, image_h, obj['object']))

            result_normalized = {}
            # Create dict with all changes
            for bb in bboxs:
                for key, bbox in bb.items():
                    if bbox is None:
                        continue
                    result_normalized[key] = []
                    if key not in info_dataset[split_folder]:
                        info_dataset[split_folder][key] = {'sum_iou': 0.0, 'total_dif_cls': 0, 'total_missing_bbox': 0, 'total_bbox': 0}
                        break

            num_dif_cls = 0
            for idx, bbox_dict in enumerate(bboxs):
                for key, bbox in bbox_dict.items():
                    if bbox is None:
                        if key in info_dataset[split_folder]:
                            info_dataset[split_folder][key]['total_missing_bbox'] += 1
                        continue
                    result_normalized[key].append(convert_bbox_to_yolo(bbox, image_w, image_h))
                    info_dataset[split_folder][key]['total_dif_cls'] += bbox['dif_cls']
                    info_dataset[split_folder][key]['sum_iou'] += bbox['iou']
                    info_dataset[split_folder][key]['total_bbox'] += 1

            for key, bboxs_normalized in result_normalized.items():
                # Save new annotations
                Path(f'{path_output}/{key}/{split_folder}/labels').mkdir(parents=True, exist_ok=True)
                with open(f'{path_output}/{key}/{split_folder}/labels/{filename[:-3]}txt', 'w') as outfile:
                    for line in bboxs_normalized:
                        outfile.write(f'{line}\n')

                # Copy imagens with the annotations (not recommended if there are many folders)
                #Path(f'{path_output}/{key}/{split_folder}/images').mkdir(parents=True, exist_ok=True)
                #shutil.copy(
                #    f'{path_images}/{split_folder}/{filename}',
                #    f'{path_output}/{key}/{split_folder}/images/{filename}'
                #)

    # Save yaml in YOLO format and some informations about the annotations mistakes
    for key, info in info_dataset['train'].items():
        with open(f'{path_output}/{key}/dataset.yaml', 'w') as outfile:
            dict_yaml = {
                'train': './train/images',
                'val': './val/images',
                'test': './test/images',

                'nc': len(classes),
                'names': list(classes.keys()),

                'wrong_cls_train': info_dataset['train'][key]['total_dif_cls'] / info_dataset['train'][key]['total_bbox'],
                'mean_iou_train': info_dataset['train'][key]['sum_iou'] / info_dataset['train'][key]['total_bbox'],
                'missing_bbox_train': info_dataset['train'][key]['total_missing_bbox'] / (info_dataset['train'][key]['total_missing_bbox'] + info_dataset['train'][key]['total_bbox']),
                'wrong_cls_test': info_dataset['test'][key]['total_dif_cls'] / info_dataset['test'][key]['total_bbox'],
                'mean_iou_test': info_dataset['test'][key]['sum_iou'] / info_dataset['test'][key]['total_bbox'],
                'missing_bbox_test': info_dataset['test'][key]['total_missing_bbox'] / (info_dataset['test'][key]['total_missing_bbox'] + info_dataset['test'][key]['total_bbox']),
                'wrong_cls_val': info_dataset['val'][key]['total_dif_cls'] / info_dataset['val'][key]['total_bbox'],
                'mean_iou_val': info_dataset['val'][key]['sum_iou'] / info_dataset['val'][key]['total_bbox'],
                'missing_bbox_val': info_dataset['val'][key]['total_missing_bbox'] / (info_dataset['val'][key]['total_missing_bbox'] + info_dataset['val'][key]['total_bbox'])
            }
            # print(f'{key} - wrong_cls_train: {dict_yaml["wrong_cls_train"]} - mean_iou_train: {dict_yaml["mean_iou_train"]}')
            yaml.dump(dict_yaml, outfile)