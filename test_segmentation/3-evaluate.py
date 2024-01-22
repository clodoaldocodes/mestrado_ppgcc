import fiftyone as fo
from fiftyone import ViewField as F
import fiftyone.utils.ultralytics as fou

import pandas as pd
import os, shutil, yaml
from pathlib import Path

from ultralytics import YOLO
import variable_direc


# #### Configurations #### #
# Path labels that will be used to evaluate the models
path_labels = variable_direc.path + '/tmp/dataset/yolo/yolo-0_5-100-100-100/bbox_official'
# Path images
path_images = variable_direc.path + '/tmp/dataset/images'

# Path to models to be evaluated
path_yolo_models = variable_direc.path + '/tmp/dataset/yolo'

# Splits dataset to be used
splits = ['test']

# Save fiftyone inferences (not recommended, it is slow)
is_save_fiftyone = False
# #### End configurations #### #

def save_results(all_results, dataset, sufix_name=''):
    """ Save results to csv file """
    columns = [
        'type', 'margin', 'right_class', 'all_class', 'size_dataset',
        'train/bbox_loss', 'train/cls_loss', 'train/dfl_loss', 'train/num_epochs', 'train/mean_time_epoch', 'train/total_time',
        'val/precision', 'val/recall', 'val/mAP50', 'val/mAP50-95', 'val/box_loss', 'val/cls_loss', 'val/dfl_loss',
        'test/accuracy', 'test/precision', 'test/recall', 'test/fscore', 'test/mAP50-95',
        'test/precision-circle', 'test/recall-circle', 'test/fscore-circle',
        'test/precision-rect', 'test/recall-rect', 'test/fscore-rect',
        'test/mean_confidence', 'test/mean_IoU',
        'dataset/mean_IoU_train', 'dataset/mean_IoU_val', 'dataset/missing_bbox_train', 'dataset/missing_bbox_val',
        'dataset/wrong_cls_train', 'dataset/wrong_cls_val',
    ]
    df_all_results = pd.DataFrame.from_dict(all_results, orient='index', columns=columns)
    df_all_results.index.name = 'model_name'
    df_all_results.to_csv('all_results.csv')
    
    # Save fiftyone dataset with all results
    if is_save_fiftyone:
        dataset.write_json(f'all_results_fiftyone{sufix_name}.json')

def load_fiftyone_dataset():
    # Copy images to the same folder as labels (YOLOv5 format)
    for folder in splits:
        for filename in os.listdir(f'{path_labels}/{folder}/labels'):
            Path(f'{path_labels}/{folder}/images').mkdir(parents=True, exist_ok=True)
            shutil.copy(f'{path_images}/{folder}/{filename[:-3]}png', f'{path_labels}/{folder}/images')

    # Load dataset to FiftyOne
    dataset = fo.Dataset()
    for split in splits:
        dataset.add_dir(
            dataset_dir=path_labels,
            dataset_type=fo.types.YOLOv5Dataset,
            split=split,
            tags=split,
    )
    return dataset

dataset = load_fiftyone_dataset()

all_results = {}
# Evaluate all models
for idx, folder_train in enumerate(os.listdir(path_yolo_models)):
    print(f'##### Starting evaluate {folder_train} ({idx + 1}-{len(os.listdir(path_yolo_models))}) #####')
    #if idx + 1 < 106:
    #    continue
    for folder_especific_train in os.listdir(f'{path_yolo_models}/{folder_train}'):
        print(f'Folder: {folder_especific_train}')
        model_name = f'{folder_train}-{folder_especific_train}'
        full_path = f'{path_yolo_models}/{folder_train}/{folder_especific_train}'
        
        # Load training results
        with open(f'{full_path}/dataset_final.yaml', 'r') as file:
            yaml_dict = yaml.safe_load(file)

        # Load best model
        model = YOLO(f'{full_path}/runs/train/weights/best.pt')

        # Inferences test dataset and save results in FiftyOne
        default_classes = dataset.default_classes
        for sample in dataset.iter_samples(progress=True):
            result = model(sample['filepath'], verbose=False)
            sample[model_name] = fou.to_detections(result[0])
            sample.save()
        
        # Evaluate test inferences
        results = dataset.evaluate_detections(
            pred_field=model_name,
            gt_field='ground_truth',
            eval_key=f'eval_{model_name.replace("-", "_")}',
            compute_mAP=True,  # 0.5, .05, .95 (COCO metrics)
            use_boxes=True
        )

        model_name_splited = model_name.split('-')
        # Get all results
        all_results[model_name] = [
            model_name_splited[-1],
            model_name_splited[1],
            model_name_splited[2],
            model_name_splited[3],
            model_name_splited[4],
            model.ckpt['train_results']['train/box_loss'][-1],
            model.ckpt['train_results']['train/cls_loss'][-1],
            model.ckpt['train_results']['train/dfl_loss'][-1],
            model.ckpt['train_results']['epoch'][-1],
            yaml_dict['total_training_time'] / 60 / model.ckpt['train_results']['epoch'][-1],
            yaml_dict['total_training_time'] / 60,
            model.ckpt['train_metrics']['metrics/precision(B)'],
            model.ckpt['train_metrics']['metrics/recall(B)'],
            model.ckpt['train_metrics']['metrics/mAP50(B)'],
            model.ckpt['train_metrics']['metrics/mAP50-95(B)'],
            model.ckpt['train_metrics']['val/box_loss'],
            model.ckpt['train_metrics']['val/cls_loss'],
            model.ckpt['train_metrics']['val/dfl_loss'],
            results.metrics()['accuracy'],
            results.metrics()['precision'],
            results.metrics()['recall'],
            results.metrics()['fscore'],
            results.mAP(),
            results.report()['circle']['precision'],
            results.report()['circle']['recall'],
            results.report()['circle']['f1-score'],
            results.report()['rect']['precision'],
            results.report()['rect']['recall'],
            results.report()['rect']['f1-score'],
            dataset.mean(F(f'{model_name}.detections.confidence')),
            dataset.mean(F(f'{model_name}.detections.eval_{model_name.replace("-", "_")}_iou')),
            yaml_dict['mean_iou_train'],
            yaml_dict['mean_iou_val'],
            yaml_dict['missing_bbox_train'],
            yaml_dict['missing_bbox_val'],
            yaml_dict['wrong_cls_train'],
            yaml_dict['wrong_cls_val'],
        ]

    if idx + 1 != 1 and (idx + 1) % 3 == 0:
        print('Saving results...')
        save_results(all_results, dataset, sufix_name=f'-{idx - 1}_{idx + 1}')
        if is_save_fiftyone:
            dataset = load_fiftyone_dataset()

    if not is_save_fiftyone:
        dataset = load_fiftyone_dataset()

save_results(all_results, dataset, sufix_name=f'-{idx - 1}_{idx + 1}')

for folder in splits:
    shutil.rmtree(f'{path_labels}/{folder}/images')