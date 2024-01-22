import os
import json
import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


np.random.seed(42)

# #### Configurations #### #
# Object colors available
shape_color = [
    "blue", "green", "red", "cyan",
    "magenta", "yellow", "black"]  # "white"
# Range object alpha color
color_alpha = [0.3, 1.0]  # Min and Max
# Range object shape
shape_attribs = {"rect": [10, 50], "circle": [10, 40]}  # Min and max
# Range objects per image
obj_per_image = (15, 40)  # Min and Max
# Classes to id
classes = {
    "rect": 0,
    "circle": 1
}
# Shapes to draw
shapes = ["rect", "circle"]
# Number of images to create
num_images = 100
# Split dataset
split_dataset = {
    "train": 0.60,
    "val": 0.20,
    "test": 0.20
}
# Image size
image_size = (640, 640)
# Folder to save images
path = "D:/teste/"
save_dir = path + "/tmp/"
# Annotation type (only tested on detections)
task_type = "detection"
# #### End configurations #### #

# need to make an option for setting up the attribs dynamically
shapes = list(set(shapes))
image_w = image_size[0]
image_h = image_size[1]


def calculate_iou(bb1, bbs2):
    """
    Calculate bbox IoU of two objects
    Font: https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
    """
    if len(bb1) == 0:
        return 1.0
    if len(bbs2) == 0:
        return 0.0

    result = [0.0]

    for bb2 in bbs2:
        x_left = max(bb1["x1"], bb2["x1"])
        y_top = max(bb1["y1"], bb2["y1"])
        x_right = min(bb1["x2"], bb2["x2"])
        y_bottom = min(bb1["y2"], bb2["y2"])

        if x_right < x_left or y_bottom < y_top:
            result.append(0.0)
            continue

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (bb1["x2"] - bb1["x1"]) * (bb1["y2"] - bb1["y1"])
        bb2_area = (bb2["x2"] - bb2["x1"]) * (bb2["y2"] - bb2["y1"])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        result.append(iou)

    return max(result)


def create_bbox(obj_type):
    """
    Generate random coordinates to draw the object and create bbox annotation

    Args:
        obj_type (int): ID of the class to be created
    Returns:
        obj (matplotlib object): Object created in Matplotlib
        bbox (dict): Object information (class, coordinates X, Y, W, H)
        bbox_normalized_xyxy (dict): Object coordinates converted to X1, Y1, X2, Y2
    """
    # Start position of the object (X, Y)
    x = np.random.randint(image_w * 0.01, image_w * 0.98)
    y = np.random.randint(image_h * 0.01, image_h * 0.98)

    # Create object
    obj, shape = _create_object(x, y, obj_type)
    # Create bbox
    bbox = _gen_bbox(x, y, shape, obj_type)

    # Convert bbox to X1, Y1, X2, Y2
    bbox_normalized_xyxy = {"x1": bbox["x"] / image_w,
                            "y1": bbox["y"] / image_h,
                            "x2": (bbox["w"] + bbox["x"]) / image_w,
                            "y2": (bbox["h"] + bbox["y"]) / image_h}

    return obj, bbox, bbox_normalized_xyxy


def _create_object(x, y, obj_type):
    """
    Create object in Matplotlib format
    """
    color = shape_color[np.random.randint(0, len(shape_color))]

    if shapes[obj_type] == "rect":
        shape = [
            np.random.randint(shape_attribs["rect"][0], shape_attribs["rect"][1]),
            np.random.randint(shape_attribs["rect"][0], shape_attribs["rect"][1])
        ]
        return plt.Rectangle((x, y), shape[0],
                             shape[1], color=color, alpha=np.random.uniform(color_alpha[0], color_alpha[1]),
                             lw=0), shape
    elif shapes[obj_type] == "circle":
        rad = np.random.randint(shape_attribs["circle"][0], shape_attribs["circle"][1])
        shape = [rad, rad]
        return plt.Circle((x, y), shape[0],
                          color=color, alpha=np.random.uniform(color_alpha[0], color_alpha[1]), lw=0), shape


def _gen_bbox(x, y, shape, obj_type):
    """
    Generate object bbox respecting the image shape
    """
    if shapes[obj_type] == "rect":
        return {
            "object": "rect", "x": max(min(x, image_w), 0), "y": max(min(y, image_h), 0),
            "w": max(min(shape[0], image_w), 1), "h": max(min(shape[1], image_h), 1)
        }

    elif shapes[obj_type] == "circle":
        xx = max(x - shape[0], 0)
        yy = max(y - shape[0], 0)
        ww = max(min(2 * shape[0], image_w), 0)
        hh = max(min(2 * shape[0], image_h), 0)

        if xx == 0:
            ww += x - shape[0]
        if yy == 0:
            hh += y - shape[0]

        return {"object": "circle", "x": xx, "y": yy, "w": ww, "h": hh}


def detection_gen():
    def create_dirs():
        img_path = os.path.join(save_dir, "dataset", "images")
        lab_path = os.path.join(save_dir, "dataset", "labels_json")

        for split in split_dataset:
            Path(f"{img_path}/{split}").mkdir(parents=True, exist_ok=True)
            Path(f"{lab_path}/{split}").mkdir(parents=True, exist_ok=True)

        return img_path, lab_path

    img_path, lab_path = create_dirs()

    for split, percentage in split_dataset.items():
        for n in tqdm.tqdm(range(int(num_images * percentage))):
            objs = []
            obj_bbox = []
            obj_bbox_normalized_xyxy = []
            # Maximum draws that will be created in the images
            objs_num = np.random.randint(obj_per_image[0], obj_per_image[1])
            for i in range(objs_num):
                obj_type = np.random.randint(0, len(shapes))
                bbox_normalized_xyxy = []
                # random x, y cord gen
                count = 0
                # Try N times to add a new object position. Not accept very high IoU
                while calculate_iou(bbox_normalized_xyxy, obj_bbox_normalized_xyxy) > 0.0 and count < 10:
                    obj, bbox, bbox_normalized_xyxy = create_bbox(obj_type)
                    count += 1

                if count >= 10:
                    continue

                obj_bbox_normalized_xyxy.append(bbox_normalized_xyxy)
                objs.append(obj)
                obj_bbox.append(bbox)

            # Draw objects
            fig, ax = plt.subplots(figsize=(image_w / 100, image_h / 100))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim([0, image_w])
            ax.set_ylim([0, image_h])
            plt.gca().invert_yaxis()

            for i, obj in enumerate(objs):
                ax.add_artist(obj)

            # Save image and annotations
            fig.savefig("%s/%s/shapes_%d.png" % (img_path, split, n))
            with open("%s/%s/shapes_%d.json" % (lab_path, split, n), "w") as outfile:
                json.dump(obj_bbox, outfile)

        plt.clf()
        plt.close()
    print("Generated dataset in %s" % save_dir)


"""
def classification_gen():
    def create_dirs():
        for shape in shapes:
            try:
                os.makedirs(os.path.join(save_dir, "dataset", shape))
            except FileExistsError as e:
                pass

    # image_w = image_h =
    create_dirs()
    for n in tqdm.tqdm(range(num_images)):

        obj_i = int(n / (num_images / len(shapes)))
        if list(shape_attribs.keys())[obj_i] == "rect":
            rect_w = rect_h = np.random.randint(image_w / 4, 3 * image_w / 4)
            shape_attribs["rect"] = [rect_w, rect_h]

            x = np.random.randint(
                0,
                image_w / 4)
            y = np.random.randint(
                0,
                image_h / 4)
        if list(shape_attribs.keys())[obj_i] == "circle":
            rad = np.random.randint(image_w / 6, image_w / 4)
            shape_attribs["circle"] = [rad]

            x = np.random.randint(
                2 * rad,
                image_w - 2 * rad)
            y = np.random.randint(
                2 * rad,
                image_h - 2 * rad)
        fig, ax = plt.subplots(
            figsize=(image_w / 100, image_h / 100))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim([0, image_w])
        ax.set_ylim([0, image_h])
        plt.gca().invert_yaxis()
        ax.add_artist(create_object(x, y, obj_i))
        fig.savefig(
            "%s/%s_%d.png"
            % (os.path.join(save_dir, "dataset", shapes[obj_i]), shapes[obj_i], n))
        plt.clf()
        plt.close()
    print("Generated dataset in %s" % save_dir)

def segmentation_gen():
    # segmentation and recogition datasets are pretty much the same at this point
    classification_gen()


if task_type == "classification":
    classification_gen()
elif task_type == "detection":
    detection_gen()
elif task_type == "segmentation":
    segmentation_gen()
"""

if task_type == "detection":
    detection_gen()
else:
    raise ValueError(f"Task type {task_type} not supported!")