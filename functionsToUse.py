import os
from PIL import Image
import numpy as np
import cv2
import imageio
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class CustomDataGenerator:
    def __init__(self, directory, image_size, batch_size, label_mode='int', subset='training', seed=123, validation_split=0.2):
        self.directory = directory
        self.image_size = image_size
        self.batch_size = batch_size
        self.label_mode = label_mode
        self.subset = subset
        self.seed = seed
        self.validation_split = validation_split
        self.classes = sorted(os.listdir(directory))
        self.num_classes = len(self.classes)

        if label_mode == 'int':
            self.class_indices = {cls: i for i, cls in enumerate(self.classes)}
        else:
            self.class_indices = None

        self.all_image_paths, self.all_image_labels = self.load_image_paths_and_labels()

        # Embaralhar os dados
        np.random.seed(seed)
        indices = np.arange(len(self.all_image_paths))
        np.random.shuffle(indices)
        self.all_image_paths = np.array(self.all_image_paths)[indices]
        self.all_image_labels = np.array(self.all_image_labels)[indices]

        # Dividir entre treinamento e validação
        split = int((1 - validation_split) * len(self.all_image_paths))
        self.train_image_paths, self.val_image_paths = self.all_image_paths[:split], self.all_image_paths[split:]
        self.train_labels, self.val_labels = self.all_image_labels[:split], self.all_image_labels[split:]

        self.current_class_names = self.classes

    def load_image_paths_and_labels(self):
        all_image_paths = []
        all_image_labels = []

        for class_name in self.classes:
            class_path = os.path.join(self.directory, class_name)
            image_paths = [os.path.join(class_path, img) for img in os.listdir(class_path)]
            all_image_paths.extend(image_paths)
            all_image_labels.extend([self.class_indices[class_name]] * len(image_paths))

        return all_image_paths, all_image_labels

    #def process_images_optimized(self, image_paths, labels):
    #    images = [np.array(Image.open(img_path).resize((self.image_size[1], self.image_size[0]))) / 255.0 for img_path in image_paths]
    #    return np.array(images), np.array(labels)

    def process_images_optimized(self, image_paths, labels):
        processed_images = []
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            batch_labels = labels[i:i + self.batch_size]
            batch_images = self.load_and_resize_images(batch_paths)
            processed_images.extend(batch_images)

        return np.array(processed_images), np.array(labels)

    def load_and_resize_image(self, img_path):
        with Image.open(img_path) as img:
            img = img.resize((self.image_size[1], self.image_size[0]))
            img_array = np.array(img) / 255.0
        return img_array

    def load_and_resize_images(self, image_paths):
        with ThreadPoolExecutor() as executor:
            images = list(executor.map(self.load_and_resize_image, image_paths))
        return images

    def generate_batches(self, image_paths, labels):
        while True:
            for i in range(0, len(image_paths), self.batch_size):
                batch_image_paths = image_paths[i:i+self.batch_size]
                batch_labels = labels[i:i+self.batch_size]
                batch_images, batch_labels = self.process_images(batch_image_paths, batch_labels)
                yield batch_images, batch_labels

    def get_train_generator(self):
        return self.generate_batches(self.train_image_paths, self.train_labels)

    def get_val_generator(self):
        return self.generate_batches(self.val_image_paths, self.val_labels)

    def get_class_names(self):
        return self.current_class_names