import tensorflow as tf
import numpy as np
# import albumentations as A
# from tensorflow.keras.utils import to_categorical
# from albumentations.core.composition import OneOf
import cv2


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, mask_paths, batch_size, img_size,
                 num_classes=8, augmentation=None, shuffle=True):
        """
        Data Generator pour la segmentation d'images

        image_paths : Liste des chemins des images
        mask_paths : Liste des chemins des masques correspondants
        batch_size : Nombre d'images par batch
        img_size : Taille des images après resize
        num_classes : Nombre de classes dans le masque
        augmentation : Transformations Albumentations (à la volée - non appliqué ici car traité en amont)
        shuffle : Mélange les données à chaque epoch
        """
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_classes = num_classes
        # self.augmentation = augmentation
        self.shuffle = shuffle
        # self.one_hot = one_hot
        self.on_epoch_end()

    def __len__(self):
        """Nombre de batches par epoch"""
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        """Retourne un batch d'images et de masques"""
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        image_paths_temp = [self.image_paths[k] for k in indexes]
        mask_paths_temp = [self.mask_paths[k] for k in indexes]

        return self.__data_generation(image_paths_temp, mask_paths_temp)

    def on_epoch_end(self):
        """Mélange les données à la fin d'un epoch si shuffle=True"""
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, image_paths_temp, mask_paths_temp):
        """Charge et pré-traite un batch de données"""
        X = np.zeros((self.batch_size, *self.img_size, 3), dtype=np.float32)
        Y = np.zeros((self.batch_size, *self.img_size), dtype=np.uint8)

        for i, (img_path, mask_path) in enumerate(zip(image_paths_temp, mask_paths_temp)):
            # Chargement de l'image
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertir en RGB
            img = cv2.resize(img, self.img_size)  # Redimensionner
            img = img.astype(np.float32) / 255.0  # Normalisation

            # Chargement du masque
            mask = np.load(mask_path)  # Chargement du masque
            mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)  # Resize sans modifier les labels
            # mask = mask.astype(np.uint8)

            # # Appliquer la data augmentation si définie
            # if self.augmentation:
            #     augmented = self.augmentation(image=img, mask=mask)
            #     img, mask = augmented["image"], augmented["mask"]

            X[i] = img
            Y[i] = mask

        return X, Y
