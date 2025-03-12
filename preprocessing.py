import cv2
import numpy as np

def preprocess_image(image_path, target_size=(256, 256)):
    """Charge et applique les transformations à une image."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0  # Normalisation
    return image
