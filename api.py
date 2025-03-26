import io
# import os
import uvicorn
import numpy as np
# import mlflow
import tensorflow as tf
import keras
from keras.saving import register_keras_serializable
from fastapi import FastAPI, File, UploadFile, Response
from PIL import Image
import cv2


app = FastAPI()
@register_keras_serializable(package="Custom", name="MeanIoUMetric")
class MeanIoUMetric(tf.keras.metrics.Metric):
    def __init__(self, num_classes=8, name="mean_iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.miou = tf.keras.metrics.MeanIoU(num_classes=num_classes)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        self.miou.update_state(y_true, y_pred)

    def result(self):
        return self.miou.result()

    def reset_state(self):
        self.miou.reset_state()


@register_keras_serializable(package="Custom", name="dice_loss")
def dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Dice Loss pour segmentation multi-classes.

    Args:
        y_true: masque de vérité terrain (shape: batch, height, width).
        y_pred: prédictions du modèle (shape: batch, height, width, num_classes).
        smooth: petit facteur pour éviter les divisions par zéro.

    Returns:
        Dice loss scalaire.
    """
    num_classes = tf.shape(y_pred)[-1]
    y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)  # One-hot encoding

    # Calcul du Dice coefficient
    intersection = tf.reduce_sum(y_true_one_hot * y_pred, axis=[1, 2, 3])  # Somme par batch
    denominator = tf.reduce_sum(y_true_one_hot, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])

    dice_coeff = (2. * intersection + smooth) / (denominator + smooth)
    return 1 - tf.reduce_mean(dice_coeff)


@register_keras_serializable(package="Custom", name="balanced_cross_entropy")
def balanced_cross_entropy(beta=0.5):
    """
    Balanced Cross Entropy (BCE pondérée).
    """

    def loss(y_true, y_pred):
        num_classes = tf.shape(y_pred)[-1]
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)  # One-hot encoding

        # Éviter log(0) en contraignant y_pred entre [1e-6, 1-1e-6]
        y_pred = tf.clip_by_value(y_pred, 1e-6, 1 - 1e-6)

        # Calcul de la BCE pondérée
        bce = - (beta * y_true_one_hot * tf.math.log(y_pred) +
                 (1 - beta) * (1 - y_true_one_hot) * tf.math.log(1 - y_pred))
        return tf.reduce_mean(bce)

    return loss

@register_keras_serializable(package="Custom", name="total_loss")
def total_loss(y_true, y_pred, dice_weight=0.5, bce_weight=0.5, beta=0.5):
    """
    Combinaison de Dice Loss et Balanced Cross Entropy.
    """
    return dice_weight * dice_loss(y_true, y_pred) + bce_weight * balanced_cross_entropy(beta)(y_true, y_pred)


#------------------------
# 1. Chargement du modèle dans le conteneur Docker
#------------------------
model_path = "/docker_models/VGG16_unet_total_loss/data/model.keras"

try:
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path, custom_objects={"MeanIoUMetric": MeanIoUMetric,"total_loss": total_loss, "dice_loss": dice_loss, "balanced_cross_entropy": balanced_cross_entropy})
    # model = mlflow.tensorflow.load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")




# # Code pour la version locale (direct MLflow, non utilisé sur le cloud)
# mlflow.set_tracking_uri('http://127.0.0.1:5000/')
# model_uri = "mlflow-artifacts:/615947391303416845/f1a90ff0bad347788cdeea4fb1ef6cdf/artifacts/VGG16_unet"
# model_path = mlflow.artifacts.download_artifacts(model_uri)
# model_path = os.path.join(model_path, "data/model.keras")
# model = keras.models.load_model(model_path)

# Fonction de prétraitement de l'image
def preprocess_image(image: Image.Image, target_size=(256, 256)):
    original_size = image.size
    image = np.array(image)
    image = cv2.resize(image, target_size)
    image = image.astype(np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image, original_size


#------------------------
# 2. Prediction
#------------------------
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model is not loaded properly"}
    try:
        # Lire l'image et la convertir
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        input_data, original_size = preprocess_image(image)

        # Prédiction du masque
        predictions = model.predict(input_data)[0]
        predicted_mask = np.argmax(predictions, axis=-1)
        predicted_mask = cv2.resize(predicted_mask, original_size, interpolation=cv2.INTER_NEAREST)

        # Sérialiser le masque en mémoire sans sauvegarde sur disque
        npy_bytes = io.BytesIO()
        np.save(npy_bytes, predicted_mask)
        npy_bytes.seek(0)

        return Response(content=npy_bytes.getvalue(), media_type="application/octet-stream")

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)