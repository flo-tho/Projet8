import io
import os
import mlflow
import numpy as np
import tensorflow as tf
import keras
from keras.saving import register_keras_serializable
import uvicorn
import cv2
from fastapi import FastAPI, HTTPException, File, UploadFile, Response
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Définition de la classe MeanIoUMetric
@register_keras_serializable(package="Custom", name="MeanIoUMetric")
class MeanIoUMetric(tf.keras.metrics.Metric):
    def __init__(self, num_classes=8, name="mean_iou", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.miou = tf.keras.metrics.MeanIoU(num_classes=num_classes)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)  # Convertir en classes prédictes
        self.miou.update_state(y_true, y_pred)

    def result(self):
        return self.miou.result()

    def reset_state(self):
        self.miou.reset_state()

# Dictionnaire de couleurs avec des clés en entiers
color_map = {
    0: [50, 50, 50],     # void (gris foncé neutre)
    1: [210, 180, 140],  # flat (ton sable/taupe)
    2: [220, 120, 60],   # construction (brique/orange)
    3: [255, 215, 0],    # object (jaune doré)
    4: [60, 180, 75],    # nature (vert dynamique)
    5: [135, 206, 235],  # sky (bleu ciel clair)
    6: [255, 0, 0],      # human (rouge vif)
    7: [0, 0, 255]       # vehicle (bleu vif)
}


def apply_color_map(mask):
    """
    Applique une colormap personnalisée à un masque en niveaux de gris.
    """
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for label, color in color_map.items():
        color_mask[mask == label] = np.array(color, dtype=np.uint8)  # Assurez-vous du bon format

    return color_mask

#------------------------
# 1. Chargement du tokenizer et du modèle dans le conteneur Docker
#------------------------
#
# tokenizer_path = "/models/best_distilbert_tokenizer"
# tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
#
# model_path = "/models/model"
# model = mlflow.tensorflow.load_model(model_path)

# Code pour la version locale (direct MLflow, non utilisé sur le cloud)
# Définition de l'URI du modèle MLflow
# mlflow.set_tracking_uri('http://host.docker.internal:5000')
mlflow.set_tracking_uri('http://127.0.0.1:5000/')
model_uri = "mlflow-artifacts:/615947391303416845/3ea901938c0446ebb1bbe9729f403322/artifacts/VGG16_unet"

# Chargement du modèle TensorFlow
model_path = mlflow.artifacts.download_artifacts(model_uri)  # Télécharge le modèle localement
model_path = os.path.join(model_path, "data/model.keras")
model = keras.models.load_model(model_path)  # Charge le modèle au bon format


# Fonction de prétraitement de l'image
def preprocess_image(image: Image.Image, target_size=(256, 256)):
    original_size = image.size  # Stocker la taille originale
    image = np.array(image)
    image = cv2.resize(image, target_size)  # Redimensionner
    image = image.astype(np.float32) / 255.0  # Normalisation
    image = np.expand_dims(image, axis=0)  # Ajout de la dimension batch
    return image, original_size


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Chargement de l'image
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        input_data, original_size = preprocess_image(image)

        # Prédiction du masque
        predictions = model.predict(input_data)[0]  # Prédiction sur une seule image
        predicted_mask = np.argmax(predictions, axis=-1)  # Classification des pixels

        # Rétablir la taille originale du masque
        predicted_mask = cv2.resize(predicted_mask, original_size, interpolation=cv2.INTER_NEAREST)


        color_mask = apply_color_map(predicted_mask)

        # Conversion en image PNG
        mask_image = Image.fromarray(color_mask.astype(np.uint8), mode="RGB")
        img_byte_arr = io.BytesIO()
        mask_image.save(img_byte_arr, format="PNG")
        img_byte_arr = img_byte_arr.getvalue()

        return Response(content=img_byte_arr, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

