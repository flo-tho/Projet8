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

mlflow.set_tracking_uri('http://127.0.0.1:5000/')
model_uri = "mlflow-artifacts:/615947391303416845/3ea901938c0446ebb1bbe9729f403322/artifacts/VGG16_unet"
model_path = mlflow.artifacts.download_artifacts(model_uri)
model_path = os.path.join(model_path, "data/model.keras")
model = keras.models.load_model(model_path)

IMAGE_DIR = "images/test"  # Répertoire des images de test

@app.get("/images/")
async def list_images():
    try:
        images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]
        return {"images": images}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        original_size = image.size
        image = np.array(image)
        image = cv2.resize(image, (256, 256)).astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)

        predictions = model.predict(image)[0]
        predicted_mask = np.argmax(predictions, axis=-1)
        predicted_mask = cv2.resize(predicted_mask, original_size, interpolation=cv2.INTER_NEAREST)

        mask_bytes = io.BytesIO()
        np.save(mask_bytes, predicted_mask)
        mask_bytes.seek(0)

        return Response(content=mask_bytes.getvalue(), media_type="application/octet-stream")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)