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




#------------------------
# 1. Chargement du modèle dans le conteneur Docker
#------------------------
model_path = "/docker_models/VGG16_unet/data/model.keras"
model = keras.models.load_model(model_path)
# model = mlflow.tensorflow.load_model(model_path)


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