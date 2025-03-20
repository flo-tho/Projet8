import mlflow
import os

# Récupérer l'URI de tracking MLflow depuis une variable d'environnement
mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")  #si var d'env pas défini, on a une URI par défaut
mlflow.set_tracking_uri(mlflow_tracking_uri)

# Définition du répertoire de destination pour les modèles (dans le dépôt GitHub)
models_dir = "docker_models"

# On s'assure que le dossier "models" existe:
os.makedirs(models_dir, exist_ok=True)

run_et_model_id = "615947391303416845/f1a90ff0bad347788cdeea4fb1ef6cdf"

# Téléchargement du modèle
model_uri = f"mlflow-artifacts:/{run_et_model_id}/artifacts/VGG16_unet"
model_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=models_dir)