
#FROM mcr.microsoft.com/devcontainers/python:1.2.2-3.12-bookworm
## si on veut partir de l'image Python 3.12-slim, hebergée directement sur Azure Container Registry:
#FROM registryprojet8.azurecr.io/python:3.12-slim

## Utilisation directe de Python 3.12 en local
FROM python:3.12-slim

# Création d'un répertoire pour l'application
WORKDIR /api

# Copie et installation des dépendances
COPY requirements.txt /api/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copie des fichiers utiles pour l'application
COPY api.py /api/


# Copie du modèle téléchargé en amont du build, avec Git Hub Actions depuis MLflow (local)
RUN mkdir -p /docker_models
COPY docker_models /docker_models

RUN ls -R /docker_models

# Exposition du port 8000
EXPOSE 8000

# Commande pour démarrer l'application avec uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]