# lancer streamlit avec streamlit run interface.py

import streamlit as st
import requests
import numpy as np
import io
import os
from PIL import Image
import matplotlib.pyplot as plt

# URL de l'API hébergée sur Azure
API_URL = "https://appprojet8seg-e4audkeuaxa9hwaj.francecentral-01.azurewebsites.net/predict/"
# API_URL = "http://127.0.0.1:8000/predict/"

# Récupérer les secrets
AZURE_BLOB_URL = st.secrets["azure"]["blob_url"]
SAS_TOKEN = st.secrets["azure"]["sas_token"]

st.title("Segmentation d'Images - Test API")

# Sélection de l'image locale
uploaded_file = st.file_uploader("Choisissez une image", type=["png", "jpg", "jpeg"])

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
        color_mask[mask == label] = color  # Remplace chaque pixel par sa couleur

    return color_mask

def get_original_mask(image_name):
    """
    Récupère le chemin d'accès (URL) du masque original correspondant au nom de l'image depuis Azure Blob Storage.
    """
    base_name = image_name.split(".")[0].replace("_leftImg8bit", "")
    # Construire l'URL du masque dans Azure Blob Storage
    mask_url = f"{AZURE_BLOB_URL}{base_name}_gtFine_labelIds.npy?{SAS_TOKEN}"
    
    # Retourner l'URL du masque (chemin d'accès sur Azure)
    return mask_url

if uploaded_file is not None:
    # Afficher l'image originale
    image = Image.open(uploaded_file)
    st.image(image, caption="Image originale", use_container_width=True)

    # Bouton pour envoyer l'image à l'API
    if st.button("Lancer la prédiction"):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            # Vérifier le type de contenu de la réponse
            if response.headers['Content-Type'] == 'application/json':
                st.error("L'API a renvoyé une erreur sous forme de JSON.")
                st.json(response.json())  # Affiche l'erreur sous forme de JSON pour plus de détails
            else:
                try:
                    # Charger le masque depuis la réponse
                    npy_bytes = io.BytesIO(response.content)
                    predicted_mask = np.load(npy_bytes)
                    predicted_mask = apply_color_map(predicted_mask)

                    # Affichage du masque prédit
                    st.subheader("Masque Prédit")
                    fig, ax = plt.subplots()
                    ax.imshow(predicted_mask)
                    ax.axis("off")
                    st.pyplot(fig)

                    # Récupérer l'URL du masque original depuis Azure Blob Storage
                    original_mask_url = get_original_mask(uploaded_file.name)
                    if original_mask_url:
                        # Télécharger le masque original depuis Azure Blob Storage
                        response = requests.get(original_mask_url)
                        if response.status_code == 200:
                            npy_bytes = io.BytesIO(response.content)
                            original_mask = np.load(npy_bytes)
                            original_mask = apply_color_map(original_mask)

                            # Affichage du masque original pour comparaison
                            st.subheader("Masque original")
                            fig, ax = plt.subplots()
                            ax.imshow(original_mask)
                            ax.axis("off")
                            st.pyplot(fig)
                        else:
                            st.warning("Impossible de récupérer le masque original depuis Azure.")

                    else:
                        st.warning("URL du masque original non trouvée.")

                except Exception as e:
                    st.error(f"Erreur lors du chargement du fichier: {e}")
        else:
            st.error(f"Erreur lors de la prédiction: {response.status_code}")
