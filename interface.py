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
MASKS_DIR = r"C:\Users\flore\Openclassrooms\Projet 8\data\test"

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
    Récupère le masque original correspondant au nom de l'image.
    """
    base_name = image_name.split(".")[0].replace("_leftImg8bit", "")
    for root, dirs, files in os.walk(MASKS_DIR):
        for file in files:
            if base_name in file and file.endswith("_gtFine_labelIds.npy"):
                return os.path.join(root, file)
    return None



if uploaded_file is not None:
    # Afficher l'image originale
    image = Image.open(uploaded_file)
    st.image(image, caption="Image originale", use_container_width=True)

    # Bouton pour envoyer l'image à l'API
    if st.button("Lancer la prédiction"):
        files = {"file": uploaded_file.getvalue()}
        response = requests.post(API_URL, files=files)

        if response.status_code == 200:
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

            # Récupérer le masque original
            original_mask_path = get_original_mask(uploaded_file.name)
            if original_mask_path:
                original_mask = np.load(original_mask_path)
                original_mask = apply_color_map(original_mask)

                # Affichage du masque original pour comparaison
                st.subheader("Masque original")
                fig, ax = plt.subplots()
                ax.imshow(original_mask)
                ax.axis("off")
                st.pyplot(fig)
            else:
                st.warning("Masque original non trouvé.")

        else:
            st.error(f"Erreur lors de la prédiction: {response.status_code}")
