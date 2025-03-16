import streamlit as st
import requests
import numpy as np
import io
from PIL import Image
import matplotlib.pyplot as plt

API_URL = "http://127.0.0.1:8000/predict/"

st.title("Segmentation d'Images - Interface de Test")

# Sélection de l'image locale
uploaded_file = st.file_uploader("Choisissez une image", type=["png", "jpg", "jpeg"])

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

            # Affichage du masque
            st.subheader("Masque Prédit")
            fig, ax = plt.subplots()
            ax.imshow(predicted_mask, cmap="jet")  # Appliquer une colormap pour visualiser les classes
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.error(f"Erreur lors de la prédiction: {response.status_code}")
