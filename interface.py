import streamlit as st
import requests
import numpy as np
import io
import PIL.Image as Image

API_URL = "http://127.0.0.1:8000"

st.title("Segmentation d'Images - Interface de Test")

# Récupérer la liste des images disponibles
st.sidebar.header("Sélection de l'image")
response = requests.get(f"{API_URL}/images/")
if response.status_code == 200:
    image_list = response.json().get("images", [])
    selected_image = st.sidebar.selectbox("Choisissez une image :", image_list)
else:
    st.sidebar.error("Impossible de récupérer la liste des images")
    selected_image = None

if selected_image:
    image_path = f"images/test/{selected_image}"
    with open(image_path, "rb") as img_file:
        image_data = img_file.read()

    st.image(image_path, caption="Image originale", use_column_width=True)

    # Lancer la prédiction
    if st.button("Lancer la Prédiction"):
        files = {"file": (selected_image, image_data, "image/jpeg")}
        response = requests.post(f"{API_URL}/predict/", files=files)
        if response.status_code == 200:
            mask_bytes = io.BytesIO(response.content)
            predicted_mask = np.load(mask_bytes)

            # Charger le masque réel
            mask_path = image_path.replace("images/test", "masks").replace(".jpg", "_gtFine_labelIds.png")
            real_mask = Image.open(mask_path)

            st.image(real_mask, caption="Masque Réel", use_column_width=True)
            st.image(predicted_mask, caption="Masque Prédit", use_column_width=True)
        else:
            st.error("Erreur lors de la prédiction")
