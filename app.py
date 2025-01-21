import streamlit as st
import tensorflow as tf
from logging import PlaceHolder
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout,BatchNormalization,Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import streamlit  # 👈 Add the caching decorator
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np

# Charger le modèle
model = tf.keras.models.load_model("C:/Users/bmd tech/projet_master/rf_model.pkl")

import streamlit as st

# # Utilisez file_uploader pour permettre aux utilisateurs de télécharger une image
# uploaded_file = st.file_uploader("Uploader une image d'échographie abdominale", type=["jpg", "jpeg", "png"])

# # Vérifiez si une image a été téléchargée
# if uploaded_file is not None:
#     # Affichez l'image téléchargée
#     st.image(uploaded_file, caption="Image d'échographie abdominale", use_column_width=True)

# # config 
# # Bouton de prediction
# result = st.radio(["DIAGNOSTIC"])

# def loade_model(path):
#     model=load_model(path)
#     return model 

# # Utilisation du modèle pour faire des prédictions 
# def faire_prediction(uploaded_file):

#     prediction = model.predict(uploaded_file)

#     st.write(prediction)



import streamlit as st
from PIL import Image  # Utilisez la bibliothèque Python Imaging Library (PIL) pour manipuler les images

# Fonction de prédiction
def predict(images):
    # Ajoutez ici la logique de prédiction avec votre modèle
    model = load_model('modele_echog.h5')
# def loade_model(path):
#         model=load_model(path)
#         return model

def main():
    st.title("Application de prédiction d'images")

    # Widget pour télécharger une image
    uploaded_file = st.file_uploader("Uploader une image d'échographie abdominale", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Affichez l'image téléchargée
        image = Image.open(uploaded_file)
        st.image(image, caption="Image d'échographie abdominale", use_column_width=True)

        # Bouton de prédiction
        if st.button("PREDICTION"):
            # Effectuez la prédiction en utilisant votre fonction predict
            prediction_result = predict(image)

            # Affichez le résultat de la prédiction
            st.success(f"Résultat de la prédiction : {prediction_result}")

if __name__ == "__main__":
    main()



from util import config
import streamlit as st

# Définir la taille de la police souhaitée
font_size = "30px"

# Définir la police de caractères souhaitée
font_family = "Times New Roman, serif"


# Définir la couleur du texte souhaitée (code hexadécimal)
text_color = "white"

# Custom style settings
button_style = f"font-size: 50px; color: white; background-color: #2e6b8e; border-radius: 8px; padding: 10px 20px;"

result_text_style = f"font-size: 30px; color: white; font-family: 'Times New Roman', serif;"

# def loade_model(path):
#         model=load_model(path)
#         return model 

       
#         st.write(
#                 f'<div style="font-size: {font_size}; color: {text_color}; font-family: {font_family};">'
#                 f"Sentiment: {sentiment} (Probability: {probability:.3f})"
#                 '</div>',
#                 unsafe_allow_html=True
#             )   
 
            