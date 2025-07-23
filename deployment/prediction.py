#import library
import pandas as pd
import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow_hub.keras_layer import KerasLayer 
from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow as tf
from tensorflow.keras.models import load_model

#load model
def run():
    st.title("Weather Image Classification")
    st.markdown("""This model will help you to classify Weather from an Image""")  
    file = st.file_uploader("Upload an image", type=["jpg", "png"])

    model = load_model('src/weather_classifier.keras', custom_objects={'KerasLayer': KerasLayer})
    target_size=(224, 224)

    def import_and_predict(image_data, model): # Target size dilakukan resize sesuai dengan kebutuhan Transfer Learning 
        # Load image and resize to target size
        image = load_img(image_data, target_size=target_size)
        # Convert image to array
        img_array = img_to_array(image)
        # Preprocess image for EfficientNet
        img_array = preprocess_input(img_array)
        # Expand dims to create batch of 1
        img_array = np.expand_dims(img_array, axis=0)

         # Predict
        preds = model.predict(img_array)
        predicted_class = np.argmax(preds, axis=-1)[0]

        class_labels = ['Cloudy', 'Rain', 'Shine', 'Sunrise']
        result = f"Prediction: {class_labels[predicted_class]}"

        return result

    if file is None:
        st.text("Please upload an image file")
    else:
        result = import_and_predict(file, model)
        st.image(file)
        st.write(result)
        
if __name__ == "__main__":
    run()