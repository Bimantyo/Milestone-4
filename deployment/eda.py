import streamlit as st 
import pandas as pd
from PIL import Image

def run():
    # Membuat title
    st.title('Weather Image Classification with CNN Model')

    # Membuat Sub Header
    st.subheader('This page will contain Exploratory Data Analysis from Multi-class Weather Dataset by Bimantyo Arya')

    # Menambahkan Gambar
    image = Image.open('./src/image.webp')
    st.image(image, caption='Weather Classifier.')

    # Penyampaian Background
    st.markdown("""
### Background

Weather classification using images is a challenging computer vision task that involves recognizing visual patterns from environmental features. This project uses a CNN-based model trained on four classes: **Cloudy**, **Rain**, **Shine**, and **Sunrise**.  

Rapid weather changes impact transportation, agriculture, and daily planning. Accurate weather classification helps in better forecasting and public preparedness.

The dataset used consists of total 1,125 images sourced from Kaggle.
                
The idea of this weather classification project is not to replace weather prediction that has advanced technology with powerful sensor, but to fill the gap for those who live in remote areas and have minimal technological infrastructure.
""")

    # Section: Sample Images from Each Class
    st.markdown('### Sample Images from Each Weather Class')

    # Load sample gambar dari setiap kelas 
    st.markdown('### Class Cloudy')
    cloudy_img = Image.open('./src/image_cloudy.jpg')
    st.image(cloudy_img, caption='For Cloudy in this dataset have 300 images')
    st.markdown('### Class Rain')     
    rain_img = Image.open('./src/image_rain.jpeg')
    st.image(rain_img, caption='For Rain in this dataset have 215 images')
    st.markdown('### Class Shine')  
    shine_img = Image.open('./src/image_shine.jpg')
    st.image(shine_img, caption='For Shine in this dataset have 253 images')
    st.markdown('### Class Sunrise')  
    sunrise_img = Image.open('./src/image_sunrise.jpg')
    st.image(sunrise_img, caption='For Sunrise in this dataset have 315 images')  

    # Section: Sample Images from Each Class
    st.markdown('### Images Distribution from Each Weather Class')
    img_dist = Image.open('./src/Distribusi_kelas.png')
    st.image(img_dist)
    st.markdown("""Based on the barplot above, it can be seen that the dataset is considered as imbalanced, although not too extreme with the distribution of each class

- Total Data - Cloudy: 300 or in percentage 26.7%
- Total Data - Rain: 215 or in percentage 19.1%
- Total Data - Sunrise: 357 or in percentage 31.7%
- Total Data - Shine: 253 or in percentage 22.5%

- Total All Data: 1125 equals 100%

A dataset that is imbalanced will affect model performance because the model tends to learn more from the majority class, and this will make the model tend to predict the class that has the most, because the number of classes with minimal data will tend to be more difficult for the model to predict.
""")

    # Section: Sample Images from Each Class
    st.markdown('### RGB Color Histogram from Each Weather Class')

    image_paths = {
    'Cloudy': './src/hist_cloudy.png',
    'Rain': './src/hist_rain.png',
    'Shine': './src/hist_shine.png',
    'Sunrise': './src/hist_sunrise.png'
}
    # Create 4 columns (1 row)
    cols = st.columns(4)

    # Loop over each image and column
    for i, (label, path) in enumerate(image_paths.items()):
        with cols[i]:
            img = Image.open(path)
            st.image(img, caption=label, use_container_width=True)

    st.markdown("""
#### Cloudy
- The *Cloudy* class has the most widespread color distribution between 50-200 where the majority of colors in the red, green and blue spectrum have almost the same range. This states that the Cloudy class is dominated by dark gray to bright colors. There may be many images that have cloudy conditions but there are still slightly bright conditions.

- There is a spike at pixels 0 and 255 which can indicate that the *Cloudy* condition is not completely dark - because the nature of clouds spreads light widely, it can still produce very bright areas.

#### Rain
- The *Rain* class has its most color distribution in the middle (values ​​75–160) where the majority of colors in the rain images are at the gray, light gray, and medium color levels.

- This is quite reasonable because when it rains, images tend to have cloudy lighting, little contrast, and are neither too bright nor too dark.

- There is a spike in pixel value 255, this indicates that there are bright white areas in many images, where these white areas could come from water reflections, raindrops or bright lights.

#### Shine 
- The *Shine* class has a color distribution that tends to be low and not too volatile, where at pixel 0 there is a very high red spectrum. This indicates that many images that have dark areas in the red spectrum could be a shadows/backlights that often occur when taking pictures while facing the sun.

- There is a spike at pixel 255 in the blue spectrum, indicating that there is very bright light, and the green spectrum has a stable frequency indicating a neutral or natural tone, such as leaves or grass during the day.               

#### Sunrise
- The *Sunrise* class has a color distribution that tends to be flat but shows spikes at pixels 0 and 255 in the red and blue spectrum. The spikes in the blue spectrum indicate that many blue areas are very dark, this can happen because of shadows or early morning skies often filled with a dark blue color. While the spikes in the red spectrum indicate that many areas are very bright with red nuances typical of sunrise. 

- The green spectrum looks neutral and spreads not too dominantly, tends to be a color balancer.                

""")

if __name__ == '__main__':
    run()