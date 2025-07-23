import streamlit as st
import eda
import prediction

# Membuat page lebih lebar 
st.set_page_config(
    page_title='Weather Image Classification', # Mengubah namanya ketika dihover ke web bukan jadi localhost lagi
    layout='wide',
    initial_sidebar_state='expanded'
)


page = st.sidebar.selectbox('Pilih Halaman', ('EDA', 'Prediction'))

if page == 'EDA':
    eda.run()

else:
    prediction.run()