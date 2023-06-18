# to run streamlit run stream.py 
import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp
import time
import glob
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from colorsentiment import ColorRecognition
import streamlit as st
from inference import Sentiment_recognition
from inferenceCam import Sentiment_recognition_cam
from allinference import Sentiment_recognition as Strall
import os 
import imageio 
import LipReading
from LipReading.utils import load_data, num_to_char
from LipReading.modelutil import load_model
from Muse import randomArtist


def get_dominant_sentiment(*sentiments):
    sentiment_pairs = [(sentiments[i], sentiments[i + 1]) for i in range(0, len(sentiments), 2)]
    max_percentage = 0
    dominant_sentiment = ""

    for sentiment, percentage in sentiment_pairs:
        if percentage > max_percentage:
            max_percentage = percentage
            dominant_sentiment = sentiment

    return dominant_sentiment

device_name = tf.test.gpu_device_name()

def most_frequent(List):
    counter = 0
    num = List[0]
     
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
 
    return num

st.markdown("""
    <style type="text/css">
    body {
        background-color: #2b2b2b;
        color: white;
    }
    [data-testid=stSidebar] {
        background-color: rgb(129, 164, 182);
        color: #FFFFFF;
    }
    [aria-selected="true"] {
         color: #000000;
    }
    </style>
""", unsafe_allow_html=True)

st.write(""" 
# Sound-IT """)
         
with st.sidebar: 
    st.image('/Users/kevynkrancenblum/Desktop/Data Science/Final Project/UISoundIT/eyes.png')
    st.title('SoundIT')


mode = st.radio("Select Mode", ("Upload File", "Use Camera","Lip Reading","All"))

if mode == "Upload File":
    uploaded_file = st.file_uploader("Choose a video...") 
    if uploaded_file is not None:
        video_bytes = uploaded_file.read()
        filename = uploaded_file.name
        with open(filename, 'wb') as f:
            f.write(video_bytes)
        path = filename
        col1, col2 = st.columns(2)
        with col1:
            st.write("Original Video")
            st.video(video_bytes)
        with col2:
            faceSentiments,facePourcentage, bodySentiments,Bodypourcentage,ColorEmotion,ColorPourcentage = Sentiment_recognition(path)
            st.write("Face: ", faceSentiments,round(facePourcentage,2))
            st.write("Body: ", bodySentiments,round(Bodypourcentage,2))
            st.write("Color: ", ColorEmotion,round(ColorPourcentage,2))
            result = (faceSentiments,facePourcentage, bodySentiments,Bodypourcentage,ColorEmotion,ColorPourcentage)
            dominant_sentiment = get_dominant_sentiment(*result)
            st.write("Dominance: ", dominant_sentiment)
            Artist=randomArtist(dominant_sentiment)
            st.write("Music to generate: ", Artist)
elif mode == "Use Camera":
    Facial_recognition,Frame_Window=(Sentiment_recognition_cam(0))

    col1, col2 = st.beta_columns(2)
    with col1:
        st.write("Body : ", Facial_recognition)
        #st.write("Color: " , color)

    with col2:
        st.video(1)
if mode == "All":
    for file_name in os.listdir("/Users/kevynkrancenblum/Desktop/Data Science/Final Project/VideoBody/allvideos/"):
        video_path = os.path.join("/Users/kevynkrancenblum/Desktop/Data Science/Final Project/VideoBody/allvideos/", file_name)
        print(video_path)
        if video_path == '/Users/kevynkrancenblum/Desktop/Data Science/Final Project/VideoBody/allvideos/.DS_Store':
            pass
        else:
            with open(video_path, 'rb') as f:
                video_bytes = f.read()
            col1, col2 = st.columns(2)
            with col1:
                st.write("Original Video")
                st.video(video_bytes)
            with col2:
                faceSentiments,facePourcentage, bodySentiments,Bodypourcentage,ColorEmotion,ColorPourcentage = Strall(video_path)
                result = (faceSentiments,facePourcentage, bodySentiments,Bodypourcentage,ColorEmotion,ColorPourcentage)
                dominant_sentiment = get_dominant_sentiment(*result)
                st.write("Face: ", faceSentiments,round(facePourcentage,2))
                st.write("Body: ", bodySentiments,round(Bodypourcentage,2))
                st.write("Color: ", ColorEmotion,round(ColorPourcentage,2))
                st.write("Dominance: ", dominant_sentiment)


    # Setup the sidebar
elif mode == "Lip Reading":
    st.title('LipNet For Sound-IT') 
    # Generating a list of options or videos 
    options = os.listdir(os.path.join('/Users/kevynkrancenblum/Desktop/Data Science/Final Project/LipReading', 'data', 's1'))
    selected_video = st.selectbox('Choose video', options)

    # Generate two columns 
    col1, col2 = st.columns(2)

    if options: 

        # Rendering the video 
        with col1: 
            st.info('The video below displays the converted video in mp4 format')
            file_path = os.path.join('/Users/kevynkrancenblum/Desktop/Data Science/Final Project/LipReading','data','s1', selected_video)
            video = open('/Users/kevynkrancenblum/Desktop/Data Science/Final Project/LipReading/data/s1/'+selected_video, 'rb') 
            video_bytes = video.read() 
            st.video(video_bytes, format='video/mpg')
            print("Here")


        with col2: 
            st.info('This is all the machine learning model sees when making a prediction')
            video, annotations = load_data(tf.convert_to_tensor(file_path))
            imageio.mimsave('animation.gif', video, fps=10)
            st.image('animation.gif', width=400) 

            st.info('This is the output of the machine learning model as tokens')
            model = load_model()
            yhat = model.predict(tf.expand_dims(video, axis=0))
            decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
            st.text(decoder)

            # Convert prediction to text
            st.info('Decode the raw tokens into words')
            converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
            st.text(converted_prediction)
            

