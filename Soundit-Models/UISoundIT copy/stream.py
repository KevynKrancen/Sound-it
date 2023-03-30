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

device_name = tf.test.gpu_device_name()
import streamlit as st
from facial import facial

from colorsentiment import ColorRecognition

st.write(""" 
# Sound-IT """)



mode = st.sidebar.radio("Select Mode", ("Upload File", "Use Camera"))

if mode == "Upload File":
    uploaded_file = st.file_uploader("Choose an video...") 
    if uploaded_file is not None:
        path="/Users/kevynkrancenblum/Desktop/Data Science/Final Project/VideoBody/ModelVideos/"+uploaded_file.name
        video_file = open("/Users/kevynkrancenblum/Desktop/Data Science/Final Project/VideoBody/ModelVideos/"+uploaded_file.name, 'rb')
        Facial_recognition,Frame_Window=(facial(path))
        color=ColorRecognition(path)
        st.write("Body : ", Facial_recognition)
        st.write("Color: " , color)
else:
    uploaded_file = 2
    video_file = open(uploaded_file)
    Facial_recognition,Frame_Window=(facial(uploaded_file))
    color=ColorRecognition(uploaded_file)
    st.write("Body : ", Facial_recognition)
    st.write("Color: " , color)




