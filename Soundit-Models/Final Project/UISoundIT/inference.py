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
import streamlit as st
import sys
from keypointsdetection import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GRU, Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.regularizers import L2
from tensorflow.keras.callbacks import TensorBoard
from keras import regularizers
import dlib 
from collections import Counter

device_name = tf.test.gpu_device_name()
dlib_detector = dlib.get_frontal_face_detector()

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


def ColorRecognition(frame):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the color ranges for each emotion
    yellow_range = [(20, 100, 100), (40, 255, 255)]
    green_range = [(40, 50, 50), (80, 255, 255)]
    purple_range = [(120, 50, 50), (160, 255, 255)]
    red_range1 = [(0, 100, 100), (10, 255, 255)]
    red_range2 = [(170, 100, 100), (180, 255, 255)]
    blue_range = [(100, 50, 50), (140, 255, 255)]

    # Threshold the image to detect pixels within each color range
    yellow_mask = cv2.inRange(hsv, np.array(yellow_range[0]), np.array(yellow_range[1]))
    green_mask = cv2.inRange(hsv, np.array(green_range[0]), np.array(green_range[1]))
    purple_mask = cv2.inRange(hsv, np.array(purple_range[0]), np.array(purple_range[1]))
    red_mask1 = cv2.inRange(hsv, np.array(red_range1[0]), np.array(red_range1[1]))
    red_mask2 = cv2.inRange(hsv, np.array(red_range2[0]), np.array(red_range2[1]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    blue_mask = cv2.inRange(hsv, np.array(blue_range[0]), np.array(blue_range[1]))

    # Count the number of pixels in each color range
    yellow_count = cv2.countNonZero(yellow_mask)
    green_count = cv2.countNonZero(green_mask)
    purple_count = cv2.countNonZero(purple_mask)
    red_count = cv2.countNonZero(red_mask)
    blue_count = cv2.countNonZero(blue_mask)

    # Determine the dominant color and assign an emotion
    max_count = max(yellow_count, green_count, purple_count, red_count, blue_count)
    if max_count == yellow_count:
        emotion = 'Happy'
    elif max_count == green_count:
        emotion = 'Happy'
    elif max_count == purple_count:
        emotion = 'Happy'
    elif max_count == red_count:
        emotion = 'Angry'
    elif max_count == blue_count:
        emotion = 'Sad'
    else:
        emotion = 'Neutral'

    return emotion




def most_frequent(List):
    counter = 0
    list_len = len(List)
    print(List) 
    for i in List:
        curr_frequency = List.count(i)
        curr_percentage = (curr_frequency / list_len) * 100
        if curr_frequency > counter:
            counter = curr_frequency
            num = i
            percentage = curr_percentage
    if list_len == 0 :
        num, percentage = 0
    return num, percentage

def detection_preprocessing(image, h_max=360):
        h, w, _ = image.shape
        if h > h_max:
            ratio = h_max / h
            w_ = int(w * ratio)
            image = cv2.resize(image, (w_, h_max))
        return image

def resize_face(face):
    # Check if the input face image is empty
    if face is None or face.size == 0:
        print("Error: Empty face image.")
        return None

    # Converts the face image to grayscale, adds a channel dimension, and resizes it to 48x48
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = np.expand_dims(face, axis=-1)  # Add the channel dimension
    face_tensor = tf.convert_to_tensor(face)  # Convert the NumPy array to a TensorFlow tensor
    return tf.image.resize(face_tensor, (48, 48))


def recognition_preprocessing(faces):
    # Processes a list of face images by resizing them and converting them to a tensor
    resized_faces = [resize_face(f) for f in faces]
    resized_faces = [f for f in resized_faces if f is not None]  # Remove any None elements
    if not resized_faces:
        print("Error: All input faces are empty.")
        return None
    x = tf.convert_to_tensor(resized_faces)
    return x

# model class for FER (face emotion recognition)
class VGGNet(Sequential):
    def __init__(self, input_shape, num_classes, checkpoint_path, lr=1e-3):
        super().__init__()
        self.add(Rescaling(1./255, input_shape=input_shape))
        self.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'))
        self.add(BatchNormalization())
        self.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D())
        self.add(Dropout(0.5))

        self.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D())
        self.add(Dropout(0.4))

        self.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D())
        self.add(Dropout(0.5))

        self.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same'))
        self.add(BatchNormalization())
        self.add(MaxPool2D())
        self.add(Dropout(0.4))

        self.add(Flatten())
        
        self.add(Dense(1024, activation='relu'))
        self.add(Dropout(0.5))
        self.add(Dense(256, activation='relu'))

        self.add(Dense(num_classes, activation='softmax'))

        self.compile(optimizer=Adam(learning_rate=lr),
                    loss=categorical_crossentropy,
                    metrics=['accuracy'])
        
        self.checkpoint_path = checkpoint_path

class BodySentimentModel(Sequential):
    def __init__(self, input_shape, num_classes):
        super().__init__()
        # CNN layers    
        self.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
        self.add(MaxPooling1D(pool_size=2))
        self.add(BatchNormalization())
        self.add(Dropout(0.5))

        self.add(Conv1D(128, kernel_size=3, activation='relu'))
        self.add(MaxPooling1D(pool_size=2))
        self.add(BatchNormalization())
        self.add(Dropout(0.5))

        self.add(Conv1D(256, kernel_size=3, activation='relu'))
        self.add(MaxPooling1D(pool_size=2))
        self.add(BatchNormalization())
        self.add(Dropout(0.5))

        # GRU layers
        self.add(GRU(256, return_sequences=True))
        self.add(BatchNormalization())
        self.add(Dropout(0.5))

        self.add(GRU(128, return_sequences=False))
        self.add(BatchNormalization())
        self.add(Dropout(0.5))

        self.add(Dense(64, activation='relu', activity_regularizer=L2(0.01)))
        self.add(Dropout(0.2))
        self.add(Dense(32, activation='relu', activity_regularizer=L2(0.01)))

        self.add(Dense(actions.shape[0], activation='softmax'))

        self.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


def Sentiment_recognition(video):
    FaceEm=['Angry','Happy','Sad']
    emotions = {
        0: ['Angry', (0,0,255), (255,255,255)],
        1: ['Disgust', (0,102,0), (255,255,255)],
        2: ['Fear', (255,255,153), (0,51,51)],
        3: ['Happy', (153,0,153), (255,255,255)],
        4: ['Sad', (255,0,0), (255,255,255)],
        5: ['Surprise', (0,255,0), (255,255,255)],
        6: ['Neutral', (160,160,160), (255,255,255)]
    }
    num_classes = len(emotions)
    input_shape = (48, 48, 1)
    weights_1 = '/Users/kevynkrancenblum/Desktop/Data Science/Final Project/Facial_emotion_recognition/saved_models/vggnet.h5'
    weights_2 = '/Users/kevynkrancenblum/Desktop/Data Science/Final Project/Facial_emotion_recognition/saved_models/vggnet_up.h5'

    model_1 = VGGNet(input_shape, num_classes, weights_1)
    model_1.load_weights(model_1.checkpoint_path)

    model_2 = VGGNet(input_shape, num_classes, weights_2)
    model_2.load_weights(model_2.checkpoint_path)

    actions = np.array(['Angry','Happy','Sad'])

    body_input_shape = (30, 1662)
    model_V1=BodySentimentModel(body_input_shape, actions.shape[0])
    model_V1.load_weights('/Users/kevynkrancenblum/Desktop/Data Science/Final Project/Body_Language_recognition/modelsSaved/BodyModel.h5')

    model_V2=BodySentimentModel(body_input_shape, actions.shape[0])
    model_V2.load_weights('/Users/kevynkrancenblum/Desktop/Data Science/Final Project/Body_Language_recognition/modelsSaved/BodyModelv6.h5')

    

    # Lists to store body and face sentiments
    bodySentiments = []
    faceSentiments = []
    coloremotions=[]
    resultat = []

    def inference(image):
        # Detects the face in the input image and predicts the emotion using the two models (model_1 and model_2) 
        # making detection of the face with dlib 
        #image = detection_preprocessing(image, target_h)
        emotion = ''
        H, W, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = dlib_detector(rgb_image)

        if detections:
            detection = detections[0]
            x1, y1, x2, y2 = detection.left(), detection.top(), detection.right(), detection.bottom()

            # Increase face bounding box by 1.5 times for better recognition
            width = x2 - x1
            height = y2 - y1
            
            x_center = x1 + width // 2
            y_center = y1 + height // 2
            
            new_width = int(width * 1)
            new_height = int(height * 1.3)
            
            x1_new = max(0, x_center - new_width // 2)
            x2_new = min(W, x_center + new_width // 2)
            y1_new = max(0, y_center - new_height // 2) - int(height * 0.05)  
            y2_new = min(H, y_center + new_height // 2) - int(height * 0.05)
            face = image[y1_new:y2_new, x1_new:x2_new]

            if face is not None:
                x = recognition_preprocessing([face])
                if x is not None:  
                    y_1 = model_1.predict(x)
                    y_2 = model_2.predict(x)
                    l = np.argmax(y_1 + y_2, axis=1)
                    # Draw emotion prediction on the image
                    cv2.rectangle(image, (x1_new, y1_new), (x2_new, y2_new), emotions[l[0]][1], 2, lineType=cv2.LINE_AA)
                    cv2.rectangle(image, (x1_new, y1_new - 20), (x2_new + 20, y1_new), emotions[l[0]][1], -1, lineType=cv2.LINE_AA)

                    cv2.putText(image, f'{emotions[l[0]][0]}', (x1_new, y1_new - 5),
                                0, 0.6, emotions[l[0]][2], 2, lineType=cv2.LINE_AA)

                    resultat.append(emotions[l[0]][0])
                    emotion = emotions[l[0]][0]
                    #print("lemotion reconue est" , emotion)
        return image, emotion

    num_happy_faces=0
    num_angry_faces=0
    flag=0
    sentence=[]
    sequence = []
    test= []
    threshold = 0.6
    modelType=''
    cap = cv2.VideoCapture(video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_h = frame_height
    target_w = int(target_h * frame_width / frame_height)
    output_file = 'test11.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'H264')# Set the codec for the .mov format on Macs
    out = cv2.VideoWriter(output_file, fourcc, fps, (target_w, target_h))
    st.write("TESTING ON REALTIME")
    Frame_Window = st.image([])
    frame_count = 0
    start_time = time.time()
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            success, image = cap.read()
            if success:
                coloremotion=ColorRecognition(image)
                print(coloremotion)
                coloremotions.append(coloremotion)
                image,faceemmotion = inference(image)
                if faceemmotion == FaceEm[1]:
                    num_happy_faces += 1
                if faceemmotion == FaceEm[0]:
                    num_angry_faces += 1
                image, results = mediapipe_detection(image, holistic)
                draw_styled_landmarks(image, results)
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]
                if len(sequence) == 30:
                    if num_happy_faces >= 8 :
                        modelType='V1'
                        num_angry_faces=0
                        res = model_V1.predict(np.expand_dims(sequence, axis=0))[0]
                        flag +=1
                        if flag > 10 : 
                            num_happy_faces = 0
                            flag= 0 
                    elif num_angry_faces >= 2:
                        num_happy_faces = 0
                        modelType='V2'
                        res = model_V2.predict(np.expand_dims(sequence, axis=0))[0]
                        flag +=1
                        if flag > 10 : 
                            num_angry_faces = 0 
                            flag = 0 
                    else:
                        modelType='V2'
                        res = model_V2.predict(np.expand_dims(sequence, axis=0))[0]
                    max_prob_action = actions[np.argmax(res)]
                    bodySentiments.append(max_prob_action)  # Add the highest predicted action to the list

                    # Viz probabilities
                    image = prob_viz(res, actions, image, colors)
                    rect_start = (0, 0)
                    rect_end = (int(image.shape[1]), int(image.shape[0] * 0.09))

                    # Draw the rectangle on the image
                    cv2.rectangle(image, rect_start, rect_end, (0, 0, 0), -1)
                    # Add labels on the same line
                    cv2.putText(image, 'Model:', (int(image.shape[1] * 0.05), int(image.shape[0] * 0.03)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(image, 'Happy:', (int(image.shape[1] * 0.2), int(image.shape[0] * 0.03)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(image, 'Angry:', (int(image.shape[1] * 0.35), int(image.shape[0] * 0.03)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(image, 'Sad:', (int(image.shape[1] * 0.5), int(image.shape[0] * 0.03)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(image, 'FER:', (int(image.shape[1] * 0.65), int(image.shape[0] * 0.03)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(image, 'CER:', (int(image.shape[1] * 0.80), int(image.shape[0] * 0.03)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)                              

                    cv2.putText(image, modelType, (int(image.shape[1] * 0.05), int(image.shape[0] * 0.08)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(res[1], 2)), (int(image.shape[1] * 0.2), int(image.shape[0] * 0.08)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(res[0], 2)), (int(image.shape[1] * 0.35), int(image.shape[0] * 0.08)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(res[2], 2)), (int(image.shape[1] * 0.5), int(image.shape[0] * 0.08)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(image, faceemmotion, (int(image.shape[1] * 0.65), int(image.shape[0] * 0.08)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(image, coloremotion, (int(image.shape[1] * 0.80), int(image.shape[0] * 0.08)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)




                frame_count += 1
                if frame_count % 10 == 0:
                    end_time = time.time()
                    fps = frame_count / (end_time - start_time)
                    fps_text = "FPS: {:.2f}".format(fps)
                    frame_count = 0
                    start_time = end_time
                try:
                    cv2.putText(image, fps_text, (10, image.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except :
                    pass
                Frame_Window.image(image)
                image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                out.write(image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
            
        cap.release()    

    most_frequent_face_sentiment,facePourcentage = most_frequent(resultat)
    most_frequent_body_sentiment,Bodypourcentage = most_frequent(bodySentiments)
    most_frequent_color_sentiment,coloremotionspourcentage = most_frequent(coloremotions)

    return most_frequent_face_sentiment,facePourcentage, most_frequent_body_sentiment,Bodypourcentage,most_frequent_color_sentiment,coloremotionspourcentage 

                   


