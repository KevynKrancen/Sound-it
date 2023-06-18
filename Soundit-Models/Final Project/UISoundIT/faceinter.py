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
import sys
from keypointsdetection import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GRU, Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.regularizers import L2
from tensorflow.keras.callbacks import TensorBoard
from keras import regularizers
from colorsentiment import ColorRecognition
import dlib 

dlib_detector = dlib.get_frontal_face_detector()

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


actions = np.array(['Angry','Happy','Sad'])
label_map = {label:num for num, label in enumerate(actions)}
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

input_shape = (30, 1662)

model = Sequential()

# CNN layers
model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Conv1D(128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Conv1D(256, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# GRU layers
model.add(GRU(256, return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(GRU(128, return_sequences=False))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(64, activation='relu', activity_regularizer=L2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu', activity_regularizer=L2(0.01)))

model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.load_weights('/Users/kevynkrancenblum/Desktop/Data Science/Final Project/Body_Language_recognition/BodyModel.h5')



from collections import Counter
def most_frequent(List):
    counter = 0
    num = List[0]
     
    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
 
    return num

def facial(video):
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


    model_1 = VGGNet(input_shape, num_classes, weights_1)
    model_1.load_weights(model_1.checkpoint_path)

    model_2 = VGGNet(input_shape, num_classes, weights_2)
    model_2.load_weights(model_2.checkpoint_path)

    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.95)


    def detection_preprocessing(image, h_max=360):
        h, w, _ = image.shape
        if h > h_max:
            ratio = h_max / h
            w_ = int(w * ratio)
            image = cv2.resize(image, (w_, h_max))
        return image

    def resize_face(face):
        x = tf.expand_dims(tf.convert_to_tensor(face), axis=2)
        return tf.image.resize(x, (48, 48))

    def recognition_preprocessing(faces):
        x = tf.convert_to_tensor([resize_face(f) for f in faces])
        return x

    bodySentiments = []
    faceSentiments = []
    resultat = []

    def inference(image):
        image=detection_preprocessing(image, target_h)
        emotion=''
        H, W, _ = image.shape
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #results = face_detection.process(rgb_image)
        detections = dlib_detector(rgb_image)

        if results.detections:
            # Only process the first face detected
            detection = results.detections[0]
            box = detection.location_data.relative_bounding_box
            mp_drawing.draw_detection(image, detection)

            x = int(box.xmin * W)
            y = int(box.ymin * H)
            w = int(box.width * W)
            h = int(box.height * H)

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(x + w, W)
            y2 = min(y + h, H)

            face = image[y1:y2, x1:x2]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            x = recognition_preprocessing([face])

            y_1 = model_1.predict(x)
            y_2 = model_2.predict(x)
            l = np.argmax(y_1 + y_2, axis=1)

            cv2.rectangle(image, (x1, y1), (x2, y2), emotions[l[0]][1], 2, lineType=cv2.LINE_AA)
            cv2.rectangle(image, (x1, y1 - 20), (x2 + 20, y1), emotions[l[0]][1], -1, lineType=cv2.LINE_AA)

            cv2.putText(image, f'{emotions[l[0]][0]}', (x1, y1 - 5),
                        0, 0.6, emotions[l[0]][2], 2, lineType=cv2.LINE_AA)

            resultat.append(emotions[l[0]][0])
            emotion = emotions[l[0]][0]
            
        return image  , emotion



    sentence=[]
    sequence = []
    test= []
    threshold = 0.8
    cap = cv2.VideoCapture(video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_h = frame_height
    target_w = int(target_h * frame_width / frame_height)
    output_file = 'test2.MOV'
    fourcc = cv2.VideoWriter_fourcc(*'H264')# Set the codec for the .mov format on Macs
    out = cv2.VideoWriter(output_file, fourcc, fps, (target_w, target_h))
    st.write("TESTING ON REALTIME")
    Frame_Window = st.image([])
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            success, image = cap.read()
            if success:
                image,faceemmotion = inference(image)
                image, results = mediapipe_detection(image, holistic)
                draw_styled_landmarks(image, results)
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]
                if len(sequence) == 30:
                    if faceemmotion=='Happy':
                        res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    else :
                        res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    if res[np.argmax(res)] > threshold: 
                        bodySentiments.append(sentence)        
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                                if len(sentence )> 15:
                                    test.append(sentence)
                                    cv2.rectangle(image, (0, 0), (image.shape[1], 80), (0, 0, 0), -1)
                                    cv2.putText(image, f"res: {res}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                                    cv2.putText(image, f"sentence: {sentence}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                                    # Define the rectangle coordinates
                                    
                        else:
                            sentence.append(actions[np.argmax(res)])

                    if len(sentence) > 5: 
                        sentence = sentence[-5:]

                    # Viz probabilities
                    image = prob_viz(res, actions, image, colors)
                    rect_start = (0, 0)
                    rect_end = (200, 60)

                    # Draw the rectangle on the image
                    cv2.rectangle(image, rect_start, rect_end, (0, 0, 0), -1)
                    # Add text to the rectangle
                    cv2.putText(image, 'Happy', (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(res[1], 2)), (1, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                    cv2.putText(image, 'Angry', (80, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(res[0], 2)), (80, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                    cv2.putText(image, 'Sad', (140, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(res[2], 2)), (140, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


                #image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                out.write(image)
                Frame_Window.image(image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
            
        cap.release()    

    print(bodySentiments)
    most_frequent_face_sentiment = most_frequent(faceSentiments)
    most_frequent_body_sentiment = most_frequent(bodySentiments)

    return most_frequent_face_sentiment, most_frequent_body_sentiment

                   


