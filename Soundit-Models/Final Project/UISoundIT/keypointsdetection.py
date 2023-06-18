import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import threading
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GRU, Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.regularizers import L2
from tensorflow.keras.callbacks import TensorBoard
from keras import regularizers


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

model.load_weights('/Users/kevynkrancenblum/Desktop/Data Science/Final Project/Body_Language_recognition/modelsSaved/BodyModelv5.h5')

def mediapipe_detection(image, model):
    if image.size == 0:
        raise ValueError("Input image is empty")
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False                  # Image is no longer writeable
        results = model.process(image)                 # Make prediction
        return image, results
    except:
        raise RuntimeError("Failed to make a prediction with mediapipe model")
    
def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 
    
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose,face, lh, rh])
    
colors = [(245,117,16), (117,245,16), (16,117,245)]


import cv2

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        y_start = 120 + num * 40  # Adjusted Y-coordinate for rectangle start
        y_end = 150 + num * 40  # Adjusted Y-coordinate for rectangle end
        text_y = 135 + num * 40  # Adjusted Y-coordinate for text
        
        cv2.rectangle(output_frame, (0, y_start), (int(prob*100), y_end), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
    return output_frame




def InferenceVideo(results,test,image):
    keypoints = extract_keypoints(results)
    sequence.append(keypoints)
    print(sequence)
    sequence = sequence[-50:]
    if len(sequence) == 50:
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        prediction=actions[np.argmax(res)]
        print(actions[np.argmax(res)])                    
    #3. Viz logic
        if res[np.argmax(res)] > 0.6: 
            if len(sentence) > 0: 
                if actions[np.argmax(res)] != sentence[-1]:
                    sentence.append(actions[np.argmax(res)])
                    if len(sentence )> 15:
                        test.append(sentence)

            else:
                sentence.append(actions[np.argmax(res)])

        if len(sentence) > 5: 
            sentence = sentence[-5:]

        # Viz probabilities
        image = prob_viz(res, actions, image, colors)
        return results,test,image