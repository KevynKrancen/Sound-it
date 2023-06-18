import streamlit as st
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import time

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def creatingFolder(actions,no_sequences,DATA_PATH):
    for action in actions: 
        for sequence in range(no_sequences):
            try: 
                os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
            except:
                pass

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):

    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             )   
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


def button1_func():
    st.write("Jumping Slow")
    DATA_PATH='/Users/kevynkrancenblum/Desktop/Data Science/Final Project/Body_Language_recognition/KeypointsData/Happy2/'
    st.write("Recording in progress! ")
    actions = np.array(["Happy2"])
    no_sequences = 30
    sequence_length = 30
    start_folder = 0
    Frame_Window= st.image([])
    cap = cv2.VideoCapture(0)
    creatingFolder(actions,no_sequences,DATA_PATH)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            for sequence in range(no_sequences):
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    image, results = mediapipe_detection(frame, holistic)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    black_screen = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
                    draw_styled_landmarks(black_screen, results)
                    if frame_num == 0: 
                        cv2.putText(black_screen, 'STARTING COLLECTION', (120,200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(black_screen, '{} Video Number {}'.format(action, sequence), (0,200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 10, cv2.LINE_AA)

                        Frame_Window.image(black_screen)
                        cv2.waitKey(2000)
                    else: 
                        cv2.putText(black_screen, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 4, cv2.LINE_AA)

                    Frame_Window.image(black_screen)
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break            
    cap.release()
    cv2.destroyAllWindows()

def button2_func():
    st.write("Jumping Slow")
    DATA_PATH='/Users/kevynkrancenblum/Desktop/Data Science/Final Project/Body_Language_recognition/KeypointsDataVideo/Happy/'
    st.write("Recording in progress! ")
    actions = np.array(["Happyfull"])
    no_sequences = 500
    sequence_length = 30
    start_folder = 0
    Frame_Window= st.image([])
    cap = cv2.VideoCapture(1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    creatingFolder(actions,no_sequences,DATA_PATH)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            for sequence in range(no_sequences):
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    image, results = mediapipe_detection(frame, holistic)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    black_screen = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
                    draw_styled_landmarks(black_screen, results)
                    if frame_num == 0: 
                        cv2.putText(black_screen, 'STARTING COLLECTION', (120,200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(black_screen, '{} Video Number {}'.format(action, sequence), (0,200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 10, cv2.LINE_AA)

                        Frame_Window.image(black_screen)
                        cv2.waitKey(2000)
                    else: 
                        cv2.putText(black_screen, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 4, cv2.LINE_AA)

                    Frame_Window.image(black_screen)
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break            
    cap.release()
    cv2.destroyAllWindows()



def button2_func():
    st.write("Angry")
    DATA_PATH='/Users/kevynkrancenblum/Desktop/Data Science/Final Project/Body_Language_recognition/KeypointsDataVideo/Angry/'
    st.write("Recording in progress! ")
    actions = np.array(["Angryfull1"])
    no_sequences = 500
    sequence_length = 30
    start_folder = 0
    Frame_Window= st.image([])
    cap = cv2.VideoCapture('/Users/kevynkrancenblum/Desktop/Data Science/Final Project/angryall.mov')
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    creatingFolder(actions,no_sequences,DATA_PATH)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            for sequence in range(no_sequences):
                for frame_num in range(sequence_length):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    image, results = mediapipe_detection(frame, holistic)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    black_screen = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
                    draw_styled_landmarks(black_screen, results)
                    if frame_num == 0: 
                        cv2.putText(black_screen, 'STARTING COLLECTION', (120,200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(black_screen, '{} Video Number {}'.format(action, sequence), (0,200), 
                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 10, cv2.LINE_AA)

                        Frame_Window.image(black_screen)
                        cv2.waitKey(2000)
                    else: 
                        cv2.putText(black_screen, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 4, cv2.LINE_AA)

                    Frame_Window.image(black_screen)
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break            
    cap.release()
    cv2.destroyAllWindows()

def app():
    st.set_page_config(page_title="Button Example", page_icon=":guardsman:", layout="wide")
    st.write("Welcome Please select the type of data collection you want collect")
    st.sidebar.title("Record Data")
    btn1 = st.sidebar.button("Start Recording", key='1')
    btn2 = st.sidebar.button("Start Recording", key='2')
    if btn1:
        button1_func()
    if btn2:
        button2_func()
if __name__ == "__main__":
    app()