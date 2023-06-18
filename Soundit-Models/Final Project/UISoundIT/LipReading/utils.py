import tensorflow as tf
from typing import List
import cv2
import os 
import mediapipe as mp  

mp_face_detection = mp.solutions.face_detection

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
# Mapping integers back to original characters
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)


import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

def load_video(path:str) -> List[float]: 
    cap = cv2.VideoCapture(path)
    frames = []
    with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
        for _ in range(75): 
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)

            # Draw the face mesh annotations on the image.
            image.flags.writeable = True
            image_annotated = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Extract the lower part of the face
                    h, w, _ = image.shape
                    y1 = int(face_landmarks.landmark[11].y * h) - 100
                    y2 = int(face_landmarks.landmark[152].y * h) + 50
                    x1 = int(face_landmarks.landmark[234].x * w) - 20
                    x2 = int(face_landmarks.landmark[454].x * w) + 20

                    lower_face = image[y1:y2, x1:x2]
                    lower_face_resized = cv2.resize(lower_face, (140, 46))
                    #lower_face_resized = cv2.cvtColor(lower_face_resized, cv2.COLOR_BGR2GRAY)
                    lower_face = tf.image.rgb_to_grayscale(lower_face_resized)
                    frames.append(lower_face)

        cap.release()

    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std
    
def load_alignments(path:str) -> List[str]: 
    #print(path)
    with open(path, 'r') as f: 
        lines = f.readlines() 
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil': 
            tokens = [*tokens,' ',line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]

def load_data(path: str): 
    path = bytes.decode(path.numpy())
    file_name = path.split('/')[-1].split('.')[0]

    video_path = os.path.join('/Users/kevynkrancenblum/Desktop/Data Science/Final Project/LipReading/data/s1',f'{file_name}.mpg')
    alignment_path = os.path.join('/Users/kevynkrancenblum/Desktop/Data Science/Final Project/LipReading/data/alignments/s1',f'{file_name}.align')
    frames = load_video(video_path) 
    alignments = load_alignments(alignment_path)
    
    return frames, alignments