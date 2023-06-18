<h1 align="center" style="font-size: 48px;">Sound-IT</h1>
Sound-IT is a research project aimed at using artificial intelligence to recognize emotions in videos and generate an music that fits those emotions. We have created four models to determine the emotional tone of a video: facial expression detection, background hue recognition, body emotion recognition, and lip reading the models are running in parralele so may works slow if you only work on a cpu. After that, music is generated that matches the detected emotion using JukeBox .

The proposed method offers a potential solution to the problem of high costs associated with composing and recording original music for films.

<h1 align="center" style="font-size: 48px;">How it works</h1>
Our model is capable of recognizing a scene's dominant emotion by analyzing factors such as body language, facial expressions, lip reading, and background color. Using this emotion, the model generates a new piece of music ideally suited to the scene.
<div align="center">
  <br><br>
  <h1 align="center" style="font-size: 48px;">Sound-it PIPELINE</h1>
  <img width="876" alt="image" src="https://github.com/KevynKrancen/Sound-it/assets/102467763/480cc6a5-a2d2-4805-9d8e-6eaade0f8f30">

  <br><br>
 <h1 align="center" style="font-size: 48px;">Sound-it NN Achitectures & datasets</h1>
<img align="center" width="300" height="415" alt="Screenshot 2023-03-30 at 11 32 57" src="https://user-images.githubusercontent.com/102467763/228778334-103d1250-7fdc-4f0b-be9a-6146af27f999.png"> <img width="300" align="center" height="415" alt="Screenshot 2023-03-30 at 11 33 13" src="https://user-images.githubusercontent.com/102467763/228778357-ea8cc5dc-8087-42be-bd3b-9f78d8307986.png">
</div>
<br><br>
 <p >First, We detect what emotion the video is trying to portray by using 4 models:
<p>facial expression detection - Initially, we created
and trained a face recognition model based on the
VGG-16 architecture, then we used MediaPipe, an
open-source framework for building cross-platform
machine learning pipelines for perception tasks
such as object detection, tracking, and facial recog-
nition (Lugaresi et al., 2019), to locate the face in
given videos, and then we used the aforementioned
model to detect the emotion from the face.
background hue recognition - We extract the
color values of each pixel in the video, calculate
the average color of the video by averaging the
color values of the pixels, and then assign the color
to a specific emotion according to 4.</p>
<div>
<img align="center"width="300" height="200" alt="Screenshot 2023-03-30 at 11 32 57" src="https://user-images.githubusercontent.com/102467763/228815290-950b8dd9-a1d6-49a4-b897-7576f7c332a9.png"><img align="center"width="300" height="200" alt="Screenshot 2023-03-30 at 11 32 57" src="https://user-images.githubusercontent.com/102467763/228815392-a4deee54-312f-4798-b7b3-8782ece64353.png"> 
</div>


<h1 align="center" style="font-size: 48px;">Approch for Facial Emotion Recognition</h1>
<p align="center" >Using FER2013 as dataset and implementing VGG16 Neural Network Achitechture</p>
<div align="center">
<br>
<img align="center" width="300" height="200" alt="Screenshot 2023-03-30 at 11 32 57" src="https://user-images.githubusercontent.com/102467763/228867413-becccc6d-1a3e-43df-abba-6b6b83f54d01.png"><img align="center" width="300" height="200" alt="Screenshot 2023-03-30 at 11 32 57" src="https://user-images.githubusercontent.com/102467763/228867789-a14e6978-bd91-4931-86bd-9e233f22d217.png">
</div>





<h1 align="center" style="font-size: 48px;">Approch for BodyLanguage</h1>
<p align="center" >Collecting Keypoints with Mediapipe Holistic Model than Training The model With 30 frames per action</p>
<div align="center">
<br>
<img align="center" width="400" height="300" alt="Screenshot 2023-03-30 at 11 32 57" src="https://user-images.githubusercontent.com/102467763/228850744-968608cd-6b4b-4e61-a46f-b4b6d40906e4.png"><img align="center" width="200" height="300" alt="Screenshot 2023-03-30 at 11 32 57" src="https://user-images.githubusercontent.com/102467763/228852091-802a6e10-7eb4-4ac5-966f-d3d1a3bb2d66.png">
</div>
<br><br>
<h1 align="center" style="font-size: 48px;">Approch for LipReading</h1>
<p align="center" >Collecting the frame of the down face than Training The model With 75 frames per action</p>
<div align="center">
<br>
<img align="center" width="200" height="300" alt="Screenshot 2023-03-30 at 11 32 57" src="https://user-images.githubusercontent.com/102467763/228855123-4b3e754d-8170-4a9a-b97e-fd2a830affdc.png"><img align="center" width="200" height="100" alt="Screenshot 2023-03-30 at 11 32 57" src="https://user-images.githubusercontent.com/102467763/228855621-7521554d-4419-47c0-9090-814c082c9e50.gif">
<br><br>

Here are some examples of the emotions detected by our model and the corresponding music generated:

Example 1 <br>
[![Sound-IT Demo](https://img.youtube.com/vi/26NKvjD6Ays/0.jpg)](https://www.youtube.com/watch?v=VvsgzLhzV5Q)
<p>Click on the thumbnail to watch the Sound-IT demo.</p>

Getting Started
To get started with Sound-IT, you can clone our repository and follow the instructions in the README file.

To run the code :
-first of all run the pip install -r requirements.txt to install all the packages:
-then change the models path in the UISOUND folder in the files : 
-allinference.py
-inference.py
-inferenceCam.py

for example : 
    weights_1 = '/Users/kevynkrancenblum/Desktop/Data Science/Final Project/Facial_emotion_recognition/saved_models/vggnet.h5'
    weights_2 = '/Users/kevynkrancenblum/Desktop/Data Science/Final Project/Facial_emotion_recognition/saved_models/vggnet_up.h5'
    model_V1=BodySentimentModel(body_input_shape, actions.shape[0])
    model_V1.load_weights('/Users/kevynkrancenblum/Desktop/Data Science/Final Project/Body_Language_recognition/modelsSaved/BodyModelCamv1.h5')

    model_V2=BodySentimentModel(body_input_shape, actions.shape[0])
    model_V2.load_weights('/Users/kevynkrancenblum/Desktop/Data Science/Final Project/Body_Language_recognition/modelsSaved/BodyModelCamv2.h5')
    Change the path to where the your model is located : 

To train on your own emotion or own action recognition base on the body language : 
Go to :  Body_language_recognition/streamlitRecording.py an change the code where you need to add, remove emotions or actions
To train on your own emotion or facial micro emotion train you data by first of all adding your own data then run the model ( IMPORTANT THAT BECAUSE ITS MICRO EMOTION RECOGNITION YOU WILL NEED AN SIGNIFICANTE AMOUNT OF DATA ) : 

<h1> FOR THE LIP READING CONSIDER RUNNING THE CODE IN LipReading/lipnet.ipynb to download the model weight </h1>
or download it using those lines : 
url = 'https://drive.google.com/uc?id=1YlvpDLix3S-U8fd-gqRwPcWXAXm8JwjL'
output = 'data.zip'
gdown.download(url, output, quiet=False)
gdown.extractall('data.zip')

you can also train in you own language and own sentence by creating your own dataset with video and text aligment. important that the models is a CNN+RNN achitechture that mean thats for the Recurent Neural network you <h2> MUST ! </h2> have an predifine sentence lenght for example here 75 frames is the sentence lenght for every video and aligment otherwise the model won't work  
    
Contributing
We welcome contributions from the community. If you have any suggestions or would like to contribute, please open an issue or pull request on our GitHub repository.

