<h1 align="center" style="font-size: 48px;">Sound-IT</h1>
Sound-IT is a research project aimed at using artificial intelligence to recognize emotions in videos and create music that fits those emotions. We have created four models to determine the emotional tone of a video: facial expression detection, background hue recognition, body emotion recognition, and lip reading. After that, music is generated that matches the detected emotion.

The proposed method offers a potential solution to the problem of high costs associated with composing and recording original music for films.

<h1 align="center" style="font-size: 48px;">How it works</h1>
Our model is capable of recognizing a scene's dominant emotion by analyzing factors such as body language, facial expressions, lip reading, and background color. Using this emotion, the model generates a new piece of music ideally suited to the scene.
<div align="center">
  <br><br>
  <h1 align="center" style="font-size: 48px;">Sound-it PIPELINE</h1>
  <img align="center" alt="Screenshot 2023-03-30 at 11 32 57" src="https://user-images.githubusercontent.com/102467763/228813572-49ed49ec-b79c-4937-b905-3c933affe210.png"> 
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
<img align="center" alt="Screenshot 2023-03-30 at 11 32 57" src="https://user-images.githubusercontent.com/102467763/228815290-950b8dd9-a1d6-49a4-b897-7576f7c332a9.png"><img align="center" alt="Screenshot 2023-03-30 at 11 32 57" src="https://user-images.githubusercontent.com/102467763/228815392-a4deee54-312f-4798-b7b3-8782ece64353.png"> 
</div>


<h1 align="center" style="font-size: 48px;">Approch for Facial Emotion Recognition</h1>
<p align="center" >Using FER2013 as dataset and implementing VGG16 Neural Network Achitechture</p>
<div align="center">
<br>
<img align="center" width="400" height="300" alt="Screenshot 2023-03-30 at 11 32 57" src="https://user-images.githubusercontent.com/102467763/228867413-becccc6d-1a3e-43df-abba-6b6b83f54d01.png"><img align="center" width="350" height="300" alt="Screenshot 2023-03-30 at 11 32 57" src="https://user-images.githubusercontent.com/102467763/228867789-a14e6978-bd91-4931-86bd-9e233f22d217.png">
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
<br><br> 
[![Sound-IT Demo](https://img.youtube.com/vi/26NKvjD6Ays/0.jpg)](https://www.youtube.com/watch?v=26NKvjD6Ays) 
<br><br>
<p align="center">Click on the thumbnail to watch the Sound-IT demo.</p>

Getting Started
To get started with Sound-IT, you can clone our repository and follow the instructions in the README file.

Contributing
We welcome contributions from the community. If you have any suggestions or would like to contribute, please open an issue or pull request on our GitHub repository.


<h1 align="center" style="font-size: 48px;">IMPORTANT ! Acknowledgments </h1>


I would like to take a moment to express my gratitude to Nicolas Renotte for inspiring me to pursue my project. Nicolas Renotte's innovative ideas and achievements in the field have served as a guiding light for my work.
  
I would like to thank Nicolas Renotte for setting the bar high and showing me what is possible with passion, hard work, and dedication. Your inspiring work has been a great help in shaping my project, and I am grateful for your contribution to the field.

Once again, I would like to express my heartfelt thanks to Nicolas Renotte for being an inspiration and driving force behind my project.
