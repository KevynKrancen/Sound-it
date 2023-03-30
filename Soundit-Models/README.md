<h1 align="center" style="font-size: 48px;">Sound-IT</h1>
Sound-IT is a research project aimed at using artificial intelligence to recognize emotions in videos and create music that fits those emotions. We have created four models to determine the emotional tone of a video: facial expression detection, background hue recognition, body emotion recognition, and lip reading. After that, music is generated that matches the detected emotion.

The proposed method offers a potential solution to the problem of high costs associated with composing and recording original music for films.

<h1 align="center" style="font-size: 48px;">How it works</h1>
Our model is capable of recognizing a scene's dominant emotion by analyzing factors such as body language, facial expressions, lip reading, and background color. Using this emotion, the model generates a new piece of music ideally suited to the scene.
<div align="center">
  <br><br>
  <h1 align="center" style="font-size: 48px;">Sound-it PIPELINE</h1>
  <img align="center" alt="Screenshot 2023-03-30 at 11 32 57" src="https://user-images.githubusercontent.com/102467763/228813572-49ed49ec-b79c-4937-b905-3c933affe210.png"> 
  <br>
 <p>First, We detect what emotion the video is trying to portray by using 4 models:
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
  <img align="center" alt="Screenshot 2023-03-30 at 11 32 57" src="https://user-images.githubusercontent.com/102467763/228815290-950b8dd9-a1d6-49a4-b897-7576f7c332a9.png"> 
    <img align="center" alt="Screenshot 2023-03-30 at 11 32 57" src="https://user-images.githubusercontent.com/102467763/228815392-a4deee54-312f-4798-b7b3-8782ece64353.png"> 


  <h1 align="center" style="font-size: 48px;">Sound-it NN Achitectures & datasets</h1>
<img align="center" width="331" height="450" alt="Screenshot 2023-03-30 at 11 32 57" src="https://user-images.githubusercontent.com/102467763/228778334-103d1250-7fdc-4f0b-be9a-6146af27f999.png"> <img width="331" align="center" height="450" alt="Screenshot 2023-03-30 at 11 33 13" src="https://user-images.githubusercontent.com/102467763/228778357-ea8cc5dc-8087-42be-bd3b-9f78d8307986.png">
</div>
<br><br>


Here are some examples of the emotions detected by our model and the corresponding music generated:

Example 1
[![Sound-IT Demo](https://img.youtube.com/vi/26NKvjD6Ays/0.jpg)](https://www.youtube.com/watch?v=26NKvjD6Ays)
<p>Click on the thumbnail to watch the Sound-IT demo.</p>

Getting Started
To get started with Sound-IT, you can clone our repository and follow the instructions in the README file.

Contributing
We welcome contributions from the community. If you have any suggestions or would like to contribute, please open an issue or pull request on our GitHub repository.


<h1 align="center" style="font-size: 48px;">IMPORTANT ! Acknowledgments </h1>


I would like to take a moment to express my gratitude to Nicolas Renotte for inspiring me to pursue my project. Nicolas Renotte's innovative ideas and achievements in the field have served as a guiding light for my work.

The dedication and commitment that Nicolas Renotte has put into his projects have motivated me to push beyond my limits and strive for excellence in my own work. His vision has been instrumental in shaping my ideas and approach, and I am grateful for his inspiring work.

I would like to thank Nicolas Renotte for setting the bar high and showing me what is possible with passion, hard work, and dedication. Your inspiring work has been a great help in shaping my project, and I am grateful for your contribution to the field.

Once again, I would like to express my heartfelt thanks to Nicolas Renotte for being an inspiration and driving force behind my project.
