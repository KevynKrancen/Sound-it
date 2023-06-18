import cv2
import numpy as np 

def most_frequent(color):   
        return (color.count('Angry'),color.count('Happy'),color.count('Sad'),color.count('Neutral'))

def pourcentageofeachfromlist(color):
        return (color.count('Angry'),color.count('Happy'),color.count('Sad'),color.count('Neutral'))
 

def ColorRecognition(path):
    color=[]
    emotion=''
    cap = cv2.VideoCapture(path)
    num_frames = 0
    average_color = np.array([0.0, 0.0, 0.0])
    while True:
        # Process Key (ESC: end)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
        # Camera capture
        success, image = cap.read()
        if not success:
            break
        num_frames += 1
        image = cv2.flip(image, 1)
        average_color += np.average(np.average(image, axis=0), axis=0)
    cap.release()
    cv2.destroyAllWindows()
    average_color = average_color / num_frames
    average_color = average_color.astype(int)
    
    b, g, r = average_color
    if r >= 128 and g >= 128 and b <= 128:  # Yellow
        emotion = 'Happy'
    elif r <= 128 and g >= 128 and b <= 128:  # Green
        emotion = 'Happy'
    elif r >= 128 and g <= 128 and b >= 128:  # Purple
        emotion = 'Happy'
    elif r >= 192 and g <= 64 and b <= 64:  # Red
        emotion = 'Angry'
    elif r <= 128 and g <= 128 and b >= 128:  # Blue
        emotion = 'Sad'
    elif r <= 16 and g <= 16 and b <= 16:  # Black
        emotion = 'Neutral'
    elif r >= 128 and g >= 128 and b >= 128:  # White or Gray
        if r >= 192 and g >= 192 and b >= 192:  # White
            emotion = 'Neutral'
        else:  # Gray
            if r <= 64 and g <= 64 and b <= 64:  # Deep Gray
                emotion = 'Angry'
            else:
                emotion = 'Neutral'

    return emotion
   
    