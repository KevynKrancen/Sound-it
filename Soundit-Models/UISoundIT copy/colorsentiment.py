import cv2
import numpy as np 
def ColorRecognition(path):
    color=[]
    cap = cv2.VideoCapture(path)
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
        image = cv2.flip(image, 1)
        average_color_row = np.average(image, axis=0)
        average_color = np.average(average_color_row, axis=0)
        d_img = np.ones((312,312,3), dtype=np.uint8)
        d_img[:,:] = average_color
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Classify emotions based on color
        b, g, r = average_color.astype(int)
        if r >= 128 and g <= 160 and b <= 147:  # Red
            color.append('Angry')
        elif r <= 173 and g >= 100 and b <= 170:  # Green
            color.append('Happy')
        elif r <= 240 and g <= 248 and b >= 112:  # Blue
            color.append('Sad')
        elif r > 240 or g > 248 or b > 170:  # White
            color.append('Neutral')

        print(color)

    cap.release()
    cv2.destroyAllWindows()

    def most_frequent(color):   
        return (color.count('Angry'),color.count('Happy'),color.count('Sad'),color.count('Neutral'))
    def pourcentageofeachfromlist(color):
        return (color.count('Angry'),color.count('Happy'),color.count('Sad'),color.count('Neutral'))
    return most_frequent(color)   
    