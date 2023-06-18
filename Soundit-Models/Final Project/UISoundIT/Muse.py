import string
import pandas as pd
import numpy as np


def randomArtist(emotion):

    dataFrame = pd.read_csv('/Users/kevynkrancenblum/Desktop/Data Science/Final Project/UISoundIT/muse_v3_3.csv')

    string_list_happy=['uplifting','cheerful','happy','euphoria']
    string_list_sad=['sad']
    string_list_fear=['scared','spooky','halloween']
    string_list_angry=['angry']

    Result=[]

    if emotion == 'Happy':
        happy=[]
        for word in string_list_happy:
            happy.append(dataFrame[dataFrame['seeds'].str.contains(word)])
        happy=pd.concat(happy)
        row = happy.sample(n=1)
        Result.append(row['artist'].to_string(index=False))
        Result.append(row['genre'].to_string(index=False))
        print(row['artist'].to_string(index=False) + '\n' + row['genre'].to_string(index=False))


    elif emotion == 'Sad':
        sad=[]
        for word in string_list_sad:
            sad.append(dataFrame[dataFrame['seeds'].str.contains(word)])

        sad=pd.concat(sad)
        row = sad.sample(n=1)
        Result.append(row['artist'].to_string(index=False))
        Result.append(row['genre'].to_string(index=False))
        print(row['artist'].to_string(index=False) + '\n' + row['genre'].to_string(index=False))

    elif emotion == 'Fear':
        fear=[]
        for word in string_list_fear:
            fear.append(dataFrame[dataFrame['seeds'].str.contains(word)])

        fear=pd.concat(fear)
        row = fear.sample(n=1)
        Result.append(row['artist'].to_string(index=False))
        Result.append(row['genre'].to_string(index=False))
        print(row['artist'].to_string(index=False) + '\n' + row['genre'].to_string(index=False))

    elif emotion == 'Angry':
        angry=[]
        for word in string_list_angry:
            angry.append(dataFrame[dataFrame['seeds'].str.contains(word)])

        angry=pd.concat(angry)
        row=angry.sample(n=1)
        Result.append(row['artist'].to_string(index=False))
        Result.append(row['genre'].to_string(index=False))
        print(row['artist'].to_string(index=False)+'\n'+row['genre'].to_string(index=False))

    return Result
