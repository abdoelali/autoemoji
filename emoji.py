
from window import window_data

## The skeleton of a solution

import window
import numpy as np
import sklearn.datasets, sklearn.linear_model, sklearn.neighbors
import matplotlib.pyplot as plt
#import seaborn as sns
import sys, os, time
import scipy.io.wavfile, scipy.signal
#%matplotlib inline
import matplotlib as mpl
from IPython.core.display import HTML
mpl.rcParams['figure.figsize'] = (18.0, 10.0)
import pandas as pd
import time
import cv2
import pyaudio
import wave

# load the wave file and normalise
# load "data/rub_1.wav" and "data/rub_2.wav"

def load_wave(fname):
    # load and return a wave file
    sr, wave = scipy.io.wavfile.read(fname)
    return wave/32768.0



def emoji_classify():

    rub_1 = load_wave("emoji.wav")[:,0]
    rub_2 = load_wave("anythingelse.wav")[:,0]

    rub_1_features = window.window_data(rub_1, 120)
    rub_2_features = window.window_data(rub_2, 120)

    rub_1_labels = np.zeros(len(rub_1_features,))
    rub_2_labels = np.ones(len(rub_2_features,))

    rub_features = np.vstack([rub_1_features, rub_2_features])
    rub_labels = np.hstack([rub_1_labels, rub_2_labels])

    rubfft_features =  np.abs(np.fft.fft(rub_features))
    rubfft_train_features, rubfft_test_features, rub_train_labels, rub_test_labels = sklearn.cross_validation.train_test_split(
        rubfft_features, rub_labels, test_size=0.3, random_state=0)

    rub_train_features, rub_test_features, rub_train_labels, rub_test_labels = sklearn.cross_validation.train_test_split(
        rub_features, rub_labels, test_size=0.3, random_state=0)

    #print rub_train_features.shape, rub_train_labels.shape

    svm = sklearn.svm.SVC(gamma=0.1, C=100)
    svm.fit(rub_train_features, rub_train_labels)


    return svm

svm = emoji_classify()

CHUNK = 1024 
FORMAT = pyaudio.paInt16 #paInt8
CHANNELS = 2 
RATE = 44100 #sample rate
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK) #buffer

print("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data) # 2 bytes(16 bits) per channel

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

recorded = load_wave("output.wav")[:,0]

recorded_features = window.window_data(recorded, 120)

recorded_features = np.vstack(recorded_features)
recorded_labels = svm.predict(recorded_features)

def readFacialExpression():
    cap = cv2.VideoCapture(0)
    
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH);
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT); 

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc('d','i','v','x')
    out = cv2.VideoWriter('output.avi',fourcc, 25.0, (int(w),int(h)))
    #out = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))

    timeInit = time.time()
    # only record for 200 ms 
    while(cap.isOpened() & (500 > ((time.time() - timeInit)*1000))):
        ret, frame = cap.read()
        if ret==True:
            #frame = cv2.flip(frame,0)
            # write the flipped frame
            out.write(frame)

        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
#print recorded_labels

#window_index to decide whether to open the window or not
window_index = 0

for i in range(len(recorded_labels.tolist())):
    if (recorded_labels[i] == 0 ):
        window_index = 1

if (window_index == 1):
    readFacialExpression()





