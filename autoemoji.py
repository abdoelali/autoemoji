
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




# Process

# - get external labeled facial images from available datasets
# - image database: extract relevant features, e.g., color histograms, contours, etc. (use off shelf packages)
# - features split into training and test data
# - classifier must be trained on extracted training features

# - Voice input: “emoji” command, starts window function (e.g., 200 ms), draws randomly sampled snapshot
# - snapshot used as input to trained classifier (e.g., scikit.svm.predict(snapshot)). returns label (happy, sad, neutral, angry)
# - get label and input image, and convert image to cartoon. store final cartoon image into database, with associated label. this gives personalized emoji database
# - user says “emoji”, and if facial features matches existing emoji in database
# - two windows, one with your face and other with output emoji

# corner features, sift features, gabor features






# load the wave file and normalise
# load "data/rub_1.wav" and "data/rub_2.wav"

def load_wave(fname):
    # load and return a wave file
    sr, wave = scipy.io.wavfile.read(fname)
    return wave/32768.0



def breath_classify():

    rub_1 = load_wave("data/in.wav")[:,0]
    rub_2 = load_wave("data/out.wav")[:,0]

    rub_1_features = window.window_data(rub_1, 120)
    rub_2_features = window.window_data(rub_2, 120)

    rub_1_labels = np.zeros(len(rub_1_features,))
    rub_2_labels = np.ones(len(rub_2_features,))

    rub_features = np.vstack([rub_1_features, rub_2_features])
    rub_labels = np.hstack([rub_1_labels, rub_2_labels])
    print rub_features.shape, rub_labels.shape

    rubfft_features =  np.abs(np.fft.fft(rub_features))
    rubfft_train_features, rubfft_test_features, rub_train_labels, rub_test_labels = sklearn.cross_validation.train_test_split(
        rubfft_features, rub_labels, test_size=0.3, random_state=0)

    rub_train_features, rub_test_features, rub_train_labels, rub_test_labels = sklearn.cross_validation.train_test_split(
        rub_features, rub_labels, test_size=0.3, random_state=0)

    print rub_train_features.shape, rub_train_labels.shape

    svm = sklearn.svm.SVC(gamma=0.1, C=100)
    svm.fit(rub_train_features, rub_train_labels)

    # we can plot the receiver-operator curve: the graph of false positive rate against true positive rate
    # scores = svm.decision_function(rub_test_features)
    # print scores.shape, rub_test_labels.shape
    # fpr, tpr, thresholds = sklearn.metrics.roc_curve(rub_test_labels, scores)
    # plt.plot(fpr,tpr)
    # plt.plot([0,1], [0,1])
    # plt.plot([0,1], [1,0])
    # plt.fill_between(fpr, tpr, facecolor='none', hatch='/', alpha=0.2)
    # plt.xlabel("False positive rate")
    # plt.ylabel("True positive rate")
    # plt.legend(["ROC", "Chance", "EER line"])

    # print svm

    return svm


breath_classify()
 # takes feature vector and returns a label


# split into windows

# test/train split

# train classifier

# evaluate classifier

