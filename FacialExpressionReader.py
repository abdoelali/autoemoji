
# coding: utf-8

# In[70]:

import numpy as np
import cv2
import time

# def readFacialExpression():
#     cap = cv2.VideoCapture(0)

#     w = cap.get(cv2.CAP_PROP_FRAME_WIDTH);
#     h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT);

#     # Define the codec and create VideoWriter object
#     fourcc = cv2.VideoWriter_fourcc('d','i','v','x')
#     out = cv2.VideoWriter('output.avi',fourcc, 25.0, (int(w),int(h)))
#     #out = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))

#     timeInit = time.time()
#     # only record for 200 ms
#     while(cap.isOpened() & (500 > ((time.time() - timeInit)*1000))):
#         ret, frame = cap.read()
#         if ret==True:
#             #frame = cv2.flip(frame,0)
#             # write the flipped frame
#             out.write(frame)

#         else:
#             break

#     # Release everything if job is finished
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()


# In[71]:

def extractFacialFeatures(ffile):

    return 0,0,0


# In[72]:

from PIL import Image

# snapshot used as input to trained classifier (e.g., scikit.svm.predict(snapshot)).
# returns label (happy, sad, neutral, angry)
def classifyFacialExpression():

    readFacialExpression()
    HOGs_list = []
    Cs_list = []
    best_list = []

    cap = cv2.VideoCapture('output.avi')

    print int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()

        pil_im = Image.fromarray(frame)

        # pass resizedFrame to a-boooody's classifier
        HOGs, Cs, bestFrame = extractFacialFeatures(frame)

        HOGs_list.append(HOGs)
        Cs_list.append(Cs)
        best_list.append(bestFrame)

    ## filter according to some criterion and find the  best image!

    cap.release()
    cv2.destroyAllWindows()

    return bestFrame


# In[73]:

classifyFacialExpression()


# In[ ]:



