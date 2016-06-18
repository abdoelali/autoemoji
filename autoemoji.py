import sys
import os
import wget
import zipfile
import gspread
import matplotlib.pyplot as plt
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score
from skimage import color, exposure, feature, io, transform

import numpy as np
import cv2
# from matplotlib import pyplot as plt

# setup_run = True
# data_base_path = 'https://ait.ethz.ch/public-data/computational_interaction2016/'

# if not os.path.exists('train'):
#     print('[INFO]: Looks like you do not have training data. Let me fetch that for you.')
#     sys.stdout.flush()
#     url_traindata = data_base_path+'train.zip'
#     filename = wget.download(url_traindata)
#     zip_ref = zipfile.ZipFile(filename, 'r')
#     zip_ref.extractall('./')
#     zip_ref.close()
#     print('[INFO]: Training data fetching completed.')
#     sys.stdout.flush()

# if not os.path.exists('./test_T30_R60'):
#     print('[INFO]: Looks like you do not have testing data. Let me fetch that for you')
#     sys.stdout.flush()
#     url_testdata = data_base_path+'test_T30_R60.zip'
#     filename = wget.download(url_testdata)
#     zip_ref = zipfile.ZipFile(filename, 'r')
#     zip_ref.extractall('./')
#     zip_ref.close()
#     print('[INFO]: Testing data fetching completed.')
#     sys.stdout.flush()

# Additionally, there's a second, more challenging dataset that you can download from
# url_testdata_hard = 'https://ait.inf.ethz.ch/teaching/courses/2016-SS-User-Interface-Engineering/downloads/exercises/test_T30_R90.zip '

# Compute accuracy, precision, recall and confusion matrix and (optionally) prints them on screen
def compute_scores(y_pred, y_true, verbose=False):

    hits = 0
    for p in range(1,len(y_true)):
        if y_pred[p] == y_true[p]:
            hits += 1

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    conf_mat = confusion_matrix(y_true, y_pred)

    if(verbose):
        print ("(RW) Accuracy: " + str(accuracy) + "(" + str(hits) + "/" + str(len(y_true)) + ")")
        print ("Precision: " + str(precision))
        print ("Recall: " + str(recall))
        print ("Confusion Matrix")
        print (conf_mat)
        sys.stdout.flush()

    return accuracy, precision, recall


# Extract HOG features from an image and (optionally) show the features superimposed on it
def extractHOG(inputimg, showHOG=False):

    # convert image to single-channel, grayscale
    image = color.rgb2gray(inputimg)

    #extract HOG features
    if showHOG:
        fd, hog_image = feature.hog(image, orientations=36,
                                    pixels_per_cell=(16, 16),
                                    cells_per_block=(2, 2),
                                    visualise=showHOG)
    else:
        fd = feature.hog(image, orientations=8, pixels_per_cell=(16, 16),
                         cells_per_block=(1, 1), visualise=showHOG)
    if(showHOG):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
        ax1.axis('off')
        ax1.imshow(image, cmap=plt.cm.gray)
        ax1.set_title('Input image')
        ax1.set_adjustable('box-forced')
        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        ax1.set_adjustable('box-forced')
        plt.show()
    return fd


# Load a dataset (Data, Labels) from a folder.
# Return data (HOGs, Class) and image list (as image file ames on disk)
def load_dataset_from_folder(root_folder, rgb_folder, segmentation_folder):

    HOGs_list = []
    Cs_list = []
    image_list = []
    if os.path.exists(root_folder):
        class_folders = next(os.walk(root_folder))[1]
        class_folders.sort()
        print("[INFO] Found " + str(len(class_folders)) + " class folders")
        print(class_folders)
        sys.stdout.flush()
        tot_classes = len(class_folders)
        #used to resize the images
        image_size = (128, 128)
        class_list = range(tot_classes)
        for class_folder,this_class in zip(class_folders,class_list):
            print("\n[INFO] Processing folder " + class_folder)
            sys.stdout.flush()
            current_gesture_folder_rgb = root_folder + class_folder + "/" + rgb_folder + "/*.jpg"
            current_gesture_folder_segmentation = root_folder + class_folder + "/" + segmentation_folder + "/*.png"
            allfiles_imgs = glob.glob(current_gesture_folder_rgb)
            allfiles_masks = glob.glob(current_gesture_folder_segmentation)
            #for each image/mask pair
            line_percentage_cnt = 0
            for file_img,mask_img in zip(allfiles_imgs,allfiles_masks):
                # Print completion percentage
                sys.stdout.write('\r')
                progress_bar_msg = "[%-100s] %d%% " + str(line_percentage_cnt) + "/" + str(len(allfiles_imgs))
                update_step = int( (float(100)/float(len(allfiles_imgs))) * float(line_percentage_cnt) )
                sys.stdout.write(progress_bar_msg % ('='*update_step, update_step))
                sys.stdout.flush()
                img = io.imread(file_img)
                mask = io.imread(mask_img)
                mask = 255 - mask
                img *= mask
                # you can see the segmented image using:
                #io.imshow(img)
                #io.show()
                feat = extractHOG(transform.resize(img, image_size))
                HOGs_list.append(feat)
                Cs_list.append(this_class)
                image_list.append(file_img)
                line_percentage_cnt += 1
        print("[INFO] Loaded data in. Number of samples: "+ str(len(image_list)))
    else:
        print("[ERROR] Folder " + root_folder + " does not exist!")
        print("[ERROR] Have you run the setup cell?")
        sys.stdout.flush()
        exit()


    HOGs = np.array(HOGs_list)
    Cs = np.array(Cs_list)
    return HOGs, Cs, image_list



# Class to store parameters of an SVM
class SVMparameters:

    def __init__(self, k='rbf', c='1', g='0.1', d=1):
        self.kernel = k
        self.C = c
        self.gamma=g
        self.degree = g

    def setkernel(self, k):
        self.kernel = k

    def setgamma(self, g):
        self.gamma = g

    def setc(self, c):
        self.C = c

    def setdegree(self,d):
        self.degree = d

    def printconfig(self):
        print("Kernel: " + self.kernel)
        if self.kernel is "poly":
            print("Degree: " + str(self.degree))
        print("C: " + str(self.C))
        print("Gamma: " + str(self.gamma))
        sys.stdout.flush()




train_folders = next(os.walk("KDEF/"))[1]
train_folders.sort()
print("[INFO] Found " + str(len(train_folders)) + " class folders")
sys.stdout.flush()
tot_classes = len(train_folders)

class_list = range(tot_classes)

image_size = (128, 128)

image_list = []
HOGs_list = []
Cs_list = []

emotion_list = ['AFS.JPG', 'ANS.JPG', 'DIS.JPG', 'HAS.JPG', 'NES.JPG', 'SAS.JPG', 'SUS.JPG']

ffile = []

for class_folder,this_class in zip(train_folders,class_list):
#     print("\n[INFO] Processing folder " + class_folder)
    sys.stdout.flush()

    ffile = os.listdir('KDEF/' + class_folder)

    for i in ffile:

        for j in range(len(emotion_list)):

            if i.endswith(emotion_list[j]): #afraid

                directory = 'KDEF/train/' + emotion_list[j][0:3] + '/' + i

#                 print directory

                if not os.path.exists(directory):
                    os.makedirs(directory)

                tmp_img = io.imread("KDEF/" + class_folder + "/" + i)
                cv2.imwrite(str(directory), tmp_img)


#                 print i

#                 current_face_folder_rgb = class_folder + "/" + i

#                 print current_face_folder_rgb

#                 sys.stdout.write('\r')
#                 progress_bar_msg = "[%-100s] %d%% " + str(line_percentage_cnt) + "/" + str(len(allfiles_imgs))
#                 update_step = int( (float(100)/float(len(allfiles_imgs))) * float(line_percentage_cnt) )
#                 sys.stdout.write(progress_bar_msg % ('='*update_step, update_step))
#                 sys.stdout.flush()

#                 feat = extractHOG(transform.resize(img, image_size))
#                 HOGs_list.append(feat)
#     #             Cs_list.append(this_class)
#                 image_list.append(file_img)


#                 # img = cv2.imread("KDEF/AF01/AF01AFFL.JPG")

#                 img = cv2.imread("KDEF/AF01/AF01AFFL.JPG")
#                 gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#                 ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#                 # noise removal
#                 kernel = np.ones((3,3),np.uint8)
#                 opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

#                 # sure background area
#                 sure_bg = cv2.dilate(opening,kernel,iterations=3)

#                 # Finding sure foreground area
#                 dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
#                 ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

#                 # Finding unknown region
#                 sure_fg = np.uint8(sure_fg)
#                 mask_img = cv2.subtract(sure_bg,sure_fg)

#                 cv2.imwrite('KDEF/test.png', mask_img)

#                 # io.imshow(mask_img)
#                 # io.show()

#                 # img = io.imread("KDEF/AF01/AF01AFFL.JPG")
#                 mask = io.imread("KDEF/test.png")
#                 mask = 255 - mask

#                 # print mask.shape
#                 # print gray.shape
#                 # (762, 562, 3)
#                 # 388 rows, 647 columns, and 3 channels (the RGB components).

#                 gray *= mask
#                 feat = extractHOG(transform.resize(img, image_size))
#                 extractHOG(img, showHOG=False)


#     current_gesture_folder_segmentation = root_folder + class_folder + "/" + segmentation_folder + "/*.png"

# #used to resize the images
# image_size = (128, 128)
# class_list = range(tot_classes)
# for class_folder,this_class in zip(class_folders,class_list):
#     print("\n[INFO] Processing folder " + class_folder)
#     sys.stdout.flush()
#     current_gesture_folder_rgb = root_folder + class_folder + "/" + rgb_folder + "/*.jpg"
#     current_gesture_folder_segmentation = root_folder + class_folder + "/" + segmentation_folder + "/*.png"
