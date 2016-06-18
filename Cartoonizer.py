import numpy as np
import cv2
 
num_down = 2       # number of downsampling steps
num_bilateral = 7  # number of bilateral filtering steps
 
img_rgb = cv2.imread("scottishfold.jpg")
 
# # downsample image using Gaussian pyramid
img_color = img_rgb
for _ in xrange(num_down):
    img_color = cv2.pyrDown(img_color)
 
# # repeatedly apply small bilateral filter instead of
# # applying one large filter
for _ in xrange(num_bilateral):
    img_color = cv2.bilateralFilter(img_color, d=9,sigmaColor=9, sigmaSpace=7)
 
for _ in xrange(num_down):
    img_color = cv2.pyrUp(img_color)


# # convert to grayscale and apply median blur
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
img_blur = cv2.medianBlur(img_gray, 7)




# # detect and enhance edges
img_edge = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
    cv2.THRESH_BINARY, blockSize=9, C=2)


# # convert back to color, bit-AND with color image
img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)


# img_edge = cv2.resize(img_edge,None,fx=img_color.shape[0], fy=img_color.shape[1], interpolation = cv2.INTER_AREA)
#cv2.imshow("edge", img_edge)
#cv2.imshow("img_color", img_color)


tmp = []
for i in range(img_edge.shape[0]):
    tmp.append(img_color[i][:img_edge.shape[1]])
img_color = np.array(tmp)
print img_color.shape


img_cartoon = cv2.bitwise_and(img_color, img_edge)
 
# # display
cv2.imshow("cartoon", img_cartoon)
cv2.waitKey()

