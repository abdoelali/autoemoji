import numpy as np
import cv2

img_rgb = cv2.imread("sad1.png")

def convert(img_rgb):
	num_down = 2       # number of downsampling steps
	num_bilateral = 7  # number of bilateral filtering steps

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

	tmp = []
	for i in range(img_edge.shape[0]):
	    tmp.append(img_color[i][:img_edge.shape[1]])
	img_color = np.array(tmp)


	img_cartoon = cv2.bitwise_and(img_color, img_edge)
	r = 100.0 / img_cartoon.shape[1]
	dim = (100, int(img_cartoon.shape[0] * r))

	# perform the actual resizing of the image and show it
	resized = cv2.resize(img_cartoon, dim, interpolation = cv2.INTER_AREA)
	#cv2.imshow("resized", resized)
	height,width,depth = resized.shape
	circle_img = np.zeros((height,width), np.uint8)
	cv2.circle(circle_img,(width/2,height/2),40,1,thickness=-1)
	masked_data = cv2.bitwise_and(resized, resized, mask=circle_img)
	cv2.imshow("masked", masked_data)
	cv2.imwrite(str(emoji_index) + '.png',masked_data)

# emoji_list = [index, label]


## gives the image an index and  store the image with its label into database
emoji_index = 1
# emoji_index = emoji_index + 1
emoji_label = 1 # it's a loop

index_list = []
label_list = []

#get a new image and recognize whether the new image matches existing emojis
if (len(label_list) > 0):
	for i in range(len(label_list)):
		if (emoji_label == label_list[i]):
			Toshow = cv2.imread(str(index_list[i]) + '.png')
			Toshow.imshow(); #show the image in emoji_list with corresponding emoji_index
			cv2.waitKey()
		else:
			#emoji_list.append([emoji_index,emoji_label])
			index_list.append(emoji_index)
			label_list.append(emoji_label)
			#emoji_index = emoji_index + 1
			#emoji_label = emoji_label + 1
			convert(img_rgb); #show this image
			cv2.waitKey()
else:
	index_list.append(emoji_index)
	label_list.append(emoji_label)
	convert(img_rgb); #show this image
	cv2.waitKey()

print index_list
print label_list

