# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import time
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
	help="path to input image")
ap.add_argument("-east", "--east", type=str,
	help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
	help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=320,
	help="resized image width (should be multiple of 32)")
ap.add_argument("-e", "--height", type=int, default=320,
	help="resized image height (should be multiple of 32)")
args = vars(ap.parse_args())

# Detect and localize the text in image
def detection(image):
	
	orig = image.copy()
	cv2.imshow('orig_image',orig)
	cv2.waitKey()
	(H, W) = image.shape[:2]
	# set the new width and height and then determine the ratio in change for both the width and height
	(newW, newH) = (args["width"], args["height"])
	rW = W / float(newW)
	rH = H / float(newH)
	# resize the image and grab the new image dimensions
	image = cv2.resize(image, (newW, newH))
	(H, W) = image.shape[:2]
	# cv2.imshow('image',image)
	# cv2.waitKey()

	# define the two output layer names for the EAST detector model that
	# we are interested -- the first is the output probabilities and the
	# second can be used to derive the bounding box coordinates of text
	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]

	# load the pre-trained EAST text detector
	print("[INFO] loading EAST text detector...")
	net = cv2.dnn.readNet(args["east"])
	# construct a blob from the image and then perform a forward pass of
	# the model to obtain the two output layer sets
	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	start = time.time()
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)
	end = time.time()
	# show timing information on text prediction
	print("[INFO] text detection took {:.6f} seconds".format(end - start))

	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []
	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the geometrical
		# data used to derive potential bounding box coordinates that
		# surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

	# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability, ignore it
			if scoresData[x] < args["min_confidence"]:
				continue
			# compute the offset factor as our resulting feature maps will
			# be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)
			# extract the rotation angle for the prediction and then
			# compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)
			# use the geometry volume to derive the width and height of
			# the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]
			# compute both the starting and ending (x, y)-coordinates for
			# the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x])) 
			startX = int(endX - w) 
			startY = int(endY - h) 
			# add the bounding box coordinates and probability score to
			# our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	boxes = non_max_suppression(np.array(rects), probs=confidences)

	# loop over the bounding boxes

	startYs = []
	startXs = []
	endXs = []
	endYs = []
	ws = []
	hs = []
	for (startX, startY, endX, endY) in boxes:
		
		# scale the bounding box coordinates based on the respective
		# ratios
		startX = int(startX * rW) 
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH) 

		w = endX - startX
		h = endY - startY
		ws.append(w)
		hs.append(h)
		startYs.append(startY)
		startXs.append(startX)
		endXs.append(endX)
		endYs.append(endY)

		cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 1)

	cv2.imshow('orig',orig)
	cv2.waitKey()

	return orig ,startXs,startYs, endXs,endYs,ws,hs

def crop_image():
	img = cv2.imread(args["image"])
	img = img[1:img.shape[0] // 5 * 4,1:img.shape[1]]
	orig,startXs,startYs, endXs,endYs,ws,hs = detection(img)

	rois_list= list(zip(startXs,startYs, endXs,endYs,ws,hs))
	min_startx = min(startXs)
	max_endx = max(endXs)
	span = max_endx - min_startx

	for roi in rois_list:
		if roi[0] == min_startx:
			start_width = roi[4]
			start_endx = roi[2]

	if span/orig.shape[1] > 0.4: # check if the text on the left side of container door is detected
		if start_width / span <= 0.7:
			orig = orig[1: orig.shape[0], (min_startx + span//3) : orig.shape[1]]
		else:
			orig = orig[1: orig.shape[0], start_endx : orig.shape[1]]

	cv2.imshow('box',orig)
	cv2.waitKey(0)
	return orig

# Localize plate numbers
def crop_image_again():
	cropped_image = crop_image()
	second_detection,startXs,startYs, endXs,endYs,ws,hs = detection(cropped_image)
	rois_list= list(zip(startXs,startYs, endXs,endYs,ws,hs))
	startYs.sort()
	startYs = startYs[:3]
	plates = []
	for y in startYs:
		for roi in rois_list:
			if roi[1] == y :
				plates.append(roi)
				rois_list.remove(roi)
	plates = list(dict.fromkeys(plates))

	start_x = min(plates[0][0],plates[1][0],plates[2][0])
	start_y = min(plates[0][1],plates[1][1],plates[2][1])
	end_x = max(plates[0][2],plates[1][2],plates[2][2])
	end_y = max(plates[0][3],plates[1][3],plates[2][3])

	expand_w = max(plates[0][4],plates[1][4],plates[2][4])
	expand_h = max(plates[0][5], plates[1][5], plates[2][5])

	y = start_y - expand_h
	new_y = end_y + expand_h //2 *3
	x = start_x - expand_w
	new_x = end_x + expand_w

	if y <= 1:
		y = 1
	if new_y >= second_detection.shape[0]:
		new_y = second_detection.shape[0]
	if x <= 1:
		x = 1
	if new_x >= second_detection.shape[1]:
		new_x = second_detection.shape[1]

	crop = second_detection[y:new_y, x:new_x]
	# plate_path = os.path.splitext(args["image"])[0]
	# cv2.imwrite(plate_path+'_crop.jpg',crop)

	cv2.imshow('crop',crop)
	cv2.waitKey(0)
	return crop

# crop and save plates images 
def get_plate_crop():
	crop = crop_image_again()
	_,startXs,startYs, endXs,endYs,ws,hs = detection(crop)
	rois_list= list(zip(startXs,startYs, endXs,endYs,ws,hs))
	startYs.sort()
	startYs = startYs[:3]
	plates = []
	for y in startYs:
		for roi in rois_list:
			if roi[1] == y :
				plates.append(roi)
				rois_list.remove(roi)
	plates = list(dict.fromkeys(plates))
	plate_path = os.path.splitext(args["image"])[0]
	
	os.mkdir(plate_path)

	orig = crop.copy()
	i = 0
	for plate in plates:
		i +=1
		cv2.rectangle(orig, (plate[0], plate[1]), (plate[2], plate[3]), (0, 255, 0), 1)
		plate_img = orig[plate[1]-2:plate[3]+5, plate[0]-2:plate[2]+5]
		cv2.imwrite(plate_path+'/plate%d.jpg'%(i),plate_img)
		
	cv2.imshow('orig',orig)
	cv2.waitKey(0)

	return plate_path

# get_plate_crop()
img = cv2.imread(args["image"])
detection(img)
