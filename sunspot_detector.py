from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2
import glob
import time
import re
from datetime import datetime
import itertools
import math

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True,
	help="path to the image folder")

args = ap.parse_args()

def findSun(image):
	# detect circles in the image
	circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1.2, 100)

	# ensure at least some circles were found
	if circles is not None:
		circles = np.round(circles[0, :]).astype("int")
		for (xc, yc, r) in circles:
			H, W = image.shape
			x, y = np.meshgrid(np.arange(W), np.arange(H))
			d2 = (x - xc)**2 + (y - yc)**2
			mask = d2 < (r-10)**2
			image[~mask] = 0
			cv2.circle(image, (xc, yc), r, 0, 7)

			#cv2.rectangle(image, (xc - 5, yc - 5), (xc + 5, yc + 5), (0, 128, 255), -1)

		# show the output image
		#cv2.imshow("output", image)
		#cv2.waitKey(0)
		return image,r,(xc,yc)

def toOrthographicSphereCoords(x,y, R):
    p = np.sqrt(x**2 + y**2)
    c = np.arcsin(p/R)
    lat0 = 0
    lon0 = 0
    lat = np.arcsin(np.cos(c)*lat0 + (y*np.sin(c)*np.cos(lat0))/p )
    lon = lon0 + np.arctan2(x*np.sin(c),p*np.cos(c)*np.cos(lat0) - y*np.sin(c)*np.sin(lat0))
    return np.array((lon,lat))

def filterPossibleMatches(matchedPairs):
	if not matchedPairs:
		return [] 
	sumOfMovement = 0
	for mp in matchedPairs:
		sumOfMovement += mp[2]
	averageMovement = sumOfMovement/len(matchedPairs)
	filteredPairs = []
	remainingPairs = matchedPairs
	while(remainingPairs):
		minPair = min(remainingPairs , key=lambda x: abs(x[2]-averageMovement))
		remainingPairs = list(filter(lambda x: not np.array_equal(x[0], minPair[0]), remainingPairs))
		remainingPairs = list(filter(lambda x: not np.array_equal(x[1], minPair[1]), remainingPairs))
		filteredPairs.append(minPair)
	return filteredPairs

def matchSpotsBetweenFrames(oldCenters, newCenters):
	if not oldCenters or not newCenters:
		return []
	allPairedCenters = []
	for oc in oldCenters:
		for nc in newCenters:
			movementVector = nc-oc
			movementDistance = np.linalg.norm(movementVector)
			if movementVector[0] > 0 and movementDistance < 70:
				allPairedCenters.append((oc,nc,movementDistance))
	
	return filterPossibleMatches(allPairedCenters)

def findSunSpots(image,blurred):
	thresh = cv2.threshold(blurred, 140, 255, cv2.THRESH_BINARY)[1]

	labels = measure.label(thresh, neighbors=8, background=0)
	mask = np.zeros(thresh.shape, dtype="uint8")

	for label in np.unique(labels):
		if label == 0:
			continue

		labelMask = np.zeros(thresh.shape, dtype="uint8")
		labelMask[labels == label] = 255
		numPixels = cv2.countNonZero(labelMask)

		if numPixels < 300:
			mask = cv2.add(mask, labelMask)

	cntrs = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cntrs = imutils.grab_contours(cntrs)
	if len(cntrs) < 1:
		return [],image
	cntrs = contours.sort_contours(cntrs)[0]

	spotCenters = []
	for (i, c) in enumerate(cntrs):
		(x, y, w, h) = cv2.boundingRect(c)
		((cX, cY), radius) = cv2.minEnclosingCircle(c)
		spotCenters.append(np.array((cX,cY),dtype=np.int32))
		cv2.circle(image, (int(cX), int(cY)), int(radius),
			(0, 0, 255), 3)
		cv2.putText(image, "#{}".format(i + 1), (x, y - 15),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
	cv2.circle(image, sunCenter, sunRadius, (0,255,0), 2)
	return spotCenters,image

imageFilenames = glob.glob("{imagePath}/*.jpg".format(imagePath=args.images)) + glob.glob("{imagePath}/*.png".format(imagePath=args.images))
imageFilenames.sort()

oldCenters = None
lastRecordTime = None
for i in imageFilenames:
	timeString = re.search(r'[\d]{8}_[\d]{4}', i).group()
	recordTime = datetime.strptime(timeString, '%Y%m%d_%H%M')
	print(str(recordTime)+":")
	if lastRecordTime:
		print("DT = {DT}\n".format(DT=recordTime-lastRecordTime))
	lastRecordTime = recordTime

	# load image, convert to grayscale, and blur it
	image = cv2.imread(i)

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = ~gray
	
	#gray[gray > 250] = 0
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	#blurred = gray
	blurred,sunRadius,sunCenter = findSun(blurred)
	print(sunRadius)
	newCenters,image = findSunSpots(image,blurred)
	timePairedCenters = matchSpotsBetweenFrames(oldCenters,newCenters)
	for tpc in timePairedCenters:
		cv2.arrowedLine(image, tuple(tpc[0]),tuple(tpc[1]),(0, 255, 0),2)
		tpc = (tpc[0] - np.array(sunCenter),tpc[1] - np.array(sunCenter))
		diffX = tpc[1] - tpc[0]
		print("X = ({X1},{X2})".format(X1=tpc[0][0],X2=tpc[0][1]))
		print("DX = ({DX1},{DX2})\n".format(DX1=diffX[0],DX2=diffX[1]))
		l0 = (toOrthographicSphereCoords(tpc[0][0],tpc[0][1], sunRadius)/math.pi)*180
		l1 = (toOrthographicSphereCoords(tpc[1][0],tpc[1][1], sunRadius)/math.pi)*180
		print("L = ({L1_0},{L1_1})".format(L1_0=l1[0],L1_1=l1[1]))
		diffL = l1 - l0
		print("DL = ({DL1},{DL2})\n".format(DL1=diffL[0],DL2=diffL[1]))
	oldCenters = newCenters
	cv2.imshow("Image", image)
	print("-"*30)
	cv2.waitKey(500)