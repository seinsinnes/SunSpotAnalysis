import imutils
import cv2
from imutils import contours
from skimage import measure
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import argparse, glob, time, re, itertools, math, copy
from datetime import datetime


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="Path to the image folder")
ap.add_argument('-d', "--display", help='Display image frames during processing', action='store_true')

args = ap.parse_args()

class SunImage:
    def __init__(self,image):
        self.image = image
        self.radius = -1
        self.centerCoords = None
        self.findSun(image)
    
    def findSun(self, image):
        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1.2, 100)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (xc, yc, r) in circles:
                H, W = image.shape
                x, y = np.meshgrid(np.arange(W), np.arange(H))
                d2 = (x - xc)**2 + (y - yc)**2
                mask = d2 < (r-10)**2
                image[~mask] = 0
                cv2.circle(image, (xc, yc), r, 0, 7)

            self.image = image
            self.radius = r
            self.centerCoords = (xc, yc)

class SunSpot:
    def __init__(self, sunImage, cartesianCenterCoords, radius, boundingRect, filename, time):
        self.sunImage = sunImage
        self.cartesianCoords = cartesianCenterCoords
        self.filename = filename
        self.time = time
        self.boundingRect = boundingRect

    def getSunCenteredCartestianCoords(self):
        return self.cartesianCoords - np.array(self.sunImage.centerCoords)

    def getSunCenteredBoundingRectCoords(self):
        return np.array((self.boundingRect[0],self.boundingRect[1])) - np.array(self.sunImage.centerCoords)

    def getOrthographicSphereBoundingRectDegrees(self):   
        sunCenteredBoundingBoxCoords = self.getSunCenteredCartestianCoords()
        boxSphericalPoints = []
        for boxCorners in ((0,0),(0, self.boundingRect[3]),(self.boundingRect[2],self.boundingRect[3]),(self.boundingRect[2],0)):
            boxSphericalPoints.append((toOrthographicSphereCoords(sunCenteredBoundingBoxCoords[0] + boxCorners[0], sunCenteredBoundingBoxCoords[1] + boxCorners[1], self.sunImage.radius)/math.pi)*180)
        return boxSphericalPoints
    
    def getOrthographicSphereAreaDegreesSqrd(self):
        boxSphericalPoints = self.getOrthographicSphereBoundingRectDegrees()
        left = (boxSphericalPoints[0][0] + boxSphericalPoints[1][0])/2
        right = (boxSphericalPoints[2][0] + boxSphericalPoints[3][0])/2

        bottom = (boxSphericalPoints[0][1] + boxSphericalPoints[3][1])/2
        top = (boxSphericalPoints[1][1] + boxSphericalPoints[2][1])/2
        return (right-left)*(top-bottom)

    def getOrthographicSphereCoordsDegrees(self):
        sunCenteredCoords = self.getSunCenteredCartestianCoords()
        return (toOrthographicSphereCoords(sunCenteredCoords[0], sunCenteredCoords[1], self.sunImage.radius)/math.pi)*180

def findSunspotUncertainty(sunspot):
    sunRadiusUncertainty = 3
    sunCenterUncertainty = 2
    sunspotPositionUncertainty = 1
    sunspotTimeUncertainty = 30
    sunspotCopy = copy.deepcopy(sunspot)
    for possibleVariance in itertools.product([-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1]):
        sunspotCopy.sunImage.radius = sunspot.sunImage.radius + possibleVariance[0]*sunRadiusUncertainty
        sunspotCopy.sunImage.centerCoords[0] = sunspot.sunImage.centerCoords[0] + possibleVariance[1]*sunCenterUncertainty
        sunspotCopy.sunImage.centerCoords[1] = sunspot.sunImage.centerCoords[1] + possibleVariance[2]*sunCenterUncertainty
        sunspotCopy.cartesianCoords[0] = sunspot.cartesianCoords[0] + possibleVariance[2]*sunspotPositionUncertainty
        sunspotCopy.cartesianCoords[1] = sunspot.cartesianCoords[1] + possibleVariance[3]*sunspotPositionUncertainty
        #sunspotCopy.time = 
        print(sunspotCopy.getOrthographicSphereCoordsDegrees())

    
    
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
        remainingPairs = list(filter(lambda x: not np.array_equal(x[0].cartesianCoords, minPair[0].cartesianCoords), remainingPairs))
        remainingPairs = list(filter(lambda x: not np.array_equal(x[1].cartesianCoords, minPair[1].cartesianCoords), remainingPairs))
        filteredPairs.append(minPair)
    return filteredPairs

def matchSpotsBetweenFrames(previousSunSpots, newSunSpots):
    if not previousSunSpots or not newSunSpots:
        return []
    allPairedCenters = []
    for oc in previousSunSpots:
        for nc in newSunSpots:
            movementVector = nc.cartesianCoords-oc.cartesianCoords
            movementDistance = np.linalg.norm(movementVector)
            if movementVector[0] > 0 and movementDistance < 70:
                allPairedCenters.append((oc,nc,movementDistance))
    
    return filterPossibleMatches(allPairedCenters)

def findSunSpots(sun, image, blurred, imagefilename, imagetime):
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

    sunSpots = []
    for (i, c) in enumerate(cntrs):
        (x, y, w, h) = cv2.boundingRect(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        sunSpots.append(SunSpot(sun, np.array((cX,cY), dtype=np.int32), radius, (x, y, w, h), imagefilename, imagetime))
        cv2.circle(image, (int(cX), int(cY)), int(radius),
            (0, 0, 255), 3)
        cv2.putText(image, "#{}".format(i + 1), (x, y - 15),
            cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 0, 255), 2)
    cv2.circle(image, sun.centerCoords, sun.radius, (0,255,0), 2)
    return sunSpots,image

imageFilenames = glob.glob("{imagePath}/*.jpg".format(imagePath=args.images)) + glob.glob("{imagePath}/*.png".format(imagePath=args.images))
imageFilenames.sort()

previousSunSpots = None
lastRecordTime = None
diffT = None
samples = pd.DataFrame(columns=['latitude','period'])
activeVectorChains = []
for i in imageFilenames:
    timeString = re.search(r'[\d]{8}_[\d]{4}', i).group()
    recordTime = datetime.strptime(timeString, '%Y%m%d_%H%M')
    print(str(recordTime)+":")
    if lastRecordTime:
        diffT = recordTime - lastRecordTime
        print("DT = {DT}\n".format(DT=diffT))
    lastRecordTime = recordTime
    # load image, convert to grayscale, and blur it
    image = cv2.imread(i)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = ~gray
    
    #gray[gray > 250] = 0
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    #blurred = gray
    sun = SunImage(blurred)
    print("Sun radius in pixels = {sunRadius}".format(sunRadius=sun.radius))
    newSunSpots,image = findSunSpots(sun, image, blurred,i,recordTime)
    timePairedCenters = matchSpotsBetweenFrames(previousSunSpots, newSunSpots)
    print(timePairedCenters)
    for tpc in timePairedCenters:
        tpcChained = False
        for chain in activeVectorChains:
            if chain['sunspots'][-1].cartesianCoords[0] == tpc[0].cartesianCoords[0] and chain['sunspots'][-1].cartesianCoords[1] == tpc[0].cartesianCoords[1]:
                chain['sunspots'].append(tpc[1])
                chain['timeElapsed'] += diffT.total_seconds()
                
                tpcChained = True
        if not tpcChained:
            activeVectorChains.append({'sunspots': [tpc[0], tpc[1]],'timeElapsed' : diffT.total_seconds()})

    for tpc in timePairedCenters:
        cv2.arrowedLine(image, tuple(tpc[0].cartesianCoords),tuple(tpc[1].cartesianCoords),(0, 255, 0),2)
        tpc = (tpc[0].cartesianCoords - np.array(sun.centerCoords),tpc[1].cartesianCoords - np.array(sun.centerCoords))
        diffX = tpc[1] - tpc[0]
        print("X = ({X1},{X2})".format(X1=tpc[0][0],X2=tpc[0][1]))
        print("DX = ({DX1},{DX2})\n".format(DX1=diffX[0],DX2=diffX[1]))
        l0 = (toOrthographicSphereCoords(tpc[0][0],tpc[0][1], sun.radius)/math.pi)*180
        l1 = (toOrthographicSphereCoords(tpc[1][0],tpc[1][1], sun.radius)/math.pi)*180
        print("L = ({L1_0},{L1_1})".format(L1_0=l1[0],L1_1=l1[1]))
        diffL = l1 - l0
        print("DL = ({DL1},{DL2})".format(DL1=diffL[0],DL2=diffL[1]))
        print(diffT)
        if diffT:
            w = diffL / diffT.total_seconds()
            periodOfRotation = (360./(w[0]*86400), 360./(w[1]*86400))
            print("DL / DT = ({w1},{w2})\n".format(w1=w[0], w2=w[1]))
            print("P = ({P1},{P2})\n".format(P1=periodOfRotation[0], P2=periodOfRotation[1]))
            
            """
            if periodOfRotation[0] > samples['period'].quantile(0.75) or periodOfRotation[0] < samples['period'].quantile(0.25):
                cv2.imshow("Image", image)
                cv2.waitKey(0)"""

    previousSunSpots = newSunSpots
    if args.display:
        cv2.imshow("Image", image)
        cv2.waitKey(500)
    print("-"*30)



for chain in activeVectorChains:
    if chain['timeElapsed'] > 3600*60:
        print(len(chain['sunspots']),chain['timeElapsed'])
        firstSunspotSphericalCoords = chain['sunspots'][0].getOrthographicSphereCoordsDegrees()
        secondSunspotSphericalCoords = chain['sunspots'][-1].getOrthographicSphereCoordsDegrees()
        print("Area: " + str(chain['sunspots'][-1].getOrthographicSphereAreaDegreesSqrd()))
        print("Area: " + str(chain['sunspots'][0].getOrthographicSphereAreaDegreesSqrd()))
        diffL = secondSunspotSphericalCoords - firstSunspotSphericalCoords
        w = diffL / chain['timeElapsed']
        periodOfRotation = 360./(w[0]*86400)
        print(periodOfRotation)
        samples = samples.append({'latitude': (firstSunspotSphericalCoords[1] + secondSunspotSphericalCoords[1])/2.,'period': periodOfRotation}, ignore_index=True)
        print("Uncertainties: ")
        findSunspotUncertainty(chain['sunspots'][0])
        print("-"*20)
#plt.hexbin(latitudeSamples,periodSamples,gridsize=(15,150))
plt.scatter(samples['latitude'],samples['period'],s=4)
samples['bucket'] = pd.cut(samples['latitude'],np.arange(-60,60,5))
print(samples.groupby(samples['bucket'])['period'].median())
plt.ylim(10,50)
plt.show()