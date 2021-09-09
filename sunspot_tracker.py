#External modules
import imutils
import cv2
from imutils import contours
from skimage import measure
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#Standard python modules
import argparse, glob, time, re, itertools, math, copy
from datetime import datetime


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="Path to the image folder")
ap.add_argument('-d', "--display", help='Display image frames and sunspot information during processing', action='store_true')

args = ap.parse_args()

#Round number to n decimal places
def round_to_n(x, n):
    format = "%." + str(n-1) + "e"
    as_string = format % x
    return float(as_string)

#A helper function to convert from coordinates in an image of a sphere to coordinates on that sphere. 
def toSphereCoordsFromOrthographic(x,y, R):
    p = np.sqrt(x**2 + y**2)
    c = np.arcsin(p/R)
    lat0 = 0
    lon0 = 0
    lat = np.arcsin(np.cos(c)*lat0 + (y*np.sin(c)*np.cos(lat0))/p )
    lon = lon0 + np.arctan2(x*np.sin(c),p*np.cos(c)*np.cos(lat0) - y*np.sin(c)*np.sin(lat0))
    return np.array((lon,lat))

#Container class for finding the size and position of the sun in an image.
class SunImage:
    def __init__(self,image):
        self.image = image
        self.radius = -1
        self.centerCoords = None
        self.findSun(image)
    
    #Find a circle which matches the disc of the sun.
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

#Container class for sunspots and their various statistics.
#Has methods for both pixel coordinates in the image and conversions the spherical coordinates of the sun.
class SunSpot:
    def __init__(self, sunImage, cartesianCenterCoords, radius, boundingRect, filename, time):
        self.sunImage = sunImage
        self.cartesianCoords = cartesianCenterCoords
        self.filename = filename
        self.time = time
        self.boundingRect = boundingRect

    #Get the flat image coordinates sun centered by simply subtracting the found solar disc center
    # from the found center of the sunspot.
    def getSunCenteredCartestianCoords(self):
        return self.cartesianCoords - np.array(self.sunImage.centerCoords)

    #Get a bounding rectangle which contains the sunspot in sun centered coordinates.
    def getSunCenteredBoundingRectCoords(self):
        return np.array((self.boundingRect[0],self.boundingRect[1])) - np.array(self.sunImage.centerCoords)

    #Get a bounding rectangle which contains the sunspot in spherical coordinates.
    def getOrthographicSphereBoundingRectDegrees(self):   
        sunCenteredBoundingBoxCoords = self.getSunCenteredCartestianCoords()
        boxSphericalPoints = []
        for boxCorners in ((0,0),(0, self.boundingRect[3]),(self.boundingRect[2],self.boundingRect[3]),(self.boundingRect[2],0)):
            boxSphericalPoints.append((toSphereCoordsFromOrthographic(sunCenteredBoundingBoxCoords[0] + boxCorners[0], sunCenteredBoundingBoxCoords[1] + boxCorners[1], self.sunImage.radius)/math.pi)*180)
        return boxSphericalPoints
    
    #The area of the bounding rectangle in spherical coordinates.
    def getOrthographicSphereAreaDegreesSqrd(self):
        boxSphericalPoints = self.getOrthographicSphereBoundingRectDegrees()
        left = (boxSphericalPoints[0][0] + boxSphericalPoints[1][0])/2
        right = (boxSphericalPoints[2][0] + boxSphericalPoints[3][0])/2

        bottom = (boxSphericalPoints[0][1] + boxSphericalPoints[3][1])/2
        top = (boxSphericalPoints[1][1] + boxSphericalPoints[2][1])/2
        return (right-left)*(top-bottom)

    #Get the center of the sunspot in spherical coordinates.
    def getOrthographicSphereCoordsDegrees(self):
        sunCenteredCoords = self.getSunCenteredCartestianCoords()
        return (toSphereCoordsFromOrthographic(sunCenteredCoords[0], sunCenteredCoords[1], self.sunImage.radius)/math.pi)*180

def findSunspotUncertainty(sunspot):
    sunRadiusUncertainty = 3
    sunCenterUncertainty = 2
    sunspotPositionUncertainty = 1
    sunspotCopy = copy.deepcopy(sunspot)
    minPositionDegrees = [1000,1000]
    maxPositionDegrees = [-1000,-1000]
    for possibleVariance in itertools.product([-1,1],[-1,1],[-1,1],[-1,1],[-1,1],[-1,1]):
        sunspotCopy.sunImage.radius = sunspot.sunImage.radius + possibleVariance[0]*sunRadiusUncertainty
        sunspotCopy.sunImage.centerCoords = (sunspot.sunImage.centerCoords[0] + possibleVariance[1]*sunCenterUncertainty,
                                            sunspot.sunImage.centerCoords[1] + possibleVariance[2]*sunCenterUncertainty)
        sunspotCopy.cartesianCoords = (sunspot.cartesianCoords[0] + possibleVariance[3]*sunspotPositionUncertainty,
                                    sunspot.cartesianCoords[1] + possibleVariance[4]*sunspotPositionUncertainty) 
        possiblePositionDegrees = sunspotCopy.getOrthographicSphereCoordsDegrees()
        minPositionDegrees[0] = min(minPositionDegrees[0], possiblePositionDegrees[0])
        minPositionDegrees[1] = min(minPositionDegrees[1], possiblePositionDegrees[1])
        maxPositionDegrees[0] = max(maxPositionDegrees[0], possiblePositionDegrees[0])
        maxPositionDegrees[1] = max(maxPositionDegrees[1], possiblePositionDegrees[1])
    
    midPositionDegrees = sunspotCopy.getOrthographicSphereCoordsDegrees()
    uncertaintyPercentFromLower = abs((np.array(midPositionDegrees) - np.array(minPositionDegrees))/midPositionDegrees)
    uncertaintyPercentFromUpper = abs((np.array(maxPositionDegrees) - np.array(midPositionDegrees))/midPositionDegrees)
    return(max(uncertaintyPercentFromLower[0],uncertaintyPercentFromUpper[0]),max(uncertaintyPercentFromLower[1],uncertaintyPercentFromUpper[1]))

def displayImageAndSunSpotInfo(sun, timePairedCenters):
    for tpc in timePairedCenters:
        cv2.arrowedLine(image, tuple(tpc[0].cartesianCoords),tuple(tpc[1].cartesianCoords),(0, 255, 0),2)
        tpc = (tpc[0].cartesianCoords - np.array(sun.centerCoords),tpc[1].cartesianCoords - np.array(sun.centerCoords))
        diffX = tpc[1] - tpc[0]
        print("X = ({X1},{X2})".format(X1=tpc[0][0],X2=tpc[0][1]))
        print("DX = ({DX1},{DX2})\n".format(DX1=diffX[0],DX2=diffX[1]))
        l0 = (toSphereCoordsFromOrthographic(tpc[0][0],tpc[0][1], sun.radius)/math.pi)*180
        l1 = (toSphereCoordsFromOrthographic(tpc[1][0],tpc[1][1], sun.radius)/math.pi)*180
        print("L = ({L1_0},{L1_1})".format(L1_0=l1[0],L1_1=l1[1]))
        diffL = l1 - l0
        print("DL = ({DL1},{DL2})".format(DL1=diffL[0],DL2=diffL[1]))
        print(diffT)
        if diffT:
            w = diffL / diffT.total_seconds()
            periodOfRotation = (360./(w[0]*86400), 360./(w[1]*86400))
            print("DL / DT = ({w1},{w2})\n".format(w1=w[0], w2=w[1]))
            print("P = ({P1},{P2})\n".format(P1=periodOfRotation[0], P2=periodOfRotation[1]))
    
    cv2.imshow("Image", image)
    cv2.waitKey(500)
    print("-"*30)

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

    labels = measure.label(thresh, background=0)
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

def generateSunspotRotationRecord(firstSunspot, secondSunspot):
    firstSunspotMeasurementSphericalCoords = firstSunspot.getOrthographicSphereCoordsDegrees()
    firstSunspotMeasurementUncertainty = findSunspotUncertainty(firstSunspot)

    secondSunspotMeasurementSphericalCoords = secondSunspot.getOrthographicSphereCoordsDegrees()
    secondSunspotMeasurementUncertainty = findSunspotUncertainty(secondSunspot)

    averageLat = (secondSunspotMeasurementSphericalCoords[1] + firstSunspotMeasurementSphericalCoords[1])/2
    averageLatUncert = secondSunspotMeasurementSphericalCoords[1] * secondSunspotMeasurementUncertainty[1] + \
                       firstSunspotMeasurementSphericalCoords[1] * firstSunspotMeasurementUncertainty[1]

    averageLatitudeString = "{lat}{dir}±{uncert}".format(lat=round_to_n(abs(averageLat),3),
                dir = 'N' if averageLat < 0 else 'S',
                uncert=round_to_n(averageLatUncert,2))
    

    diffL = abs(secondSunspotMeasurementSphericalCoords[0] - firstSunspotMeasurementSphericalCoords[0])
    diffLUncert = abs(secondSunspotMeasurementSphericalCoords[0] * secondSunspotMeasurementUncertainty[0] + \
                  firstSunspotMeasurementSphericalCoords[0] * firstSunspotMeasurementUncertainty[0])

    diffLString = "{long}±{uncert}".format(long=round_to_n(abs(diffL),3),
                uncert=round_to_n(diffLUncert,2))

    diffT = (secondSunspot.time - firstSunspot.time).total_seconds()
    diffTUncert = 1800 + 1800

    diffTString = "{timediff}±{uncert}".format(timediff=round_to_n(diffT/3600.,3),
            uncert=round_to_n(diffTUncert/3600.,2))

    w = diffL / diffT
    wUncert = abs(w*( diffLUncert/diffL + diffTUncert/diffT ))

    wString = "{w}±{uncert}".format(w=round_to_n(w*3600.,3),
            uncert=round_to_n(wUncert*3600.,2))

    periodOfRotation = 360./(w*86400)
    periodOfRotationUncert = abs(periodOfRotation*(wUncert/w))

    periodString = "{period}±{uncert}".format(period=round_to_n(periodOfRotation,3),
        uncert=round_to_n(periodOfRotationUncert,2))

    sunspotRotationRecord = {'Average Latitude' : averageLatitudeString,'Longitude Difference' : diffLString, 'Time Difference (h)' : diffTString,
                             'Rotational Speed (degrees/h)' : wString, 'Rotational Period (d)' : periodString }
    return sunspotRotationRecord
    

def generateSunspotRecord(sunspot):
    sunspotMeasurementSphericalCoords = sunspot.getOrthographicSphereCoordsDegrees()
    sunspotMeasurementArea = sunspot.getOrthographicSphereAreaDegreesSqrd()
    sunspotMeasurementUncertainty = findSunspotUncertainty(sunspot)

    sunspotLongitudeString = "{long}{dir}±{uncert}".format(long=round_to_n(abs(sunspotMeasurementSphericalCoords[0]),3),
                            dir = 'E' if sunspotMeasurementSphericalCoords[0] > 0 else 'W',
                            uncert=round_to_n(sunspotMeasurementUncertainty[0]*abs(sunspotMeasurementSphericalCoords[0]),2))

    sunspotLatitudeString = "{lat}{dir}±{uncert}".format(lat=round_to_n(abs(sunspotMeasurementSphericalCoords[1]),3),
                    dir = 'N' if sunspotMeasurementSphericalCoords[1] < 0 else 'S',
                    uncert=round_to_n(sunspotMeasurementUncertainty[1]*abs(sunspotMeasurementSphericalCoords[1]),2))

    sunspotAreaString = "{area}±0".format(area=round_to_n(sunspotMeasurementArea,3))

    sunspotRecord = {
        'Image Filename' : sunspot.filename,
        'Sunspot Longitude' : sunspotLongitudeString, 'Sunspot Latitude' : sunspotLatitudeString,
        'Sunspot Area' : sunspotAreaString, 'Date & Time' : sunspot.time
    }
    return sunspotRecord

#Get all image files in the 
imageFilenames = glob.glob("{imagePath}/*.jpg".format(imagePath=args.images)) + glob.glob("{imagePath}/*.png".format(imagePath=args.images))
imageFilenames.sort()

previousSunSpots = None
lastRecordTime = None
diffT = None
samples = pd.DataFrame(columns=['latitude','period'])
activeVectorChains = []
for i in imageFilenames:

    #Get the time and date the image was recorded 
    timeString = re.search(r'[\d]{8}_[\d]{4}', i).group()
    recordTime = datetime.strptime(timeString, '%Y%m%d_%H%M')
    print(str(recordTime)+":")
    print(i)
    if lastRecordTime:
        diffT = recordTime - lastRecordTime
        print("DT = {DT}\n".format(DT=diffT))
    lastRecordTime = recordTime

    # load image, convert to grayscale, and blur it
    image = cv2.imread(i)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = ~gray
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    #Find the radius and center of the solar disc in the image
    sun = SunImage(blurred)

    #Find sunspots in the image
    newSunSpots,image = findSunSpots(sun, image, blurred,i,recordTime)

    #Find the matching sunspot images between this frame and previous frame.
    timePairedCenters = matchSpotsBetweenFrames(previousSunSpots, newSunSpots)
    previousSunSpots = newSunSpots

    for tpc in timePairedCenters:
        tpcChained = False
        for chain in activeVectorChains:
            if chain['sunspots'][-1].cartesianCoords[0] == tpc[0].cartesianCoords[0] and \
             chain['sunspots'][-1].cartesianCoords[1] == tpc[0].cartesianCoords[1]:

                chain['sunspots'].append(tpc[1])
                chain['timeElapsed'] += diffT.total_seconds()
                
                tpcChained = True

        if not tpcChained:
            activeVectorChains.append({'sunspots': [tpc[0], tpc[1]],'timeElapsed' : diffT.total_seconds()})

    if args.display:
        displayImageAndSunSpotInfo(sun, timePairedCenters)

sunspotChainCount = 0

firstSunspotMeasurements = pd.DataFrame(columns=['Image Filename','Sunspot Longitude', 'Sunspot Latitude',
                                                 'Sunspot Area', 'Date & Time' ])
secondSunspotMeasurements = pd.DataFrame(columns=['Image Filename','Sunspot Longitude', 'Sunspot Latitude',
                                                 'Sunspot Area', 'Date & Time' ])
sunspotRotationTable = pd.DataFrame(columns=['Average Latitude','Longitude Difference', 'Time Difference (h)',
                                                 'Rotational Speed (degrees/h)', 'Rotational Period (d)' ])

for chain in activeVectorChains:
    #if chain['timeElapsed'] > 3600*60:
    #print(len(chain['sunspots']),chain['timeElapsed'])

    #Uncertainty at the edges of the sun is higher so lets move in until we get below a threshold of uncertainty.
    beginningSunspot = None
    endSunspot = None
    countOfSunspotBeginning = 0
    for ss in chain['sunspots']:
        #Filter out reading with too much uncertainty
        if abs(findSunspotUncertainty(ss)[0]*ss.getOrthographicSphereCoordsDegrees()[0]) < 3.0:
            beginningSunspot = ss
            break
        countOfSunspotBeginning+=1
    if not beginningSunspot:
        continue
    for ss in chain['sunspots'][countOfSunspotBeginning-1::-1]:
        #Filter out reading with too much uncertainty
        if abs(findSunspotUncertainty(ss)[0]*ss.getOrthographicSphereCoordsDegrees()[0]) < 3.0:
            endSunspot = ss
            break
    if not endSunspot:
        continue
    #Let's only look at longer timeframes (Observed for more than 60 hours)
    if (endSunspot.time - beginningSunspot.time).total_seconds() < 60*60*60:
        continue
        
    firstSunspotMeasurements = firstSunspotMeasurements.append(generateSunspotRecord(beginningSunspot),ignore_index=True)

    secondSunspotMeasurements = secondSunspotMeasurements.append(generateSunspotRecord(endSunspot),ignore_index=True)

    sunspotRotationTable = sunspotRotationTable.append(generateSunspotRotationRecord(beginningSunspot, endSunspot),ignore_index=True)


print("First Sunspot Measurements")
print(firstSunspotMeasurements)
firstSunspotMeasurements.to_csv("firstSunspotMeasurements.csv")
print()
print("Second Sunspot Measurements")
print(secondSunspotMeasurements)
secondSunspotMeasurements.to_csv("secondSunspotMeasurements.csv")
print()
print("Sun Rotation Data")
print(sunspotRotationTable)
sunspotRotationTable.to_csv("sunspotRotationTable.csv")
