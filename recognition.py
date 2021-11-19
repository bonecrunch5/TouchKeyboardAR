import cv2
import sys
import numpy as np
import json
import logging
import os
from dotenv import load_dotenv
load_dotenv()

prepImgPath = './generated/imgKeyboard.jpg'
prepKeysPath = './generated/keys.json'

# Load arguments
if len(sys.argv) > 1:
    prepImgPath = sys.argv[1]

    if len(sys.argv) > 2:
        prepKeysPath = sys.argv[2]

prepImg = None
prepKeys = {}

# Load preparation image
prepImg = cv2.imread(prepImgPath)

if(prepImg is None):
    print("Couldn't load image")
    exit()

# Load keys data
try:
    fjson = open(prepKeysPath, 'r')
    prepKeys = json.load(fjson)
except (IOError, json.decoder.JSONDecodeError) as ex:
    logging.exception("Couldn't open keys file")
    exit()

# Start video capture
cap = cv2.VideoCapture(int(os.environ.get('CAMERA_ID', '0')))

if not (cap.isOpened()):
    print('Could not open video device')
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Set the resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Convert from RGB to grayscale
prepImgGrayscale = cv2.cvtColor(prepImg, cv2.COLOR_BGR2GRAY)

# Based on:
# - https://docs.opencv.org/2.4/doc/tutorials/features2d/feature_homography/feature_homography.html#feature-homography
# - https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html

# Initiate SIFT detector
sift = cv2.SIFT_create()

# Detect the keypoints and compute descriptors
prepKeypoints, prepDescriptors = sift.detectAndCompute(prepImg,None)

# Initialize the Matcher for matching the keypoints
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

matcher = cv2.FlannBasedMatcher(index_params, search_params)

bgSubtractor = None
cropImgHist = None

def getContours(image):
    # Convert from RGB to grayscale
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Smoothing
    imgSmooth = cv2.blur(imgGray, (5, 5))

    # Threshold
    (_, imgThresh) = cv2.threshold(imgSmooth, 100, 255,
                                   cv2.THRESH_BINARY)  # TODO adjust values of threshold

    # Find Countours
    imgCanny = cv2.Canny(imgThresh, 100, 200)

    contours, _ = cv2.findContours(
        imgCanny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # cv2.imshow('img gray',imgGray)
    # cv2.imshow('img smooth',imgSmooth)
    # cv2.imshow('img threshold',imgThresh)
    # cv2.imshow('img canny', imgCanny)

    return contours

while(True):
    # Capture frame-by-frame
    _, captImg = cap.read()

    captImgSquare = captImg.copy()

    captHeight, captWidth, _ = captImg.shape
    squareSide = 80
    
    captHeightHalf = int(captHeight / 2)
    captWidthHalf = int(captWidth / 2)
    squareSideHalf = int(squareSide / 2)

    topLeft = (captWidthHalf - squareSideHalf, captHeightHalf - squareSideHalf)
    bottomLeft = (captWidthHalf - squareSideHalf, captHeightHalf + squareSideHalf)
    topRight = (captWidthHalf + squareSideHalf, captHeightHalf - squareSideHalf)
    bottomRight = (captWidthHalf + squareSideHalf, captHeightHalf + squareSideHalf)


    cv2.line(captImgSquare,
            topLeft,
            bottomLeft,
            (0, 255, 0), 2)

    cv2.line(captImgSquare,
            bottomLeft,
            bottomRight,
            (0, 255, 0), 2)

    cv2.line(captImgSquare,
            bottomRight,
            topRight,
            (0, 255, 0), 2)
    
    cv2.line(captImgSquare,
            topRight,
            topLeft,
            (0, 255, 0), 2)

    # Convert from RGB to grayscale
    captImgGrayscale = cv2.cvtColor(captImg, cv2.COLOR_BGR2GRAY)
    
    # Detect the keypoints and compute descriptors
    captKeypoints, captDescriptors = sift.detectAndCompute(captImg,None)

    # Show the camera image
    cv2.imshow('img', captImgSquare)

    # Match keypoints
    if (captDescriptors is not None and len(captDescriptors) > 1):
        matches = matcher.knnMatch(prepDescriptors, captDescriptors,k=2)
    else:
        continue

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # Show only good matches
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]

    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = 0)

    matchesImg = cv2.drawMatchesKnn(prepImg, prepKeypoints, captImg, captKeypoints, matches, None, **draw_params)
    
    # Localize the object
    prepPoints = []
    captPoints = []

    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            prepPoints.append(prepKeypoints[m.queryIdx].pt)
            captPoints.append(captKeypoints[m.trainIdx].pt)

    if len(prepPoints) >= 4 and len(captPoints) >= 4:
        homographyMatrix_Box, _ = cv2.findHomography(np.array(prepPoints), np.array(captPoints), method=cv2.RANSAC)
        homographyMatrix_Warp, _ = cv2.findHomography(np.array(captPoints), np.array(prepPoints), method=cv2.RANSAC)

        prepImgRows, prepImgCols, _ = prepImg.shape

        if(homographyMatrix_Box is not None):
            prepCorners = [(float(0), float(0)), (float(prepImgCols), float(0)), (float(prepImgCols), float(prepImgRows)), (float(0), float(prepImgRows))]
            captCorners = cv2.perspectiveTransform(np.array(prepCorners)[None, :, :], homographyMatrix_Box)[0]

            cv2.line(matchesImg, (int(captCorners[0][0] + prepImgCols), int(captCorners[0][1])), (int(captCorners[1][0] + prepImgCols), int(captCorners[1][1])), (0, 255, 0), 4)
            cv2.line(matchesImg, (int(captCorners[1][0] + prepImgCols), int(captCorners[1][1])), (int(captCorners[2][0] + prepImgCols), int(captCorners[2][1])), (0, 255, 0), 4)
            cv2.line(matchesImg, (int(captCorners[2][0] + prepImgCols), int(captCorners[2][1])), (int(captCorners[3][0] + prepImgCols), int(captCorners[3][1])), (0, 255, 0), 4)
            cv2.line(matchesImg, (int(captCorners[3][0] + prepImgCols), int(captCorners[3][1])), (int(captCorners[0][0] + prepImgCols), int(captCorners[0][1])), (0, 255, 0), 4)
            
            # Show the images with matches
            cv2.imshow("Matches", matchesImg)

        if(homographyMatrix_Warp is not None):
            imgKeyboard = cv2.warpPerspective(captImg, homographyMatrix_Warp, (prepImgCols, prepImgRows))

            imgKeyboardCopy = imgKeyboard.copy()

            histMask = None
            bgSubMask = None

            if bgSubtractor is not None:
                fgmask = bgSubtractor.apply(imgKeyboardCopy, learningRate=0)    

                kernel = np.ones((4, 4), np.uint8)
                
                # The effect is to remove the noise in the background
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)
                # To close the holes in the objects
                fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel, iterations=2)
                
                # Apply the mask on the frame and return
                histMask = cv2.bitwise_and(imgKeyboardCopy, imgKeyboardCopy, mask=fgmask)

            if cropImgHist is not None:
                captImgTargetHsv = cv2.cvtColor(imgKeyboardCopy,cv2.COLOR_BGR2HSV)

                # apply backprojection
                dst = cv2.calcBackProject([captImgTargetHsv],[0,1],cropImgHist,[0,180,0,256],1)

                # Now convolute with circular disc
                disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
                cv2.filter2D(dst,-1,disc,dst)

                # threshold and binary AND
                ret,thresh = cv2.threshold(dst,50,255,0)
                thresh = cv2.merge((thresh,thresh,thresh))
                bgSubMask = cv2.bitwise_and(imgKeyboardCopy,thresh)

            if histMask is not None and bgSubMask is not None:
                res = cv2.bitwise_and(histMask, bgSubMask)
                kernel = np.ones((10,10),np.uint8)
                res = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)
                res = cv2.erode(res,kernel,iterations = 1)
                cv2.imshow("res", res)

                fingerContours = getContours(res)

                maxPerimeter = 0
                biggestContour = None
                
                for contour in fingerContours:
                    perimeter = cv2.arcLength(contour, False)

                    if perimeter > maxPerimeter:
                        maxPerimeter = perimeter
                        biggestContour = contour

                imgKeyboardHeight, imgKeyboardWidth, _ = imgKeyboard.shape

                if biggestContour is not None:
                    approxPolygon = cv2.approxPolyDP(biggestContour, 0.001 * maxPerimeter, True)

                    maxDistance = 0
                    furthestPoint = None

                    proximityFactor = 0.01
                    # TODO: use int instead of bool to know how many points originate from each side
                    originTop = False
                    originBot = False
                    originLeft = False
                    originRight = False

                    for point in approxPolygon:
                        x, y = point[0]

                        if not originTop and y <= imgKeyboardHeight * proximityFactor:
                            originTop = True

                        if not originBot and y >= imgKeyboardHeight * (1 - proximityFactor):
                            originBot = True

                        if not originLeft and x <= imgKeyboardWidth * proximityFactor:
                            originLeft = True

                        if not originRight and x >= imgKeyboardWidth * (1 - proximityFactor):
                            originRight = True

                    for point in approxPolygon:
                        x, y = point[0]

                        distTop = y
                        distBot = imgKeyboardHeight - y
                        distLeft = x
                        distRight = imgKeyboardWidth - x

                        avgDist = 0
                        sides = 0

                        if originTop:
                            avgDist += distTop
                            sides += 1

                        if originBot:
                            avgDist += distBot
                            sides += 1

                        if originLeft:
                            avgDist += distLeft
                            sides += 1

                        if originRight:
                            avgDist += distRight
                            sides += 1

                        if sides > 0:
                            avgDist /= sides

                        if avgDist > maxDistance:
                            furthestPoint = point
                            maxDistance = avgDist

                        cv2.circle(imgKeyboardCopy, (x, y), 3, (0, 0, 255), -1)

                    if furthestPoint is not None:
                        x, y = furthestPoint[0]
                        cv2.circle(imgKeyboardCopy, (x, y), 5, (255, 0, 0), -1)

                    hull = cv2.convexHull(biggestContour)

                    # cv2.drawContours(imgKeyboardCopy, [hull], -1, (0, 255, 0))

                    cv2.imshow('imgKeyboardCopy', imgKeyboardCopy)

            for key in prepKeys['keys']:
                if(os.environ.get('SHOW_KEY_CORNERS').upper() == 'TRUE'):
                    for point in key['points']:
                        cv2.circle(imgKeyboard, (point['x'], point['y']), 2, (0, 255, 0), -1)

                if(os.environ.get('SHOW_KEY_EDGES').upper() == 'TRUE'):
                    cv2.line(imgKeyboard, (key['points'][len(key['points']) - 1]['x'], key['points'][len(key['points']) - 1]['y']), (key['points'][0]['x'], key['points'][0]['y']), (0, 255, 0), 2)

                    for i in range(0, len(key['points']) - 1):
                        cv2.line(imgKeyboard, (key['points'][i]['x'], key['points'][i]['y']), (key['points'][i + 1]['x'], key['points'][i + 1]['y']), (0, 255, 0), 2)

                if(os.environ.get('SHOW_KEY_LABELS').upper() == 'TRUE'):
                    cv2.putText(imgKeyboard, key['symbol'], (key['points'][0]['x'], key['points'][0]['y']),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

            # Show image with only keyboard
            cv2.imshow("imgKeyboard", imgKeyboard)

    # Waits for a user input to quit the application
    k = cv2.waitKey(1) & 0xFF

    if k == ord('b'):
        bgSubtractor = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=30, detectShadows=False)
    elif k == ord('h'):
        cropImg = captImg[topLeft[1]:bottomLeft[1], topLeft[0]:topRight[0]]
        cropImgHsv = cv2.cvtColor(cropImg,cv2.COLOR_BGR2HSV)

        # calculating object histogram
        cropImgHist = cv2.calcHist([cropImgHsv],[0, 1], None, [12, 12], [0, 180, 0, 256] )
    elif k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
