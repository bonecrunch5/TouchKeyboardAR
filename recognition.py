import cv2
import sys
import time
import datetime
import numpy as np
import math
import json
import logging
import os
import opencv_utilities
from dotenv import load_dotenv
load_dotenv()

### DEBUG VARS ###

debugImagesEnv = os.environ.get('SHOW_DEBUG_IMAGES')
debugKeyCornersEnv = os.environ.get('SHOW_KEY_CORNERS')
debugKeyEdgesEnv = os.environ.get('SHOW_KEY_EDGES')
debugKeyLabelsEnv = os.environ.get('SHOW_KEY_LABELS')
debugImages = debugImagesEnv is not None and debugImagesEnv.upper() == 'TRUE'
debugKeyCorners = debugKeyCornersEnv is not None and debugKeyCornersEnv.upper() == 'TRUE'
debugKeyEdges = debugKeyEdgesEnv is not None and debugKeyEdgesEnv.upper() == 'TRUE'
debugKeyLabels = debugKeyLabelsEnv is not None and debugKeyLabelsEnv.upper() == 'TRUE'

### GLOBAL VARS ###

# Hand histogram scan square side
g_scanSquareSide = 80

# Input file paths
g_prepImgPath = './generated/imgKeyboard.jpg'
g_prepKeysPath = './generated/keys.json'
# Input file content
g_prepImg = None
g_prepKeys = None

# If the hand histogram has been obtained
g_gotHandHistrogram = False
# If a clear background frame has been obtain
g_gotClearBGFrame = False

# SIFT detector
g_sift = cv2.SIFT_create()
# Matcher for matching the keypoints
g_matcher = cv2.FlannBasedMatcher(dict(algorithm=0, trees = 5), dict(checks=50))
# Keypoints and descriptors of preparation image
g_prepKeypoints = None
g_prepDescriptors = None

# Background to subtract from image with hand
g_bgSubtractor = None
# Histogram of hand
g_cropImgHist = None

g_homographyMatrix = None

# Text input by user
g_writtenInput = ''
# Current key being pressed
g_pressedKey = None
# Last pressed key
g_lastPressedKey = None
# Last pressed key points in top view image
g_lastPressedKeyHomographyPoints = None
# When the pressed key feedback was first shown
g_showKeyStartTime = time.time()
# When the pressed key started being pressed
g_pressedKeyStartTime = time.time()
# Time the pressed key feedback will be shown
g_keyShowSeconds = 2
# Time necessary for a pressed key to be inputted
g_keyPressSeconds = 1

### ENV VARS ###

keyPressSecondsEnv = os.environ.get('KEY_PRESS_DURATION')

if keyPressSecondsEnv is not None:
    try:
        keyPressSecondsValue = float(keyPressSecondsEnv)

        if keyPressSecondsValue > 0 and keyPressSecondsValue < 20:
            g_keyPressSeconds = keyPressSecondsValue
        else:
            print('KEY_PRESS_DURATION variable has invalid value, defaulting to 1')
    except ValueError:
        print('Couldn\'t load KEY_PRESS_DURATION variable, defaulting to 1')

### FUNCTIONS ###

# Save user input to file
def saveOutput():
    if not os.path.exists("kb-output"):
        os.mkdir("kb-output")

    f = open('kb-output/' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%dT%H-%M-%SZ') + '.txt', 'w')
    f.write(g_writtenInput)
    f.close()

# Draw key outline on captured image
def drawKeyOutline(img):
    if g_lastPressedKeyHomographyPoints is None:
        return False

    if g_homographyMatrix is None:
        if debugImages:
            print('No homography matrix for the keyboard. The keyboard probably hasn\'t been detected, so can\'t show outline of key')
        return False

    # Calculate inversed homography to get points on captured image
    homographyMatrix_WarpInv = np.linalg.inv(g_homographyMatrix)
    lastPressedKeyPoints = cv2.perspectiveTransform(np.float32([g_lastPressedKeyHomographyPoints]), homographyMatrix_WarpInv)[0]

    # Draw outline
    cv2.line(img, (int(lastPressedKeyPoints[len(lastPressedKeyPoints) - 1][0]), int(lastPressedKeyPoints[len(lastPressedKeyPoints) - 1][1])),
                    (int(lastPressedKeyPoints[0][0]), int(lastPressedKeyPoints[0][1])), (0, 255, 0), 2)

    for i in range(0, len(lastPressedKeyPoints) - 1):
        cv2.line(img, (int(lastPressedKeyPoints[i][0]), int(lastPressedKeyPoints[i][1])), (int(lastPressedKeyPoints[i + 1][0]), int(lastPressedKeyPoints[i + 1][1])), (0, 255, 0), 2)

    return True

# Crop square from image and get histogram
def getImageHistogram(img):
    # Crop image
    topLeft, _, bottomRight, _ = opencv_utilities.getCenterSquareOfImg(img, g_scanSquareSide)
    cropImg = img[topLeft[1]:bottomRight[1], topLeft[0]:bottomRight[0]]
    cropImgHsv = cv2.cvtColor(cropImg,cv2.COLOR_BGR2HSV)

    # Calculating object histogram
    return cv2.calcHist([cropImgHsv],[0, 1], None, [12, 12], [0, 180, 0, 256] )

# Detect pressed key based on fingertip coordinates
def detectPressedKey(pressedPoint):
    global g_pressedKey, g_writtenInput, g_pressedKeyStartTime, g_lastPressedKey, g_lastPressedKeyHomographyPoints, g_showKeyStartTime

    x, y = pressedPoint[0]

    # Check if pressedPoint is over any key
    for key in g_prepKeys['keys']:            
        topLeftVertex = key['points'][0]
        botRightVertex = key['points'][2]
        if opencv_utilities.isInSquare({ 'x': x, 'y': y}, topLeftVertex, botRightVertex, 10):
            # If key was already being press, check if necessary time has passed
            if g_pressedKey == key['symbol']:
                if time.time() - g_pressedKeyStartTime >= g_keyPressSeconds:
                    # Necessary time has passed, to output symbol 
                    # and write to console and file (will be saved when program closes)
                    if g_pressedKey == 'ENTER':
                        print('\n', end='', flush=True)
                        g_writtenInput += '\n'
                    elif g_pressedKey == 'SPACE':
                        print(' ', end='', flush=True)
                        g_writtenInput += ' '
                    elif g_pressedKey == 'BACKSPACE':
                        # These 3 inputs delete the previous character in the console
                        # Can't delete new line inputs though
                        print('\b', end='', flush=True)
                        print(' ', end='', flush=True)
                        print('\b', end='', flush=True)
                        g_writtenInput = g_writtenInput[:-1]
                    else:
                        print(g_pressedKey, end='', flush=True)
                        g_writtenInput += g_pressedKey

                    # Play sound (only works on console terminals that allow it)
                    print('\a', end='', flush=True)

                    # Save pressed key to show on screen
                    # And start timer for that
                    g_lastPressedKey = g_pressedKey
                    g_showKeyStartTime = time.time()

                    # Get corners of key in top view image of keyboard
                    keyPointsArray = []

                    for point in key['points']:
                        keyPointsArray.append([point['x'], point['y']])

                    g_lastPressedKeyHomographyPoints = keyPointsArray

                    # Reset pressed key (no key is being pressed now)
                    g_pressedKey = None
            else:
                # Key wasn't being pressed, start counting time
                g_pressedKey = key['symbol']
                g_pressedKeyStartTime = time.time()

            break

# Get the finger tip coordinates
# TODO: the comments can probably be improved
def getFingerTip(fingerContour, img):
    imgKeyboardHeight, imgKeyboardWidth, _ = img.shape

    # Put contour with white fill over black background
    # And calculate its center of mass
    contourOverDark = np.zeros((imgKeyboardHeight, imgKeyboardWidth))
    cv2.fillPoly(contourOverDark, pts=[fingerContour], color=(255,255,255))

    if debugImages:
        cv2.imshow('Debug - Finger Contour Shape', contourOverDark)

    centerOfMass = opencv_utilities.getCenterOfMass(contourOverDark)
    
    # Get furthest point from rest of finger/hand
    maxDistance = 0
    furthestPoint = None
    proximityFactor = 0.01

    # Index order: top, bottom, left, right
    origin = [0] * 4

    # Check how many points originate from each edge of the image
    for point in fingerContour:
        x, y = point[0]

        if y <= imgKeyboardHeight * proximityFactor:
            origin[0] += 1

        if y >= imgKeyboardHeight * (1 - proximityFactor):
            origin[1] += 1

        if x <= imgKeyboardWidth * proximityFactor:
            origin[2] += 1

        if x >= imgKeyboardWidth * (1 - proximityFactor):
            origin[3] += 1

    # Now check each point of the contour
    for point in fingerContour:
        x, y = point[0]

        # Draw finger points for debug
        if debugImages:
            cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

        # Get the distance from the point to each edge
        # Index order: top, bottom, left, right
        dist = [0] * 4
        dist[0] = y
        dist[1] = imgKeyboardHeight - y
        dist[2] = x
        dist[3] = imgKeyboardWidth - x

        avgDist = 0
        sides = 0

        # Check which edge distances should be taken into account, based on the values of the origin array
        for i in range(0, 4):
            # If any edge has more than 1 point adjacent to it, consider it only
            if origin[i] > 1:
                avgDist += dist[i]
                sides += 1
                break

        # If no edge has more than 1 point adjacent to it,
        # take into account the distance to each edge that has a point adjacent to it
        if sides == 0:
            for i in range(0, 4):
                if origin[i] == 1:
                    avgDist += dist[i]
                    sides += 1

        # Average the distance
        if sides > 0:
            avgDist /= sides

        # If center of mass was calculated, also take it into account to get the average distance
        if centerOfMass is not None:
            avgDist = avgDist * 0.6 + math.sqrt(math.pow((centerOfMass[0] - x), 2) + math.pow((centerOfMass[1] - y), 2)) * 0.4

        # The point with the highest average distance will be considered the fingertip
        if avgDist > maxDistance:
            furthestPoint = point
            maxDistance = avgDist

    # Draw finger tip for debug and show
    if debugImages:
        if furthestPoint is not None:
            x, y = furthestPoint[0]
            cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow('Debug - Finger tip detection', img)

    return furthestPoint

# Get contour with area representing finger
def getFingerContour(img):
    bgSubMask = getBGSubMask(img.copy())
    histMask = getHistMask(img.copy())

    if histMask is None or bgSubMask is None:
        return None

    # Merge masks and apply on image
    imgMasked = cv2.bitwise_and(histMask, bgSubMask)

    if debugImages:
        cv2.imshow('Debug - Masked Finger', imgMasked)

    # Clean image
    kernel = np.ones((10, 10), np.uint8)
    imgMasked = cv2.morphologyEx(imgMasked, cv2.MORPH_CLOSE, kernel)
    imgMasked = cv2.erode(imgMasked, kernel, iterations = 1)

    fingerContours = opencv_utilities.getContours(imgMasked, debugImages=debugImages, namePrefix='Finger')

    # Find biggest contour (will be the finger contour)
    maxPerimeter = 0
    biggestContour = None
    
    for contour in fingerContours:
        perimeter = cv2.arcLength(contour, False)

        if perimeter > maxPerimeter:
            maxPerimeter = perimeter
            biggestContour = contour

    if biggestContour is None:
        return None

    # Get convex hull polygon to represent finger
    approxPolygon = cv2.approxPolyDP(biggestContour, 0.001 * maxPerimeter, True)
    return cv2.convexHull(approxPolygon)

# Get color histogram mask for image
def getHistMask(img):
    if g_cropImgHist is None:
        return None

    imgTargetHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # apply backprojection
    dst = cv2.calcBackProject([imgTargetHsv], [0, 1], g_cropImgHist, [0, 180, 0, 256], 1)

    # Now convolute with circular disc
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cv2.filter2D(dst, -1, disc, dst)

    # threshold and binary AND
    _, thresh = cv2.threshold(dst, 50, 255 ,0)
    thresh = cv2.merge((thresh, thresh, thresh))
    return cv2.bitwise_and(img, thresh)

# Get background subtraction mask for image
def getBGSubMask(img):
    if g_bgSubtractor is None:
        return None

    fgMask = g_bgSubtractor.apply(img, learningRate=0)    

    kernel = np.ones((4, 4), np.uint8)
    
    # The effect is to remove the noise in the background
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel, iterations=2)
    # To close the holes in the objects
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Apply the mask on the frame and return
    return cv2.bitwise_and(img, img, mask=fgMask)

# Match the keyboard and process its homography to get top view image
def matchAndProcessHomography(img):
    # Detect the keypoints and compute descriptors of captured image
    captKeypoints, captDescriptors = g_sift.detectAndCompute(img, None)

    # Match keypoints
    if (captDescriptors is not None and len(captDescriptors) > 1):
        matches = g_matcher.knnMatch(g_prepDescriptors, captDescriptors, k=2)
    else:
        return None
    
    # Localize the object
    prepPoints = []
    captPoints = []

    # Select good matches
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            prepPoints.append(g_prepKeypoints[m.queryIdx].pt)
            captPoints.append(captKeypoints[m.trainIdx].pt)

    if len(prepPoints) >= 4 and len(captPoints) >= 4:
        global g_homographyMatrix

        # Get homography for top view image
        g_homographyMatrix, _ = cv2.findHomography(np.array(captPoints), np.array(prepPoints), method=cv2.RANSAC)

        prepImgRows, prepImgCols, _ = g_prepImg.shape

        if debugImages:
            # Get homography to draw outline in original image
            homographyMatrixDebug, _ = cv2.findHomography(np.array(prepPoints), np.array(captPoints), method=cv2.RANSAC)

            if(homographyMatrixDebug is not None):
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

                matchesImg = cv2.drawMatchesKnn(g_prepImg, g_prepKeypoints, img, captKeypoints, matches, None, **draw_params)

                prepCorners = [(float(0), float(0)), (float(prepImgCols), float(0)), (float(prepImgCols), float(prepImgRows)), (float(0), float(prepImgRows))]
                captCorners = cv2.perspectiveTransform(np.array(prepCorners)[None, :, :], homographyMatrixDebug)[0]

                cv2.line(matchesImg, (int(captCorners[0][0] + prepImgCols), int(captCorners[0][1])), (int(captCorners[1][0] + prepImgCols), int(captCorners[1][1])), (0, 255, 0), 4)
                cv2.line(matchesImg, (int(captCorners[1][0] + prepImgCols), int(captCorners[1][1])), (int(captCorners[2][0] + prepImgCols), int(captCorners[2][1])), (0, 255, 0), 4)
                cv2.line(matchesImg, (int(captCorners[2][0] + prepImgCols), int(captCorners[2][1])), (int(captCorners[3][0] + prepImgCols), int(captCorners[3][1])), (0, 255, 0), 4)
                cv2.line(matchesImg, (int(captCorners[3][0] + prepImgCols), int(captCorners[3][1])), (int(captCorners[0][0] + prepImgCols), int(captCorners[0][1])), (0, 255, 0), 4)
                
                # Show the images with matches
                cv2.imshow("Debug - Matches", matchesImg)

        if(g_homographyMatrix is not None):
            return cv2.warpPerspective(img, g_homographyMatrix, (prepImgCols, prepImgRows))
        else:
            return None

# Loads arguments and input files
def loadInputFiles():
    global g_prepImgPath, g_prepKeysPath, g_prepImg, g_prepKeys

    # Load arguments
    if len(sys.argv) > 1:
        g_prepImgPath = sys.argv[1]

        if len(sys.argv) > 2:
            g_prepKeysPath = sys.argv[2]

    # Load preparation image
    g_prepImg = cv2.imread(g_prepImgPath)

    if(g_prepImg is None):
        print("Couldn't load image")
        return False

    # Load keys data
    try:
        fjson = open(g_prepKeysPath, 'r')
        g_prepKeys = json.load(fjson)
    except (IOError, json.decoder.JSONDecodeError) as ex:
        logging.exception("Couldn't open keys file")
        return False

    return True

def programLoop(cap):
    global g_gotHandHistrogram, g_cropImgHist, g_bgSubtractor, g_gotClearBGFrame

    hasKeyboardMatch = False

    while(True):
        # Capture frame-by-frame
        _, captImg = cap.read()

        # Copy image so original stays unchanged
        imgCopy = captImg.copy()

        # Hand histogram hasn't been obtained yet
        if not g_gotHandHistrogram:
            topLeft, topRight, bottomRight, bottomLeft = opencv_utilities.getCenterSquareOfImg(imgCopy, g_scanSquareSide)

            # Draw scan area on screen
            cv2.line(imgCopy, topLeft, bottomLeft, (0, 255, 0), 2)
            cv2.line(imgCopy, bottomLeft, bottomRight, (0, 255, 0), 2)
            cv2.line(imgCopy, bottomRight, topRight, (0, 255, 0), 2)
            cv2.line(imgCopy, topRight, topLeft, (0, 255, 0), 2)
        else:
            # Get top view keyboard image
            imgKeyboard = matchAndProcessHomography(captImg.copy())

            hasKeyboardMatch = imgKeyboard is not None

            if hasKeyboardMatch:
                fingerContour = getFingerContour(imgKeyboard.copy())

                if fingerContour is not None:
                    fingertip = getFingerTip(fingerContour, imgKeyboard.copy())

                    if fingertip is not None:
                        detectPressedKey(fingertip)

                # Draw debug top view image
                if debugImages:
                    debugImgKeyboard = imgKeyboard.copy()

                    for key in g_prepKeys['keys']:
                        if debugKeyCorners:
                            for point in key['points']:
                                cv2.circle(debugImgKeyboard, (point['x'], point['y']), 2, (0, 255, 0), -1)

                        if debugKeyEdges:
                            cv2.line(debugImgKeyboard, (key['points'][len(key['points']) - 1]['x'], key['points'][len(key['points']) - 1]['y']), (key['points'][0]['x'], key['points'][0]['y']), (0, 255, 0), 2)

                            for i in range(0, len(key['points']) - 1):
                                cv2.line(debugImgKeyboard, (key['points'][i]['x'], key['points'][i]['y']), (key['points'][i + 1]['x'], key['points'][i + 1]['y']), (0, 255, 0), 2)

                        if debugKeyLabels:
                            cv2.putText(debugImgKeyboard, key['symbol'], (key['points'][0]['x'], key['points'][0]['y']),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

                    # Show image with only keyboard
                    cv2.imshow("Debug - Top View Keyboard", debugImgKeyboard)
            
        # Write feedback and instructions for user on captured image
        if not g_gotHandHistrogram:
            opencv_utilities.writeTextOnImg(imgCopy, 'Fill the square area with your hand', (20, 40), (255, 255, 255), (0, 0, 0))
            opencv_utilities.writeTextOnImg(imgCopy, 'and press \'SPACE\'', (20, 75), (255, 255, 255), (0, 0, 0))
        else:
            # Show messsage indicating if keyboard is being detected or not
            if hasKeyboardMatch:
                opencv_utilities.writeTextOnImg(imgCopy, 'Keyboard detected', (20, imgCopy.shape[0] - 40), (0, 100, 0), (255, 255, 255))
            else:
                opencv_utilities.writeTextOnImg(imgCopy, 'Keyboard not detected', (20, imgCopy.shape[0] - 40), (0, 0, 160), (255, 255, 255))
            
            if g_gotClearBGFrame:
                opencv_utilities.writeTextOnImg(imgCopy, 'You can now write', (20, imgCopy.shape[0] - 110), (255, 255, 255), (0, 0, 0))

                drawKeyOutline(imgCopy)
            else:
                opencv_utilities.writeTextOnImg(imgCopy, 'Point the camera at the keyboard', (20, 40), (255, 255, 255), (0, 0, 0))
                opencv_utilities.writeTextOnImg(imgCopy, 'and press \'SPACE\'', (20, 75), (255, 255, 255), (0, 0, 0))

        global g_lastPressedKey, g_lastPressedKeyHomographyPoints

        # Write pressed key on captured image for specified time
        if g_lastPressedKey is not None:
            opencv_utilities.writeTextOnImg(imgCopy, g_lastPressedKey, (10, 80), (255, 255, 255), (0, 0, 0), size=2.5, thickness=4)

            if time.time() - g_showKeyStartTime >= g_keyShowSeconds:
                g_lastPressedKey = None
                g_lastPressedKeyHomographyPoints = None

        # Display the resulting frame
        cv2.imshow('Camera Image', imgCopy)
        
        # Gets user input
        k = cv2.waitKey(1) & 0xFF

        # SPACE key
        if k == 32:
            # If hand histogram hasn't been retrieved, do it now
            if not g_gotHandHistrogram:
                g_cropImgHist = getImageHistogram(captImg.copy())
                g_gotHandHistrogram = g_cropImgHist is not None
            # If clear background frame hasn't been retrieved, do it now
            elif not g_gotClearBGFrame:
                g_bgSubtractor = cv2.createBackgroundSubtractorMOG2(history=10, varThreshold=30, detectShadows=False)
                g_gotClearBGFrame = g_bgSubtractor is not None
        elif k == ord('h') or k == ord('H'):
            # Reset hand histogram
            g_gotHandHistrogram = False
        elif k == ord('b') or k == ord('B'):
            # Reset clear background frame
            g_gotClearBGFrame = False
        # ESCAPE key
        elif k == 27:
            return

### MAIN CODE ###

if loadInputFiles():
    # Capture video from camera
    cap = cv2.VideoCapture(int(os.environ.get('CAMERA_ID', '0')))

    if cap.isOpened():
        # Set the resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        # Based on:
        # - https://docs.opencv.org/2.4/doc/tutorials/features2d/feature_homography/feature_homography.html#feature-homography
        # - https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html

        # Detect the keypoints and compute descriptors of preparation image
        g_prepKeypoints, g_prepDescriptors = g_sift.detectAndCompute(g_prepImg, None)

        if g_prepKeypoints is not None and g_prepDescriptors is not None:
            programLoop(cap)

            # Empty line on console before finishing program to single out user input
            print('\n', end='', flush=True)

            saveOutput()

    else:
        print('Could not open video device')

    cap.release()
    cv2.destroyAllWindows()
