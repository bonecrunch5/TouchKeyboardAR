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

while(True):
    # Capture frame-by-frame
    _, captImg = cap.read()

    # Convert from RGB to grayscale
    captImgGrayscale = cv2.cvtColor(captImg, cv2.COLOR_BGR2GRAY)
    
    # Detect the keypoints and compute descriptors
    captKeypoints, captDescriptors = sift.detectAndCompute(captImg,None)

    # Match keypoints
    matches = matcher.knnMatch(prepDescriptors, captDescriptors,k=2)

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
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break