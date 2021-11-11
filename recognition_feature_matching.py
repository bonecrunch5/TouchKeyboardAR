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

# Initialize the ORB detector algorithm
orb = cv2.ORB_create()

# Initiate SIFT detector
sift = cv2.SIFT_create()

# Detect the keypoints and compute descriptors
# prepKeypoints, prepDescriptors = orb.detectAndCompute(prepImgGrayscale,None)
prepKeypoints, prepDescriptors = sift.detectAndCompute(prepImg,None)

# Initialize the Matcher for matching the keypoints
# matcher = cv2.BFMatcher_create()

# FLANN_INDEX_LSH = 6
# index_params = dict(algorithm = FLANN_INDEX_LSH, table_number = 12, key_size = 20, multi_probe_level = 2)
# search_params = dict(checks=50) # or pass empty dictionary

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
    # captKeypoints, captDescriptors = orb.detectAndCompute(captImgGrayscale,None)
    captKeypoints, captDescriptors = sift.detectAndCompute(captImg,None)

    # Match keypoints
    # matches = matcher.match(captDescriptors, prepDescriptors)
    matches = matcher.knnMatch(prepDescriptors, captDescriptors,k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]

    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]

    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = 0)

    img3 = cv2.drawMatchesKnn(prepImg, prepKeypoints, captImg, captKeypoints, matches, None, **draw_params)

    """
    # Quick calculation of max and min distances between keypoints
    maxDist = 0
    minDist = 1000

    for match in matches:
        dist = match.distance
        if dist < minDist:
            minDist = dist
        if dist > maxDist:
            maxDist = dist

    # Draw "good" matches
    goodMatches = []

    maxIndex = len(prepDescriptors)

    if len(matches) < maxIndex:
        maxIndex = len(matches)

    for i in range(0, maxIndex):
        if matches[i].distance < 3*minDist:
            goodMatches.append(matches[i])

    # draw the matches to the final image
    # containing both the images the drawMatches()
    # function takes both images and keypoints
    # and outputs the matched query image with
    # its train image
    final_img = cv2.drawMatches(captImg, captKeypoints, prepImg, prepKeypoints, goodMatches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    final_img = cv2.resize(final_img, (2000,780))
    """
    
    # Localize the object
    prep = []
    capt = []

    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            prep.append(prepKeypoints[m.queryIdx].pt)
            capt.append(captKeypoints[m.trainIdx].pt)

    if len(prep) >= 4 and len(capt) >= 4:
        homographyMatrix, _ = cv2.findHomography(np.array(prep), np.array(capt), method=cv2.RANSAC)

        prepImgRows, prepImgCols, _ = prepImg.shape
        prepCorners = [(float(0), float(0)), (float(prepImgCols), float(0)), (float(prepImgCols), float(prepImgRows)), (float(0), float(prepImgRows))]

        print(homographyMatrix)
        print(np.array(prepCorners))

        if(homographyMatrix is not None):
            captCorners = cv2.perspectiveTransform(np.array(prepCorners)[None, :, :], homographyMatrix)

            print((captCorners[0][0][0] + prepImgCols, captCorners[0][0][1]))

            cv2.line(img3, (int(captCorners[0][0][0] + prepImgCols), int(captCorners[0][0][1])), (int(captCorners[0][1][0] + prepImgCols), int(captCorners[0][1][1])), (0, 255, 0), 4)
            cv2.line(img3, (int(captCorners[0][1][0] + prepImgCols), int(captCorners[0][1][1])), (int(captCorners[0][2][0] + prepImgCols), int(captCorners[0][2][1])), (0, 255, 0), 4)
            cv2.line(img3, (int(captCorners[0][2][0] + prepImgCols), int(captCorners[0][2][1])), (int(captCorners[0][3][0] + prepImgCols), int(captCorners[0][3][1])), (0, 255, 0), 4)
            cv2.line(img3, (int(captCorners[0][3][0] + prepImgCols), int(captCorners[0][3][1])), (int(captCorners[0][0][0] + prepImgCols), int(captCorners[0][0][1])), (0, 255, 0), 4)

        # Show the final image
        # cv2.imshow("Matches", final_img)
        cv2.imshow("Matches", img3)

    # Waits for a user input to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break