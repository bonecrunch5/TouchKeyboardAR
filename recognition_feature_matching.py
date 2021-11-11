import cv2
import sys
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
# sift = cv2.SIFT_create()

# Detect the keypoints and compute descriptors
prepKeypoints, prepDescriptors = orb.detectAndCompute(prepImgGrayscale,None)
# prepKeypoints, prepDescriptors = sift.detectAndCompute(prepImg,None)

# Initialize the Matcher for matching the keypoints
matcher = cv2.BFMatcher_create()

while(True):
    # Capture frame-by-frame
    _, captImg = cap.read()

    # Convert from RGB to grayscale
    captImgGrayscale = cv2.cvtColor(captImg, cv2.COLOR_BGR2GRAY)
    
    # Detect the keypoints and compute descriptors
    captKeypoints, captDescriptors = orb.detectAndCompute(captImgGrayscale,None)
    # captKeypoints, captDescriptors = sift.detectAndCompute(captImg,None)

    # Match keypoints
    matches = matcher.match(prepDescriptors,captDescriptors)

    # draw the matches to the final image
    # containing both the images the drawMatches()
    # function takes both images and keypoints
    # and outputs the matched query image with
    # its train image
    final_img = cv2.drawMatches(captImg, captKeypoints,
        prepImg, prepKeypoints, matches[:20], None)
    
    final_img = cv2.resize(final_img, (2000,780))
    
    # Show the final image
    cv2.imshow("Matches", final_img)

    # Waits for a user input to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break