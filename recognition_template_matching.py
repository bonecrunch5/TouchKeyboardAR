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

prepImg = cv2.resize(prepImg, (400, 200))

# Load keys data
try:
    fjson = open(prepKeysPath, 'r')
    prepKeys = json.load(fjson)
except (IOError, json.decoder.JSONDecodeError) as ex:
    logging.exception("Couldn't open keys file")
    exit()

_, w, h = prepImg.shape[::-1]

# Start video capture
cap = cv2.VideoCapture(int(os.environ.get('CAMERA_ID', '0')))

if not (cap.isOpened()):
    print('Could not open video device')
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Set the resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while(True):
    # Capture frame-by-frame
    _, captImg = cap.read()

    height, width, channels = captImg.shape
    print (height)
    print (width)

    height, width, channels = prepImg.shape
    print (height)
    print (width)

    res = cv2.matchTemplate(captImg, prepImg, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    cv2.rectangle(captImg,top_left, bottom_right, 255, 2)
    
    # Show the final image
    cv2.imshow("Matches", captImg)

    # Waits for a user input to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break