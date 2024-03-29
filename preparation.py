import cv2
import numpy as np
import sys
import time
import os
import json
import opencv_utilities
from dotenv import load_dotenv
load_dotenv()

### DEBUG VARS ###

debugImagesEnv = os.environ.get('SHOW_DEBUG_IMAGES')
debugImages = debugImagesEnv is not None and debugImagesEnv.upper() == 'TRUE'

### GLOBAL VARS ###

# Keyboard width and height
g_keyboardWidth = 792
g_keyboardHeight = 380

# Number of keys to recognize
g_numKeys = -1
# List of symbols to recognize in keys (in order)
g_keySymbols = []

# Object that will be written to file
# It will have the info of all keys (symbols and positions) on the keyboard
g_jsonOutput = {}
# The top view image of the keyboard that will be saved
g_imgKeyboard = None

# If the keyboard is currently being detected
g_keyboardDetected = False
# If the keys are currently being detected
g_keysDetected = False

# If it should show the 'No Save Possible' message
g_showNoSaveMsg = False
# Time since 'No Save Possible' message is being shown
g_showNoSaveMsgStartTime = time.time()
# Time to show 'No Save Possible' message
g_showNoSaveMsgSeconds = 2

### FUNCTIONS ###

def processKeys(keysPointsList):
    # If number of keys is as expected, proceed
    if len(keysPointsList) == g_numKeys:
        # Initialize JSON object with each key and its corresponding points
        jsonKeys = []

        # Get top left corner of each key and store in keysTopLeft array
        for index, keyPoints in enumerate(keysPointsList):
            corners = []

            for point in keyPoints:
                x, y = point[0]
                corners.append({ "x": int(x), "y": int(y)})

            # Orders points
            opencv_utilities.bubbleSortPoints(corners)

            # Switch the last two points
            corners[len(corners) - 1], corners[len(corners) - 2] = corners[len(corners) - 2], corners[len(corners) - 1]

            jsonKeys.append({ 'points': corners, 'symbol': None })
            
        jsonObject = {'keys': jsonKeys}

        # Order jsonObject by key, based on keyboard layout
        # Compare y. If y is same (difference between points is lower than N), compare x.
        opencv_utilities.bubbleSortKeys(jsonObject)

        # Add symbol info to jsonOutput
        for index, symbol in enumerate(g_keySymbols):
            jsonObject['keys'][index]['symbol'] = symbol

        return jsonObject
    
    return None

# Get contours of the keys
def getKeysContours(img):
    contours = opencv_utilities.getContours(img, debugImages=debugImages, namePrefix='Keys')

    keyContours = []

    for contour in contours:
        if cv2.contourArea(contour, True) > 700:
            perimeter = cv2.arcLength(contour, True)
            approxPoly = cv2.approxPolyDP(contour, 0.04 * perimeter, True)

            keyContours.append(approxPoly)

    return keyContours

# Get outer contour of the keyboard
def getKeyboardContour(img):
    # Get outer counters of keyboard
    keyboardContours = opencv_utilities.getContours(img, debugImages=debugImages, namePrefix='Keyboard')

    # Find the contour with more than 10000 pixels of area and 4 corners
    for contour in keyboardContours:
        if cv2.contourArea(contour, True) > 10000:
            perimeter = cv2.arcLength(contour, True)
            approxPolygon = cv2.approxPolyDP(contour, 0.05 * perimeter, True)

            if(len(approxPolygon) == 4):
                if(debugImages):
                    # Draw points
                    for point in approxPolygon:
                        x, y = point[0]
                        cv2.circle(img, (x, y), 3, (0, 255, 0), -1)

                    # Draw skewed rectangle
                    cv2.drawContours(img, [approxPolygon], -1, (0, 255, 0))

                    cv2.imshow('Camera Image with detected keyboard contour', img)

                return approxPolygon

# Process homography and get top view keyboard image
def processHomography(img, keyboardContour):
    # Fix keyboard orientation error
    keyboardContour = opencv_utilities.reorder(keyboardContour)

    # Get homography matrix for top view image of keyboard with top view image dimensions
    ptsDst = np.array([[g_keyboardWidth, 1], [g_keyboardWidth, g_keyboardHeight], [1, g_keyboardHeight], [1, 1]])
    homographyMatrix, _ = cv2.findHomography(keyboardContour, ptsDst)

    return cv2.warpPerspective(img, homographyMatrix, (g_keyboardWidth, g_keyboardHeight))

# Get arguments (number of keys or file with list of keys)
def loadArguments():
    global g_numKeys

    if len(sys.argv) == 2:
        argument = sys.argv[1]

        try:
            filePointer = open(argument, "r")

            for line in filePointer:
                stripedLine = line.strip()

                if len(stripedLine) > 0:
                    g_keySymbols.append(stripedLine)

            g_numKeys = len(g_keySymbols)

            return True
        except IOError:
            print("Could not open file " + argument)

    return False

# Save resulting info
def saveOutput():
    if (g_imgKeyboard is not None and g_jsonOutput is not None):
        # Create 'generated' folder
        if not os.path.exists("generated"):
            os.mkdir("generated")

        # Save keys info file
        f = open("generated/keys.json", "w")
        f.write(json.dumps(g_jsonOutput))
        f.close()

        # Save top view keyboard image to file
        cv2.imwrite("generated/imgKeyboard.jpg", g_imgKeyboard)

        print("Keyboard and keys info saved successfully")
    else:
        print("Coudn't save info")

def programLoop(cap):
    """ 
    Steps for acquiring the top image of the keyboard:
    1 - Capture image
    2 - Convert from rgb to grayscale
    3 - Smooth the image 
    4 - Threshold
    5 - Find Contours
    6 - Find the countour with more than 10000 pixels of area
    7 - Aproximate it to a rectangle
    8 - Get the corners
    9 - Find homography to get top imageof the keyboard
    """

    while(True):
        # Capture frame-by-frame
        _, captImg = cap.read()
        
        # Copy image so original stays unchanged
        imgCopy = captImg.copy()

        keyboardContour = getKeyboardContour(captImg.copy())

        global g_keyboardDetected
        g_keyboardDetected = keyboardContour is not None

        if g_keyboardDetected:
            birdsEyeKeyboard = processHomography(captImg.copy(), keyboardContour)
            
            keyContours = getKeysContours(birdsEyeKeyboard)

            jsonObjKeys = processKeys(keyContours)

            global g_keysDetected
            g_keysDetected = jsonObjKeys is not None

        if g_keysDetected:
            global g_imgKeyboard, g_jsonOutput

            g_imgKeyboard = birdsEyeKeyboard
            g_jsonOutput = jsonObjKeys
            
            birdsEyeKeyboardLabels = birdsEyeKeyboard.copy()

            for key in jsonObjKeys['keys']:
                symbol = key['symbol']
                keyPoints = key['points']
                labelPoint = (keyPoints[0]['x'], keyPoints[0]['y'])

                # Draw key outlines
                cv2.line(birdsEyeKeyboardLabels, (keyPoints[len(keyPoints) - 1]['x'], keyPoints[len(keyPoints) - 1]['y']), (keyPoints[0]['x'], keyPoints[0]['y']), (0, 255, 0), 2)

                for i in range(0, len(keyPoints) - 1):
                    cv2.line(birdsEyeKeyboardLabels, (keyPoints[i]['x'], keyPoints[i]['y']), (keyPoints[i + 1]['x'], keyPoints[i + 1]['y']), (0, 255, 0), 2)

                cv2.putText(birdsEyeKeyboardLabels, symbol, labelPoint, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)
            
            cv2.imshow("Frontal Keyboard Image", birdsEyeKeyboardLabels)
        
        # Write user messages on camera image
        if g_keysDetected:
            opencv_utilities.writeTextOnImg(imgCopy, 'Check the keys output', (20, 40), (0, 100, 0), (255, 255, 255))
            opencv_utilities.writeTextOnImg(imgCopy, 'If it is correct, press \'SPACE\' to save', (20, 75), (0, 100, 0), (255, 255, 255))
        elif g_keyboardDetected:
            opencv_utilities.writeTextOnImg(imgCopy, 'Wait for the keys output to appear', (20, 40), (255, 255, 255), (0, 0, 0))
            opencv_utilities.writeTextOnImg(imgCopy, 'Make sure all keys are visible', (20, 75), (255, 255, 255), (0, 0, 0))
            opencv_utilities.writeTextOnImg(imgCopy, 'and the corrent amount was specified', (20, 110), (255, 255, 255), (0, 0, 0))
        else:
            opencv_utilities.writeTextOnImg(imgCopy, 'Point the camera at the keyboard', (20, 40), (255, 255, 255), (0, 0, 0))

        global g_showNoSaveMsg, g_showNoSaveMsgStartTime
        if g_showNoSaveMsg:
            opencv_utilities.writeTextOnImg(imgCopy, 'Can\'t save yet, no keyboard detected', (20, imgCopy.shape[0] - 40), (0, 0, 160), (255, 255, 255))

            # Disable the 'No Save Possible' message after some time
            if time.time() - g_showNoSaveMsgStartTime >= g_showNoSaveMsgSeconds:
                g_showNoSaveMsg = False

        # Display the resulting frame
        cv2.imshow('Camera Image', imgCopy)

        # Gets user input
        userInput = cv2.waitKey(1) & 0xFF

        # SPACE key
        if userInput == 32:
            # If it is not possible to save, show 'No Save Possible' message
            if g_jsonOutput is None or g_imgKeyboard is None:
                g_showNoSaveMsg = True
                g_showNoSaveMsgStartTime = time.time()

                print('Keyboard and keys haven\'t been fully detected yet')
            else:
                return True
        # ESCAPE key
        elif userInput == 27:
            return False


### MAIN CODE ###

if loadArguments():
    # Capture video from camera
    cap = cv2.VideoCapture(int(os.environ.get('CAMERA_ID', '0')))

    if cap.isOpened():
        # Set the resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        shouldSaveOutput = programLoop(cap)

        if shouldSaveOutput:
            saveOutput()
        else:
            print('Exited without saving')
    else:
        print('Could not open video device')

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
else:
    print('Needs valid arguments')
