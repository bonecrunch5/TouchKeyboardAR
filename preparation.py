import cv2
import numpy as np
import sys
import os
import json
from dotenv import load_dotenv
load_dotenv()

numKeys = -1
keySymbols = []

# Get arguments (number of keys or file with list of keys)
if len(sys.argv) > 1:
    argument = sys.argv[1]

    try:
        numKeys = int(argument)
    except ValueError:
        keysFile = argument

        try:
            filePointer = open(keysFile, "r")

            for line in filePointer:
                stripedLine = line.strip()

                if len(stripedLine) > 0:
                    keySymbols.append(stripedLine)

            numKeys = len(keySymbols)
        except IOError:
            print("Could not open file " + keysFile)
            
def lessThanPoint(point1, point2, offset):
    if (abs(point1['y']-point2['y']) < offset):
        return point1['x']-point2['x'] < 0
    return point1['y']-point2['y'] < 0

def bubbleSortKeys(keysArray):
    n = len(keysArray)
 
    for i in range(n-1):
        for j in range(0, n-i-1):
            if lessThanPoint(keysArray[j + 1]['point'],keysArray[j]['point'], 10):
                keysArray[j], keysArray[j + 1] = keysArray[j + 1], keysArray[j]

def distance2points(point1, point2):
    return np.sqrt(np.square(point1[0]-point2[0])+np.square(point1[1]-point2[1]))

def reorder(points):
    distances = []
    for point in points:
        distances.append(distance2points(point[0],[600,0]))
    
    minDistance = distances[0]
    minDistanceIndex = 0

    for x in range(1,4):
        if distances[x] < minDistance:
            minDistance = distances[x]
            minDistanceIndex = x

    newPoints = np.array([[[0, 0]], [[0, 0]], [[0, 0]], [[0, 0]]], ndmin=3)
    for x in range(4):
        newPoints[x] = points[(x + minDistanceIndex)%4]

    return newPoints

cap = cv2.VideoCapture(int(os.environ.get('CAMERA_ID', '0')))

if not (cap.isOpened()):
    print('Could not open video device')
    cap.release()
    cv2.destroyAllWindows()
    exit()

# To set the resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# What sorcery is happening down here to get the top image of the keyboard:
""" 
    1-Capture image
    2-Convert from rgb to grayscale
    3-Smooth the image 
    4-Threshold
    5-Find Contours
    6-Find the countour with more than 10000 pixels of area
    7-Aproximate it to a rectangle
    8-Get the corners
    9-Find homography to get top imageof the keyboard

    Note: Some shady things going on the arguments of the warpPerspective function because of the size of the output image, need to fix that
          (Its using the size of the book experiment image, maybe because de destination points are from there, need to change that)
    Note2: There's some code doing nothing, in the future we need to clean that, but by now it's there if it's needed

 """
jsonOutput = {}
imgKeyboard = None
imgKeyboardFinal = None

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

    return contours


def processKeys(contours_out):

    jsonObject = {'keys': []}
    for contour_out in contours_out:
        if cv2.contourArea(contour_out, True) > 700:
            cv2.drawContours(image=mask_out, contours=contour_out, contourIdx=-1,
                             color=(255, 255, 255), thickness=2, lineType=cv2.LINE_8)
            perimeter_out = cv2.arcLength(contour_out, True)
            approx_out = cv2.approxPolyDP(
                contour_out, 0.04 * perimeter_out, True)

            jsonKey = { 'points': [], 'symbol': None}
            for point_out in approx_out:
                x, y = point_out[0]

                jsonKey['points'].append({ "x": int(x), "y": int(y)})

                if(os.environ.get('SHOW_KEY_CORNERS').upper() == 'TRUE'):
                    cv2.circle(im_out_copy, (x, y), 3, (0, 255, 0), -1)
                    
            jsonObject['keys'].append(jsonKey)

    cv2.imshow('im_out_copy', im_out_copy)
    cv2.imshow('mask_out', mask_out)

    return jsonObject


def processHomography(approxPolygon):
    approxPolygon = reorder(approxPolygon)
    pts_dst = np.array([[792, 1], [792, 380], [1, 380], [1, 1]])
    homographyMatrix, _ = cv2.findHomography(approxPolygon, pts_dst)

    # Hardcoded proportions to be changed in the future
    global imgKeyboard
    imgKeyboard = cv2.warpPerspective(img, homographyMatrix, (792, 380))

while(True):
    # Capture frame-by-frame
    _, img = cap.read()

    contours = getContours(img)

    # Draw countours in original image
    imageCopy = img.copy()

    # Create Grayscale black image
    blackImage = np.zeros_like(img)
    blackImage = cv2.cvtColor(blackImage, cv2.COLOR_BGR2GRAY)

    for contour in contours:
        if cv2.contourArea(contour, True) > 10000:
            cv2.drawContours(image=blackImage, contours=contour, contourIdx=-1,
                             color=(255, 255, 255), thickness=2, lineType=cv2.LINE_8)
            perimeter = cv2.arcLength(contour, True)
            approxPolygon = cv2.approxPolyDP(contour, 0.05 * perimeter, True)

            for point in approxPolygon:
                x, y = point[0]
                cv2.circle(imageCopy, (x, y), 3, (0, 255, 0), -1)

            # drawing skewed rectangle
            cv2.drawContours(imageCopy, [approxPolygon], -1, (0, 255, 0))

            if(len(approxPolygon) == 4):
                processHomography(approxPolygon)
                contours_out = getContours(imgKeyboard)

                im_out_copy = imgKeyboard.copy()
                mask_out = np.zeros_like(im_out_copy)
                mask_out = cv2.cvtColor(mask_out, cv2.COLOR_BGR2GRAY)

                jsonObject = processKeys(contours_out)

                # If number of keys is as expected, proceed
                if numKeys == -1 or len(jsonObject['keys']) == numKeys:
                    # Will hold the keys with their index on jsonOutput and their top left corner
                    keysTopLeft = []

                    # Get top left corner of each key and store in keysTopLeft array
                    for i, key in enumerate(jsonObject['keys']):
                        keyPoints = key['points']

                        corners = []

                        for point in keyPoints:
                            corners.append({'point': point})

                        bubbleSortKeys(corners)

                        keysTopLeft.append({'index': i, 'point': corners[0]['point']})

                    # Order keysTopLeft array by key, based on keyboard layout
                    # Compare y. If y is same (difference between points is lower than N), compare x.
                    bubbleSortKeys(keysTopLeft)

                    keyPlacementImg = imgKeyboard.copy()

                    # Add symbol info to jsonOutput (also show symbols on image display)
                    for i, symbol in enumerate(keySymbols):
                            key = keysTopLeft[i]
                            
                            jsonObject['keys'][key['index']]['symbol'] = symbol

                            if(os.environ.get('SHOW_KEY_LABELS').upper() == 'TRUE'):
                                cv2.putText(keyPlacementImg, symbol, (key['point']['x'], key['point']['y']), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

                    cv2.imshow("Frontal Keyboard Image", keyPlacementImg)
                    
                    imgKeyboardFinal = imgKeyboard
                    jsonOutput = jsonObject 
                else:
                    print("wrong number of keys")

    # Display the resulting frame
    cv2.imshow('img', img)
    #cv2.imshow('img gray',imgGray)
    #cv2.imshow('img smooth',imgSmooth)
    #cv2.imshow('img threshold',imgThresh)
    #cv2.imshow('img canny', imgCanny)
    #cv2.imshow('img copy', imageCopy)
    #cv2.imshow('blackImage', blackImage)

    # Waits for a user input to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save info
if not os.path.exists("generated"):
    os.mkdir("generated")
f = open("generated/keys.json", "w")
f.write(json.dumps(jsonOutput))
f.close()

if (imgKeyboardFinal is not None):
    cv2.imwrite("generated/imgKeyboard.jpg", imgKeyboardFinal)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
