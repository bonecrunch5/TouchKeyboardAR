import cv2
import numpy as np

# Write text with outline on screen
def writeTextOnImg(img, text, point, color, outlineColor, size=0.9, thickness=1):
    cv2.putText(img, text, point, cv2.FONT_HERSHEY_SIMPLEX, size, outlineColor, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, point, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)

# Returns center of mass of area
# Image needs to contain black background and white area
def getCenterOfMass(img):
    try:
        massX, massY = np.where(img >= 255)
        centerX = int(np.average(massX))
        centerY = int(np.average(massY))

        return (centerX, centerY)
    except ValueError:
        return None

# Get corners of square at center of image
# Return order: topLeft, topRight, bottomRight, bottomLeft
def getCenterSquareOfImg(img, squareSide):
    imgHeight, imgWidth, _ = img.shape
    
    imgHeightHalf = int(imgHeight / 2)
    imgWidthHalf = int(imgWidth / 2)
    squareSideHalf = int(squareSide / 2)

    topLeft = (imgWidthHalf - squareSideHalf, imgHeightHalf - squareSideHalf)
    bottomLeft = (imgWidthHalf - squareSideHalf, imgHeightHalf + squareSideHalf)
    topRight = (imgWidthHalf + squareSideHalf, imgHeightHalf - squareSideHalf)
    bottomRight = (imgWidthHalf + squareSideHalf, imgHeightHalf + squareSideHalf)

    return topLeft, topRight, bottomRight, bottomLeft

# Check if point is inside square
def isInSquare(point, topLeftVertex, botRightVertex, offset=10):
    x = point['x']
    y = point['y']
    
    minX = topLeftVertex['x']
    minY = topLeftVertex['y']
    maxX = botRightVertex['x']
    maxY = botRightVertex['y']

    return minX - offset < x and minY - offset < y and maxX + offset > x and maxY + offset > y

# Gets the contours in an image
def getContours(image, debugImages=False, namePrefix='Image'):
    # Convert from RGB to grayscale
    imgGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Smoothing
    imgSmooth = cv2.blur(imgGray, (5, 5))

    # Threshold
    (_, imgThresh) = cv2.threshold(imgSmooth, 100, 255, cv2.THRESH_BINARY)

    # Find Countours
    imgCanny = cv2.Canny(imgThresh, 100, 200)

    contours, _ = cv2.findContours(imgCanny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Show images for debug
    if(debugImages):
        cv2.imshow(namePrefix + ' - Gray', imgGray)
        cv2.imshow(namePrefix + ' - Smooth', imgSmooth)
        cv2.imshow(namePrefix + ' - Threshold', imgThresh)
        cv2.imshow(namePrefix + ' - Canny', imgCanny)

    return contours

# Check if a point is to the left or above another point
# Point: { "x": <x_coordinate>, "y": <y_coordinate> }
def lessThanPoint(point1, point2, offset):
    if (abs(point1['y']-point2['y']) < offset):
        return point1['x']-point2['x'] < 0
    return point1['y']-point2['y'] < 0

# Sort points by placement on screen
# points: [
#   {
#       "x": <x_coordinate>,
#       "y": <y_coordinate>
#   },
#   ...
# ]
def bubbleSortPoints(points):
    n = len(points)
 
    for i in range(n-1):
        for j in range(0, n-i-1):
            if lessThanPoint(points[j + 1],points[j], 30):
                points[j], points[j + 1] = points[j + 1], points[j]

# Sort keys by placement on keyboard
# Only first point in points array will be used
# keysList: 
# {
#     "keys": [
#         {
#             "points": [
#                 {
#                     "x": <x_coordinate>,
#                     "y": <y_coordinate>
#                 },
#                 ...
#             ],
#             ...
#         },
#         ...
#     ]
# }
def bubbleSortKeys(keysList):
    keys = keysList['keys']
    n = len(keys)
 
    for i in range(n-1):
        for j in range(0, n-i-1):
            if lessThanPoint(keys[j + 1]['points'][0],keys[j]['points'][0], 30):
                keys[j], keys[j + 1] = keys[j + 1], keys[j]

# Get distance between two points
# point: [ <x_coordinate>, <y_coordinate> ]
def distanceBetweenPoints(point1, point2):
    return np.sqrt(np.square(point1[0]-point2[0])+np.square(point1[1]-point2[1]))

# Reorders points to fix orientation error
def reorder(points):
    distances = []
    for point in points:
        distances.append(distanceBetweenPoints(point[0],[600,0]))
    
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