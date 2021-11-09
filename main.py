import cv2
import numpy as np
import os
import json
from dotenv import load_dotenv
load_dotenv()


def reorder(points):  # TODO might have to redo this
    result = np.array([[[0, 0]], [[0, 0]], [[0, 0]], [[0, 0]]], ndmin=3)

    mediumPoint1 = [(points[0][0][0] + points[2][0][0])/2,
                    (points[0][0][1] + points[2][0][1])/2]
    mediumPoint2 = [(points[1][0][0] + points[3][0][0])/2,
                    (points[1][0][1] + points[3][0][1])/2]
    averageMediumPoint = [(mediumPoint1[0] + mediumPoint2[0])/2,
                          (mediumPoint1[1] + mediumPoint2[1])/2]

    for point in points:
        if point[0][0] > averageMediumPoint[0] and point[0][1] < averageMediumPoint[1]:
            result[0] = point
    for point in points:
        if point[0][0] > averageMediumPoint[0] and point[0][1] > averageMediumPoint[1]:
            result[1] = point
    for point in points:
        if point[0][0] < averageMediumPoint[0] and point[0][1] > averageMediumPoint[1]:
            result[2] = point
    for point in points:
        if point[0][0] < averageMediumPoint[0] and point[0][1] < averageMediumPoint[1]:
            result[3] = point

    for point in result:
        if point[0][0] == 0 and point[0][1] == 0:
            return points
    return result


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
    counter = 0
    jsonObject = {"keys": []}
    for contour_out in contours_out:
        if cv2.contourArea(contour_out, True) > 700:
            counter += 1
            cv2.drawContours(image=mask_out, contours=contour_out, contourIdx=-1,
                             color=(255, 255, 255), thickness=2, lineType=cv2.LINE_8)
            perimeter_out = cv2.arcLength(contour_out, True)
            approx_out = cv2.approxPolyDP(
                contour_out, 0.04 * perimeter_out, True)

            jsonKey = []
            for point_out in approx_out:
                x, y = point_out[0]
                cv2.circle(im_out_copy, (x, y), 3, (0, 255, 0), -1)

                jsonKey.append([int(x), int(y)])

            jsonObject['keys'].append(jsonKey)

    cv2.imshow('im_out_copy', im_out_copy)
    cv2.imshow('mask_out', mask_out)
    print(counter)
    return jsonObject


def processHomography(approxPolygon):
    approxPolygon = reorder(approxPolygon)
    pts_dst = np.array([[792, 1], [792, 380], [1, 380], [1, 1]])
    homographyMatrix, _ = cv2.findHomography(approxPolygon, pts_dst)

    # Hardcoded proportions to be changed in the future
    global imgKeyboard
    imgKeyboard = cv2.warpPerspective(img, homographyMatrix, (792, 380))
    cv2.imshow("Warped Source Image", imgKeyboard)


while(True):
    # Capture frame-by-frame
    _, img = cap.read()

    contours = getContours(img)

    # Draw countours in original image
    imageCopy = img.copy()

    # Create Grayscale black image
    blackImage = np.zeros_like(img)
    blackImage = cv2.cvtColor(blackImage, cv2.COLOR_BGR2GRAY)

    #mask[contours > 0.01*contours.max()] = 255

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
                jsonOutput = json.dumps(jsonObject)

    # Display the resulting frame
    cv2.imshow('img', img)
    #cv2.imshow('img gray',imgGray)
    #cv2.imshow('img smooth',imgSmooth)
    #cv2.imshow('img threshold',imgThresh)
    #cv2.imshow('img canny', imgCanny)
    #cv2.imshow('img copy', imageCopy)
    #cv2.imshow('img copy 2', imageCopy2)
    #cv2.imshow('img copy 3', imageCopy3)
    #cv2.imshow('img copy 4', imageCopy4)
    #cv2.imshow('blackImage', blackImage)

    #cv2.imshow('mask', mask)
    #cv2.imshow('mask2', mask2)
    #cv2.imshow('mask4', mask4)

    # cv2.imshow('img copy 4', imageCopy4)

    # Waits for a user input to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # print(contours)

        break


# When everything done, release the capture
if not os.path.exists("generated"):
    os.mkdir("generated")
f = open("generated/keys.json", "w")
f.write(jsonOutput)
f.close()

cv2.imwrite("generated/imgKeyboard.jpg", imgKeyboard)


cap.release()
cv2.destroyAllWindows()
