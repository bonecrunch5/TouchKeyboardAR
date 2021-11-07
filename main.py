import cv2
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()

def reorder(points):
    result = np.array([[[0,0]],[[0,0]],[[0,0]],[[0,0]]], ndmin=3)

    mediumPoint1 = [(points[0][0][0] + points[2][0][0])/2, (points[0][0][1] + points[2][0][1])/2]
    mediumPoint2 = [(points[1][0][0] + points[3][0][0])/2, (points[1][0][1] + points[3][0][1])/2]
    averageMediumPoint = [(mediumPoint1[0] + mediumPoint2[0])/2, (mediumPoint1[1] + mediumPoint2[1])/2]

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

while(True):
    # Capture frame-by-frame
    success, img = cap.read()

    # Convert from RGB to grayscale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Smoothing
    imgSmooth = cv2.blur(imgGray, (5, 5))

    # Threshold
    (T, imgThresh) = cv2.threshold(imgSmooth, 100, 255, cv2.THRESH_BINARY) #TODO adjust values of threshold

    # Find Countours
    imgCanny = cv2.Canny(imgThresh, 100, 200)

    contours, hierarchy = cv2.findContours(
        imgCanny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Draw countours in original image
    imageCopy = img.copy()

    # Create Grayscale black image
    blackImage = np.zeros_like(img)
    blackImage = cv2.cvtColor(blackImage, cv2.COLOR_BGR2GRAY)

    #mask[contours > 0.01*contours.max()] = 255

    for contour in contours:
        if cv2.contourArea(contour) > 10000:
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
                approxPolygon = reorder(approxPolygon)
                pts_dst = np.array([[792, 1], [792, 380], [1, 380], [1, 1]])
                h, status = cv2.findHomography(approxPolygon, pts_dst)

                # Hardcoded proportions to be changed in the future
                im_out = cv2.warpPerspective(img, h, (792, 380))
                cv2.imshow("Warped Source Image", im_out)

                # Convert from RGB to grayscale
                imgGray_out = cv2.cvtColor(im_out, cv2.COLOR_BGR2GRAY)

                # Smoothing
                imgSmooth_out = cv2.blur(imgGray_out, (5, 5))

                # Threshold
                (T, imgThresh_out) = cv2.threshold(
                    imgSmooth_out, 120, 255, cv2.THRESH_BINARY)

                # Find Countours
                imgCanny_out = cv2.Canny(imgThresh_out, 100, 200)

                cv2.imshow('img canny out', imgCanny_out)

                contours_out, hierarchy_out = cv2.findContours(
                    imgCanny_out, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

                im_out_copy = im_out.copy()
                mask_out = np.zeros_like(im_out_copy)
                mask_out = cv2.cvtColor(mask_out, cv2.COLOR_BGR2GRAY)

                for contour_out in contours_out:
                    if cv2.contourArea(contour_out) > 700:
                        cv2.drawContours(image=mask_out, contours=contour_out, contourIdx=-1,
                             color=(255, 255, 255), thickness=2, lineType=cv2.LINE_8)
                        perimeter_out = cv2.arcLength(contour_out, True)
                        approx_out = cv2.approxPolyDP(contour_out, 0.05 * perimeter_out, True)

                        for point_out in approx_out:
                            x, y = point_out[0]
                            cv2.circle(im_out_copy, (x, y), 3, (0, 255, 0), -1)

                # dst_out = cv2.cornerHarris(imgSmooth_out, 2, 3, 0.04)

                # im_out_copy = im_out.copy()

                # mask_out = np.zeros_like(im_out_copy)
                # mask_out[dst_out > 0.01*dst_out.max()] = 255
                        cv2.imshow('mask_out', mask_out)

    imageCopy3 = img.copy()
    imageCopy4 = img.copy()

    dst = cv2.cornerHarris(imgGray, 2, 3, 0.04)
    dst2 = cv2.cornerHarris(imgSmooth, 2, 3, 0.04)

    # --- create a black image to see where those corners occur ---
    mask = np.zeros_like(imageCopy3)
    mask2 = np.zeros_like(imageCopy4)

    # --- applying a threshold and turning those pixels above the threshold to white ---
    mask[dst > 0.01*dst.max()] = 255
    mask2[dst2 > 0.01*dst2.max()] = 255

    imageCopy3[dst > 0.01 * dst.max()] = [0, 0, 255]
    imageCopy4[dst2 > 0.01 * dst2.max()] = [0, 0, 255]

    #corners = cv2.goodFeaturesToTrack(mask4, 10000, 0.3, 100)
    corners = cv2.cornerHarris(blackImage, 2, 3, 0.04)

    # --- create a black image to see where those corners occur ---
    # mask4 = np.zeros_like(imageCopy4)

    # # --- applying a threshold and turning those pixels above the threshold to white ---
    # mask4[dst > 0.01*dst.max()] = 255

    # solvePNP
    """ if corners is not None :
        for i in range(1, len(corners)):
            print(corners[i,0,0])
            cv2.circle(imageCopy3, (int(corners[i,0,0]), int(corners[i,0,1])), 7, (0,255,0), 2)
    """
    # Display the resulting frame
    cv2.imshow('img',img)
    #cv2.imshow('img gray',imgGray)
    #cv2.imshow('img smooth',imgSmooth)
    #cv2.imshow('img threshold',imgThresh)
    cv2.imshow('img canny', imgCanny)
    #cv2.imshow('img copy', imageCopy)
    #cv2.imshow('img copy 2', imageCopy2)
    #cv2.imshow('img copy 3', imageCopy3)
    #cv2.imshow('img copy 4', imageCopy4)
    #cv2.imshow('mask', mask)
    #cv2.imshow('mask2', mask2)
    #cv2.imshow('mask4', mask4)

    # cv2.imshow('img copy 4', imageCopy4)

    # Waits for a user input to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # print(contours)

        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
