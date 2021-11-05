import cv2
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()

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

pts_dst = np.array([[1,1],[333,1],[333, 444],[1, 444]])

while(True):
    # Capture frame-by-frame
    success, img = cap.read()

    # Convert from RGB to grayscale
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Smoothing
    imgSmooth = cv2.blur(imgGray, (5, 5))

    # Threshold
    (T, imgThresh) = cv2.threshold(imgSmooth, 120, 255, cv2.THRESH_BINARY)

    # Find Countours
    imgCanny = cv2.Canny(imgThresh, 100, 200)

    contours, hierarchy = cv2.findContours(
        imgCanny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Draw countours in original image
    image_copy = img.copy()
    
    mask4 = np.zeros_like(image_copy)
    mask4 = cv2.cvtColor(mask4, cv2.COLOR_BGR2GRAY)
    #mask[contours > 0.01*contours.max()] = 255
    
    for contour in contours:
        if cv2.contourArea(contour) > 10000:
            cv2.drawContours(image=mask4, contours=contour, contourIdx=-1,
                     color=(255, 255, 255), thickness=2, lineType=cv2.LINE_8)
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True)

            for point in approx:
                x, y = point[0]
                cv2.circle(image_copy, (x, y), 3, (0, 255, 0), -1)

            # drawing skewed rectangle
            cv2.drawContours(image_copy, [approx], -1, (0, 255, 0))

            if(len(approx) == 4):
                h, status = cv2.findHomography(approx, pts_dst)

                im_dst = cv2.imread('book2.jpg')
                
                im_out = cv2.warpPerspective(img, h, (im_dst.shape[1],im_dst.shape[0]))
                cv2.imshow("Warped Source Image", im_out)

    """ cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1,
                     color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA) """

    ########################################################################## does not work     Approximate contours with linear features (cvApproxPoly)

    # epsilon = 0.1 * cv2.arcLength(contours[0], True)
    # approx = cv2.approxPolyDP(contours[0], epsilon, True)

    # image_copy2 = img.copy()

    # cv2.drawContours(image_copy2, approx, -1, (0, 255, 0), 10)
    ##########################################################################

    image_copy3 = img.copy()
    image_copy4 = img.copy()

    dst = cv2.cornerHarris(imgGray, 2, 3, 0.04)
    dst2 = cv2.cornerHarris(imgSmooth, 2, 3, 0.04)
    
    # --- create a black image to see where those corners occur ---
    mask = np.zeros_like(image_copy3)
    mask2 = np.zeros_like(image_copy3)

    # --- applying a threshold and turning those pixels above the threshold to white ---
    mask[dst > 0.01*dst.max()] = 255
    mask2[dst2 > 0.01*dst2.max()] = 255

    image_copy3[dst > 0.01 * dst.max()] = [0, 0, 255]  
    image_copy4[dst2 > 0.01 * dst2.max()] = [0, 0, 255]  

    #corners = cv2.goodFeaturesToTrack(mask4, 10000, 0.3, 100)
    corners = cv2.cornerHarris(mask4, 2, 3, 0.04)

    # --- create a black image to see where those corners occur ---
    # mask4 = np.zeros_like(image_copy4)

    # # --- applying a threshold and turning those pixels above the threshold to white ---
    # mask4[dst > 0.01*dst.max()] = 255

    # solvePNP
    """ if corners is not None :
        for i in range(1, len(corners)):
            print(corners[i,0,0])
            cv2.circle(image_copy3, (int(corners[i,0,0]), int(corners[i,0,1])), 7, (0,255,0), 2)
    """
    # Display the resulting frame
    #cv2.imshow('img',img)
    #cv2.imshow('img gray',imgGray)
    #cv2.imshow('img smooth',imgSmooth)
    #cv2.imshow('img threshold',imgThresh)
    cv2.imshow('img canny', imgCanny)
    cv2.imshow('img copy', image_copy)
    # cv2.imshow('img copy 2', image_copy2)
    #cv2.imshow('img copy 3', image_copy3)
    cv2.imshow('img copy 4', image_copy4)
    #cv2.imshow('mask', mask)
    cv2.imshow('mask2', mask2)
    cv2.imshow('mask4', mask4)

    # cv2.imshow('img copy 4', image_copy4)

    # Waits for a user input to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        #print(contours)

        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()