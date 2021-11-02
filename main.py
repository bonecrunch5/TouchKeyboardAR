import cv2
import numpy as np

cap = cv2.VideoCapture(2)

if not (cap.isOpened()):
    print('Could not open video device')

# To set the resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

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
    imgCanny = cv2.Canny(img, 100, 200)

    contours, hierarchy = cv2.findContours(
        imgCanny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Draw countours in original image
    image_copy = img.copy()
    cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1,
                     color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    ########################################################################## does not work     Approximate contours with linear features (cvApproxPoly)

    """ epsilon = 0.1 * cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], epsilon, True)

    image_copy2 = img.copy()

    cv2.drawContours(image_copy2, approx, -1, (0, 255, 0), 10) """
    ##########################################################################

    image_copy3 = img.copy()

    dst = cv2.cornerHarris(imgGray, 2, 3, 0.04)
    
    # --- create a black image to see where those corners occur ---
    mask = np.zeros_like(image_copy3)

    # --- applying a threshold and turning those pixels above the threshold to white ---
    mask[dst > 0.01*dst.max()] = 255

    image_copy3[dst > 0.01 * dst.max()] = [0, 0, 255]  

    # solvePNP

    # Display the resulting frame
    #cv2.imshow('img',img)
    #cv2.imshow('img gray',imgGray)
    #cv2.imshow('img smooth',imgSmooth)
    #cv2.imshow('img threshold',imgThresh)
    #cv2.imshow('img canny', imgCanny)
    #cv2.imshow('img copy', image_copy)
    #cv2.imshow('img copy 2', image_copy2)
    cv2.imshow('img copy 3', image_copy3)
    cv2.imshow('mask', mask)



    # Waits for a user input to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
