import numpy as np
import cv2
import os
from dotenv import load_dotenv

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

while True:
    _, captImg = cap.read()

    captImgTarget = captImg.copy()

    captHeight, captWidth, _ = captImg.shape
    squareSide = 80
    
    captHeightHalf = int(captHeight / 2)
    captWidthHalf = int(captWidth / 2)
    squareSideHalf = int(squareSide / 2)

    topLeft = (captWidthHalf - squareSideHalf, captHeightHalf - squareSideHalf)
    bottomLeft = (captWidthHalf - squareSideHalf, captHeightHalf + squareSideHalf)
    topRight = (captWidthHalf + squareSideHalf, captHeightHalf - squareSideHalf)
    bottomRight = (captWidthHalf + squareSideHalf, captHeightHalf + squareSideHalf)
    
    cv2.line(captImgTarget,
            topLeft,
            bottomLeft,
            (0, 255, 0), 2)

    cv2.line(captImgTarget,
            bottomLeft,
            bottomRight,
            (0, 255, 0), 2)

    cv2.line(captImgTarget,
            bottomRight,
            topRight,
            (0, 255, 0), 2)
    
    cv2.line(captImgTarget,
            topRight,
            topLeft,
            (0, 255, 0), 2)

    cv2.imshow('img', captImgTarget)

    # Waits for a user input to continue
    if cv2.waitKey(1) & 0xFF == 32:
        break

cropImg = captImg[topLeft[1]:bottomLeft[1], topLeft[0]:topRight[0]]
cropImgHsv = cv2.cvtColor(cropImg,cv2.COLOR_BGR2HSV)

# calculating object histogram
cropImgHist = cv2.calcHist([cropImgHsv],[0, 1], None, [12, 12], [0, 180, 0, 256] )

# normalize histogram
# cv2.normalize(cropImgHist,cropImgHist,0,255,cv2.NORM_MINMAX)

while True:
    _, captImg = cap.read()

    captImgTarget = captImg.copy()
    captImgTargetHsv = cv2.cvtColor(captImgTarget,cv2.COLOR_BGR2HSV)

    # apply backprojection
    dst = cv2.calcBackProject([captImgTargetHsv],[0,1],cropImgHist,[0,180,0,256],1)

    # Now convolute with circular disc
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    cv2.filter2D(dst,-1,disc,dst)

    # threshold and binary AND
    ret,thresh = cv2.threshold(dst,50,255,0)
    thresh = cv2.merge((thresh,thresh,thresh))
    res = cv2.bitwise_and(captImg,thresh)

    res = np.vstack((captImgTarget,thresh,res))
    
    # resSmooth = cv2.blur(res, (5, 5))

    cv2.imshow('cropped', cropImg)
    cv2.imshow('res', res)

    # Waits for a user input to quit the application
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
