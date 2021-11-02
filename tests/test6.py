import cv2
import numpy as np

# Enable camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 420)

# import cascade file for facial recognition
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# if you want to detect any object for example eyes, use one more layer of classifier as below:
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")


while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Getting corners around the face
    faces = faceCascade.detectMultiScale(imgGray, 1.3, 5)  # 1.3 = scale factor, 5 = minimum neighbor
    # drawing bounding box around face
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)


    #### Homography test
    imgRight = cv2.imread('face.jpg')
    imgRightGray = cv2.cvtColor(imgRight, cv2.COLOR_BGR2GRAY)

    faces2 = faceCascade.detectMultiScale(imgRightGray, 1.3, 5)  # 1.3 = scale factor, 5 = minimum neighbor


    for (x, y, w, h) in faces2:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    pts_dst = np.array([[faces2[0][0], faces2[0][1]],[faces2[0][0] + faces2[0][2], faces2[0][1] + faces2[0][3]]])


    h, status = cv2.findHomography(faces, pts_dst)
    im_out = cv2.warpPerspective(img, h, (imgRight.shape[1],imgRight.shape[0]))


    ####


    cv2.imshow('face_detect', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    cv2.imshow('face_detect2', im_out)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break 


cap.release()
cv2.destroyWindow('face_detect')
cv2.destroyWindow('face_detect2')