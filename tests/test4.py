import cv2

# Enable camera
cap = cv2.VideoCapture(2)
cap.set(3, 640)
cap.set(4, 420)

# import cascade file for facial recognition
w_cascade = cv2.CascadeClassifier('watchcascade10stage.xml')



while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Getting corners around the face
    watches = w_cascade.detectMultiScale(imgGray, 1.3, 5)  # 1.3 = scale factor, 5 = minimum neighbor
    # drawing bounding box around face
    for (x, y, w, h) in watches:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    

    cv2.imshow('face_detect', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    cv2.imshow('face_detect2', imgGray)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyWindow('face_detect')
cv2.destroyWindow('face_detect2')