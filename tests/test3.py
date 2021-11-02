import cv2

w_cascade = cv2.CascadeClassifier('watchcascade10stage.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if ret:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        watches = w_cascade.detectMultiScale(image=gray,
                                       scaleFactor=1.3,                                      
                                       minNeighbors=50)

        for (x, y, w, h) in watches:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, 'Watch', (x - w, y - h), font, 0.5, (11, 255, 255),  2, cv2.LINE_AA)

        cv2.imshow('img', img)
        k = cv2.waitKey(0) & 0xff
        if k == 27:
             break

cap.release()
cv2.destroyAllWindows()