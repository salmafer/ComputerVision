import cv2
import numpy as np


nose_cascade= cv2.CascadeClassifier('/haarcascade_mcs_nose.xml')

if nose_cascade.empty():
  raise IOError('Unable to load the classifier xml file')

cap = cv2.VideoCapture(0)#camera capture
cap.set(3, 640) #WIDTH
cap.set(4, 480) #HEIGHT


while True:
    ret, frame = cap.read() # Capture frame-by-frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    nose_rects = nose_cascade.detectMultiScale(gray, 1.3, 10)


    # Display   
    for (x,y,w,h) in nose_rects:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)
        break

    cv2.imshow('Nose Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):#Press q to exit the running script
        break


cap.release()
cv2.destroyAllWindows()



