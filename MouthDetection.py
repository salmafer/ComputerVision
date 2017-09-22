# detects mouth in streaming video 
# python MouthDetection.py to excute


import cv2
import numpy as np

mouth_cascade = cv2.CascadeClassifier('/haarcascade_mcs_mouth.xml') #path of openCv haarcascarde

if mouth_cascade.empty():
  raise IOError('Unable to load the classifier xml file')# in case of empty or wrong path

cap = cv2.VideoCapture(1) #camera capture
cap.set(3, 640) #WIDTH
cap.set(4, 480) #HEIGHT
# ds_factor = 0.5 #frame parameter  

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    # frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA) #define frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mouth_rects = mouth_cascade.detectMultiScale(gray, 1.7, 22)

        # Display
    for (x,y,w,h) in mouth_rects:
        y = int(y - 0.15*h)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        break

    cv2.imshow('Mouth', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): #Press q to exit the running script
        break

cap.release()
cv2.destroyAllWindows()



