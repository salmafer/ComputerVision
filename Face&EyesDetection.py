#detects face on video streming
# python Face&EyesDetection.py to excute 

import numpy as np
import cv2


face_cascade = cv2.CascadeClassifier('/haarcascade_frontalface_default.xml') #path of openCv haarcascarde
eye_cascade = cv2.CascadeClassifier('/haarcascades/haarcascade_eye.xml') #path of openCv haarcascarde

cap = cv2.VideoCapture(0) #camera capture
cap.set(3, 640) #WIDTH
cap.set(4, 480) #HEIGHT


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
  

    # Display 
    for (x,y,w,h) in faces:
         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),3)
         roi_gray = gray[y:y+h, x:x+w]
         roi_color = frame[y:y+h, x:x+w]
         eyes = eye_cascade.detectMultiScale(roi_gray)
         for (ex,ey,ew,eh) in eyes:
             cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,0),1)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  #Press q to exit the running script
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


