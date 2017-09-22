
# detects mouth in streaming video 
# python SmileDetection.py to excute

import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('/haarcascade_frontalface_default.xml') #path of openCv haarcascarde
smile_cascade=cv2.CascadeClassifier('/haarcascade_smile.xml')#path of openCv haarcascarde

cap = cv2.VideoCapture(1)  #camera capture
cap.set(3, 640) #WIDTH
cap.set(4, 480) #HEIGHT


while(True):
    
    ret, frame = cap.read()# Capture frame-by-frame

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    faces = face_cascade.detectMultiScale(gray, 1.05, 5)
   

    # Display
    for (x,y,w,h) in faces:
         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),3)
         roi_gray = gray[y:y+h, x:x+w]
         roi_color = frame[y:y+h, x:x+w]
         smiles = smile_cascade.detectMultiScale(roi_gray,1.5,4)
         for (x,y,w,h) in smiles:
              cv2.rectangle(roi_color,(x,y),(x+w,y+h),(0,255,0),1)
              print('smiling',len(smiles))


    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  #Press q to exit the running script
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
  
