#detects face on video streming
import numpy as np
import cv2



face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #path of openCv haarcascarde

if face_cascade.empty():
  raise IOError('Unable to load the classifier xml file') # in case of empty or wrong path

cap = cv2.VideoCapture(0)  #camera capture
cap.set(3, 640) #WIDTH
cap.set(4, 480) #HEIGHT
  
while(True):
   
    ret, frame = cap.read()  # Capture frame-by-frame

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)


    # Display 
    for (x,y,w,h) in faces:
         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),3)
         roi_gray = gray[y:y+h, x:x+w]
         roi_color = frame[y:y+h, x:x+w]

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): #Press q to exit the running script
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


