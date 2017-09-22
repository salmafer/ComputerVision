
import dlib
import cv2
import numpy as np
from __future__ import division

def resize(img, width=None, height=None, interpolation=cv2.INTER_AREA):
    global ratio
    w, h = img.shape

    if width is None and height is None:
        return img
    elif width is None:
        ratio = height / h
        width = int(w * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized
    else:
        ratio = width / w
        height = int(h * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized
  
def shape_to_np(shape, dtype="int"):

    coords = np.zeros((68, 2), dtype=dtype)     # initialisation of  the list of coordinates
    for i in range(0, 68): # loop over the landmarks and convert them to coordinates
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords     # return coordinates list

camera = cv2.VideoCapture(0) #camera capture

predictor_path = 'shape_predictor_68_face_landmarks.dat' # defining the path of the shape predictor

detector = dlib.get_frontal_face_detector() # defining face detector
predictor = dlib.shape_predictor(predictor_path) 

while True:

    ret, frame = camera.read()
    if ret == False:
        print('Failed to capture frame from camera. Check camera index in cv2.VideoCapture(0) \n')
        break

    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     frame_resized = resize(frame_grey, width=120)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(frame_resized, 1)
    if len(dets) > 0:
        for k, d in enumerate(dets):
           
           
            shape = predictor(frame_resized, d)  # determine the landmarks for the face region
            shape = shape_to_np(shape)  # convert the facial coordinates to  NumPy array

            
            # Display 
            for (x, y) in shape:
                cv2.circle(frame, (int(x/ratio), int(y/ratio)), 3, (255, 255, 255), -1)

    cv2.imshow("image", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):#Press q to exit the running script
        cv2.destroyAllWindows()
        camera.release()
        break



