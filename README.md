
# Face Detection with OpenCV
## Overview

This is a simple face detection program using Python.
I will be using the OpenCV library, which is used as the primary tool for the tasks of computer vision in Python. If you are new to OpenCV then this is the project to start with it.

 
Haar Cascade Algorithm is also used in the project


 

## Documentation

[OpenCV](https://opencv.org)

[Haar Cascade Algorithm](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html)
[  [Download .xml file]](https://github.com/opencv/opencv/tree/master/data/haarcascades)


## Installation

To install the packages you can execute the following command:-


Install OpenCV

```bash
  !pip install opencv-python
  import cv2
```

Install Numpy

```bash
  !pip install numpy
  import numpy as np
```
        
## Deployment

For Face Detection in image run:

```bash
  def face_detection(img):

    image = img.copy()
    # step-1: Converting image into gray scale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # step-2: Applying gray scale image to cascasde classifier
    box,detections = face_cascade.detectMultiScale2(gray,minNeighbors=8)
    # step-3: Drawing the bounding box
    for x,y,w,h in box:

        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)
        
    return image

```
```bash
img_detect = face_detection(img)


cv2.imshow('face detection',img_detect)
cv2.waitKey(0)
cv2.destroyAllWindows()
```


![Image-FaceDetection](https://user-images.githubusercontent.com/20362216/143088270-e7652ea5-be0a-441b-b264-7b5163ed422c.jpeg)
 

For Real Time Face Detection using WebCam run:

```bash
 cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if ret == False:
        break
        
    img_detect = face_detection(frame)
    
    cv2.imshow('Real Time Face Detection',img_detect)       
    if cv2.waitKey(1) == ord('a'):                          # Press 'a' to pause window
        break
        
        
cap.release()
cv2.destroyAllWindows()
```



![WebCam-FaceDetection](https://user-images.githubusercontent.com/20362216/143088184-83ed9605-3994-4158-bd93-d85b92a00f45.jpeg)

## Acknowledgements

 - [Face Detection with Python](https://thecleverprogrammer.com/2020/10/09/face-detection-with-python/)
 - [StackOverflow](https://stackoverflow.com)
  
