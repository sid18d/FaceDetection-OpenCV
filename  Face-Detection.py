#!/usr/bin/env python
# coding: utf-8

# # Face Detection Project

# # Face Detection in Image

# In[1]:


get_ipython().system('pip install opencv-python')
# Installing required libraries

import numpy as np
import cv2
# Importing neccesary libraries


# In[2]:


img = cv2.imread('faces.jpg')
# Reading the Image

cv2.imshow('face',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Loading the Image


# In[3]:


face_cascade = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')   # Loading Haar Cascade Algorithm


# In[4]:


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


# In[5]:


img_detect = face_detection(img)


cv2.imshow('face detection',img_detect)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # Real Time Face Detection using Webcam

# In[6]:


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


# In[ ]:





# 
