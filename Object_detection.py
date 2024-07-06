import cv2
import numpy as np
import cv2, queue, threading, time
from normal_object import object_detect
# bufferless VideoCapture
class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()
# to speech conversion
# https://github.com/rocapal/fish_detection/tree/master LD_PRELOAD=/usr/lib/arm-linux-gnueabihf/libatomic.so.1.2.0 python3

import os 

import time

from glob import glob

import random as rd
# Load Yolo
net = cv2.dnn.readNet("./weigh/fish.weights", "./weigh/fish.cfg")
classes = []
with open("fish.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

per=0
while True:

   
        ret, img = cap.read()
        
        #frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        #cv2.imshow('Input', frame)
        # Loading image

        img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        height, width, channels = img.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                   # Object detected
                   center_x = int(detection[0] * width)
                   center_y = int(detection[1] * height)
                   w = int(detection[2] * width)
                   h = int(detection[3] * height)

                   # Rectangle coordinates
                   x = int(center_x - w / 2)
                   y = int(center_y - h / 2)

                   boxes.append([x, y, w, h])
                   confidences.append(float(confidence))
                   class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)
        print(indexes)
        font = cv2.FONT_HERSHEY_PLAIN
        if len(boxes)>=1:
            for i in range(len(boxes)):
                if i in indexes:
                  x, y, w, h = boxes[i]
                  label = str(classes[class_ids[i]])
                  color = colors[0]
                  cnt=0
                  cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
                  cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        else:
            img,label=object_detect(img)
#          cv2.putText(img,'Number of fish='+str(cnt), (10, 20), font, 3, color, 3)
        print(label)
        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        if key == 27:
            break
   

cap.release()
cv2.destroyAllWindows()
