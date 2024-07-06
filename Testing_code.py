import os
import cv2
import random
#from glob import glob
import keras
from tensorflow.keras.layers import Input, Convolution2D, Conv2DTranspose, Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
#import matplotlib.pyplot as plt
import time
import sys
import numpy as np
#import scipy.io.wavfile as wav
#import ntpath
import os
#from numpy.lib import stride_tricks
#from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import cv2





import cv2
import numpy as np
import matplotlib.pyplot as plt

from tkinter import filedialog


clas1 = ['UP','Left','Right','close','Open']


from keras.preprocessing import image                  
from tqdm import tqdm

from tensorflow.keras.models import load_model
model = load_model('trained_model_DNN1.h5')

# Note: modified these two functions, so that we can later also read the inception tensors which 
# have a different format 
def path_to_tensor(img_path, width=224, height=224):
    # loads RGB image as PIL.Image.Image type
    #print(img_path)
    img = image.load_img(img_path, target_size=(width, height))
    # convert PIL.Image.Image type to 3D tensor with shape (width, heigth, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, width, height, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths, width=224, height=224):
    list_of_tensors = [path_to_tensor(img_paths, width, height)]
    return np.vstack(list_of_tensors)

#vilization_and_show()


from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 
import requests
import cv2 as cv
import cv2
import numpy as np
import mediapipe as mp 
mp_face_mesh = mp.solutions.face_mesh

# left eyes indices
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ] 

# irises Indices list
LEFT_IRIS = [464,465, 466, 467]
RIGHT_IRIS = [459, 460, 461, 462]

cap = cv.VideoCapture(0)
per=0
count1=0
count2=0
count3=0
count4=0
count5=0
start_or_stop=0
prev=0

f=0
b=0
lf=0
rt=0
st=0
##while(1):
##    r=requests.post('https://api.thingspeak.com/update?api_key=0TXP0U4I4A4D4H9P&field1=3',data='')
##    time.sleep(.2)
##    x=r.text
##    if int(x)>0:
##        break
with mp_face_mesh.FaceMesh(max_num_faces=1,min_detection_confidence=0.5,min_tracking_confidence=0.5) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        img=frame.copy()
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)
        mask = np.zeros((img_h, img_w), dtype=np.uint8)

        if results.multi_face_landmarks:
            # print((results.multi_face_landmarks[0]))

            # [print(p.x, p.y, p.z ) for p in results.multi_face_landmarks[0].landmark]
            
            mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) 
            for p in results.multi_face_landmarks[0].landmark])

            #cv.polylines(frame, [mesh_points[LEFT_IRIS]], True, (0,255,0), 1, cv.LINE_AA)
            #cv.polylines(frame, [mesh_points[RIGHT_IRIS]], True, (0,255,0), 1, cv.LINE_AA)
            
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            cv.circle(frame, center_left, int(l_radius), (0,255,0), 2, cv.LINE_AA)
            x=center_left[0]
            y=center_left[1]
            crop=frame[x:x+100,y:y+100,:]


            #cv.circle(frame, center_right, int(r_radius), (0,255,0), 2, cv.LINE_AA)

            cv.circle(frame, center_left, 1, (0,255,0), -1, cv.LINE_AA)
            #cv.circle(frame, center_right, 1, (0,255,0), -1, cv.LINE_AA)

            # drawing on the mask 
            cv.circle(mask, center_left, int(l_radius), (255,255,255), -1, cv.LINE_AA)
            #cv.circle(mask, center_right, int(r_radius), (255,255,255), -1, cv.LINE_AA)
            contours,_ = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            x,y,w,h = cv2.boundingRect(contours[0]) 
            if w>10 and h>10:
                
                new_img=img[y:y+h,x:x+w,:]
                new_img=cv2.resize(new_img,(60,60))
                cv2.imwrite('temp.jpg',new_img)
                time.sleep(.2)
                test_tensors = paths_to_tensor('temp.jpg')/255
                pred=model.predict(test_tensors)
                
                if np.argmax(pred)==0:
                    count1+=1;
                    #continue
                else:
                    count1=0;

                if np.argmax(pred)==1:
                    count2+=1;
                    #continue
                else:
                    count2=0;

                if np.argmax(pred)==2:
                    count3+=1;
                    #continue
                else:
                    count3=0;
                    
                if np.argmax(pred)==3:
                    count4+=1;
                    #continue
                else:
                    count4=0;
                if np.argmax(pred)==4:
                    count5+=1;
                    #continue
                else:
                    count5=0;
                    
                #print(count)
                if count4>5  and np.argmax(pred)==3:
                    count4=0
                    if prev==0 and f==0 :
                        prev=1
                        print('Wheel chair Started Forward1')
##                        while(1):
##                            r=requests.post('https://api.thingspeak.com/update?api_key=0TXP0U4I4A4D4H9P&field1=1',data='')
##                            time.sleep(.2)
##                            x=r.text
##                            if int(x)>0:
##                                break
                        st=0
                        b=0
                        f=1
                        rt=0
                        lf=0
                    elif prev==1 and st==0:
                        prev=0
                        print('Wheel chair Stoped1')
##                        while(1):
##                            r=requests.post('https://api.thingspeak.com/update?api_key=0TXP0U4I4A4D4H9P&field1=3',data='')
##                            time.sleep(.2)
##                            x=r.text
##                            if int(x)>0:
##                                break
                        st=1
                        b=0
                        f=0
                        rt=0
                        lf=0
                    
                elif np.argmax(pred)==0 and count1>3 and b==0:
                    count1=0
                    #print('Wheel chair Stoped')
                    #r=requests.post('https://api.thingspeak.com/update?api_key=0TXP0U4I4A4D4H9P&field1=3',data='')
                    #time.sleep(.3)
                    print('Wheel chair Back')
##                    while(1):
##                        r=requests.post('https://api.thingspeak.com/update?api_key=0TXP0U4I4A4D4H9P&field1=2',data='')
##                        time.sleep(.2)
##                        x=r.text
##                        if int(x)>0:
##                            break
                    st=0
                    b=1
                    f=0
                    rt=0
                    lf=0
                elif np.argmax(pred)==1 and count2>3 and lf==0:
                    count2=0
                    #print('Wheel chair Stoped')
                    #r=requests.post('https://api.thingspeak.com/update?api_key=0TXP0U4I4A4D4H9P&field1=3',data='')
                    #time.sleep(.3)
                    print('Wheel chair Left')
##                    while(1):
##                        r=requests.post('https://api.thingspeak.com/update?api_key=0TXP0U4I4A4D4H9P&field1=5',data='')
##                        time.sleep(.2)
##                        x=r.text
##                        if int(x)>0:
##                            break
                    st=0
                    b=0
                    f=0
                    rt=0
                    lf=1
                elif np.argmax(pred)==2 and count3>3 and rt==0:
                    count3=0
                    #print('Wheel chair Stoped')
                    #r=requests.post('https://api.thingspeak.com/update?api_key=0TXP0U4I4A4D4H9P&field1=3',data='')
                    #time.sleep(.3)
                    print('Wheel chair Right')
##                    while(1):
##                        r=requests.post('https://api.thingspeak.com/update?api_key=0TXP0U4I4A4D4H9P&field1=4',data='')
##                        time.sleep(.2)
##                        x=r.text
##                        if int(x)>0:
##                            break
                    st=0
                    b=0
                    f=0
                    rt=1
                    lf=0
                elif np.argmax(pred)==5 and count5>3 and f==0:
                    count3=0
                    #print('Wheel chair Stoped')
                    #r=requests.post('https://api.thingspeak.com/update?api_key=0TXP0U4I4A4D4H9P&field1=3',data='')
                    #time.sleep(.3)
                    print('Wheel chair Right')
##                    while(1):
##                        r=requests.post('https://api.thingspeak.com/update?api_key=0TXP0U4I4A4D4H9P&field1=4',data='')
##                        time.sleep(.2)
##                        x=r.text
##                        if int(x)>0:
##                            break
                    st=0
                    b=0
                    f=1
                    rt=0
                    lf=0
                else:
                    if prev==1 and np.argmax(pred)==4 and f==0:
                        
                        print('Wheel chair  Forward')
##                        while(1):
##                            r=requests.post('https://api.thingspeak.com/update?api_key=0TXP0U4I4A4D4H9P&field1=1',data='')
##                            time.sleep(.2)
##                            x=r.text
##                            if int(x)>0:
##                                break
                        f=1
                        st=0
                        b=0
                        rt=0
                        lf=0
                    elif prev==0 and st==0:
                        print('Wheel chair Stoped')
                        #r=requests.post('https://api.thingspeak.com/update?api_key=0TXP0U4I4A4D4H9P&field1=3',data='')
##                        while(1):
##                            r=requests.post('https://api.thingspeak.com/update?api_key=0TXP0U4I4A4D4H9P&field1=3',data='')
##                            time.sleep(.2)
##                            x=r.text
##                            if int(x)>0:
##                                break
                        #time.sleep(.3)
                        st=1
                        b=0
                        f=0
                        rt=0
                        lf=0
                
                    
                #print(np.argmax(pred))
                #print('Given Audio Predicted is : '+str(clas1[np.argmax(pred)]))
                time.sleep(1)
                per=per+1;
           
            
        #cv.imshow('Mask', mask)     
        cv.imshow('img', frame)
        key = cv.waitKey(1)
        if key ==ord('q'):
            break
cap.release()
cv.destroyAllWindows()



  



