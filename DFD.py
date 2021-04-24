#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 17:51:11 2021

@author: mrinal
"""
import itertools
import threading
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import cv2
import numpy as np
import os
import shutil
import face_recognition
from IPython.display import Image, display
from statistics import mean
import random
from tqdm import tqdm

done = False
def animate():
    for c in itertools.cycle(['|', '/', '-', '\\']):
        if done:
            break
        sys.stdout.write('\rInitializing Program Please Wait' + c)
        sys.stdout.flush()
        time.sleep(0.1)
    sys.stdout.write('\rInitializtion of Program Completed !     \n')
    sys.stdout.write('\r--------------------------------------------------------------------------------------------------------------\n')
    sys.stdout.write('\rInitializing Frame Processing !     \n')
    
    

def face_compare(img1,img2):
    a = img1
    b = img2
    img = face_recognition.load_image_file(a)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    imgt = face_recognition.load_image_file(b)
    imgt = cv2.cvtColor(imgt,cv2.COLOR_BGR2RGB)
    try:
        facedetect_img = face_recognition.face_locations(img)[0]
        facedetect_imgt = face_recognition.face_locations(imgt)[0]
        encode_img = face_recognition.face_encodings(img)[0]
        encode_imgt = face_recognition.face_encodings(imgt)[0]
        facedist=face_recognition.face_distance([encode_img],encode_imgt)
        v = facedist[0]
        perc = (100*v)
    except IndexError:
        perc = 0
    return perc
def face_compare2(img1,img2):
    a = img1
    b = img2
    img = face_recognition.load_image_file(a)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    imgt = face_recognition.load_image_file(b)
    imgt = cv2.cvtColor(imgt,cv2.COLOR_BGR2RGB)
    try:
        facedetect_img = face_recognition.face_locations(img)[0]
        facedetect_imgt = face_recognition.face_locations(imgt)[0]
        encode_img = face_recognition.face_encodings(img)[0]
        encode_imgt = face_recognition.face_encodings(imgt)[0]
        facedist=face_recognition.face_distance([encode_img],encode_imgt)
        v = facedist[0]
        perc = 100-(100*v)
    except IndexError:
        perc = 0
    return perc

def test_type_1(perc):
    
    status = ""
    if(perc<60):
        status = "Deepfake"
    elif(perc>60):
        status = "Real"
        
    print("Deepfake Detection Completed")
    print("Deepfake Detection Report:") 
    print(f"Originality Percentage:{perc:.2f}")
    print("Result:",status)
    print("Execution Completed")
    shutil.rmtree("Desktop/Major-2/img")
    os.mkdir("Desktop/Major-2/img")
    shutil.rmtree("Desktop/Major-2/data/facedata")
    os.mkdir("Desktop/Major-2/data/facedata")  
    
def test_type_2(perc,factor):
    path1 = "Desktop/Major-2/data/facedata"
    path2 = input("Enter the path of subject's image")
    perclist = np.array(())
    imglst = np.array(())
    for filename in os.listdir(path1):
        sourcepath = os.path.join(path1,filename)
        source_img = cv2.imread(sourcepath)
        if source_img is not None:
            imglst=np.append(imglst,sourcepath) 
    a = 0
    if(factor==True):
        for i in tqdm(range(0,len(imglst)),desc="Deepfake Detection Initiated"):
            a = face_compare(imglst[i],path2)
            perclist = np.append(perclist,a)
        face_dist = np.mean(perclist)
        perc = (perc+face_dist)/2
    if(factor==False):
        for i in tqdm(range(0,len(imglst)),desc="Deepfake Detection Initiated"):
            a = face_compare2(imglst[i],path2)
            perclist = np.append(perclist,a)
        face_dist = np.amax(perclist)
        perc = (perc+face_dist)/2
    
        
    shutil.rmtree("Desktop/Major-2/img")
    os.mkdir("Desktop/Major-2/img")
    shutil.rmtree("Desktop/Major-2/data/facedata")
    os.mkdir("Desktop/Major-2/data/facedata")
    print("Deepfake Detection Completed")
    print("Deepfake Detection Report:")
    print(f"Originality Percentage:{perc:.2f}")
    status = ""
    if(perc<60):
        status = "Deepfake"
    elif(perc>60):
        status = "Real"
    print("Status:",status)  
    print("Execution Completed")
    
#Frame Processing 
print("WELCOME TO THE DEEPFAKE DETECTION TOOL")
src = input("Enter the source path of the video:")
cap = cv2.VideoCapture(src)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
image_dimensions = {'height':256, 'width':256, 'channels':3}
try:
    if not os.path.exists('Desktop/Major-2/data'):
        os.makedirs('Desktop/Major-2/data')
except OSError:
    print ('Error: Creating directory of data')
currentFrame = 0
t = threading.Thread(target=animate)
t.start()
while(True):
    ret, frame = cap.read()
    if frame is None:
        break
    # Saves image of the current frame in jpg file
    name = 'Desktop/Major-2/img/' + str(currentFrame) + '.jpg'
    cv2.imwrite(name, frame)
    image = cv2.imread('Desktop/Major-2/img/{}.jpg'.format(currentFrame))
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_copy = np.copy(image)
    #gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(image_copy, 1.25, 6)
    for f in faces:
        x, y, w, h = [ v for v in f ]
        cv2.rectangle(image_copy, (x,y), (x+w, y+h), (255,0,0), 3)
        
        face_crop = image_copy[y:y+h, x:x+w]
    cv2.imwrite(os.path.join('Desktop/Major-2/data/facedata' , str(currentFrame) + '.jpg'),face_crop)

 
    currentFrame += 1

time.sleep(1)
done = True
cap.release()
cv2.destroyAllWindows()

class Classifier:
    def __init__():
        self.model = 0
    
    def predict(self, x):
        return self.model.predict(x)
    
    def fit(self, x, y):
        return self.model.train_on_batch(x, y)
    
    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)
    
    def load(self, path):
        self.model.load_weights(path)
        
class Meso4(Classifier):
    def __init__(self, learning_rate = 0.001):
        self.model = self.init_model()
        optimizer = Adam(lr = learning_rate)
        self.model.compile(optimizer = optimizer,
                           loss = 'mean_squared_error',
                           metrics = ['accuracy'])
    
    def init_model(self): 
        x = Input(shape = (image_dimensions['height'],
                           image_dimensions['width'],
                           image_dimensions['channels']))
        
        x1 = Conv2D(8, (3, 3), padding='same', activation = 'relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        
        x2 = Conv2D(8, (5, 5), padding='same', activation = 'relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)
        
        x3 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)
        
        x4 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)
        
        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation = 'sigmoid')(y)

        return Model(inputs = x, outputs = y)
    
    
#Main

meso = Meso4()
meso.load('Desktop/Major-2/Meso4_DF') 

dataGenerator = ImageDataGenerator(rescale=1./255)
generator = dataGenerator.flow_from_directory(
    'Desktop/Major-2/data',
    target_size=(256, 256),
    batch_size=1,
    class_mode='binary')   
n = len(generator.labels)
avg=0
for i in range(0,n):
    X, y = generator.next()
    #print(meso.predict(X)[0][0])
    avg = avg+(meso.predict(X)[0][0])
perc  = (avg/n)*100
factor = round(meso.predict(X)[0][0])==y[0]
print("Frame Processing Completed ")
print("--------------------------------------------------------------------------------------------------------------")
boolean = True
print("Type of Detection Available:")
print("   1.One-Layered Detection")
print("   2.Two-Layered Detection")
choice = input("Enter the Index of Your Choice:")
print("--------------------------------------------------------------------------------------------------------------")
if(choice=="1"):
    test_type_1(perc)
elif(choice=="2"):
    test_type_2(perc,factor)




    
        
    


    
#print(f"Predicted likelihood: {meso.predict(X)[0][0]:.4f}")
#print(f"Actual label: {int(y[0])}")
#print(f"\nCorrect prediction: {round(meso.predict(X)[0][0])==y[0]}")

# Showing image
#plt.imshow(np.squeeze(X)); 








