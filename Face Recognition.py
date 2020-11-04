import cv2
import os
import numpy as np
import face_recognition
from IPython.display import Image, display
from statistics import mean

def face_compare(img1,img2):
    a = img1
    b = img2
    img = face_recognition.load_image_file(a)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    imgt = face_recognition.load_image_file(b)
    imgt = cv2.cvtColor(imgt,cv2.COLOR_BGR2RGB)
    facedetect_img = face_recognition.face_locations(img)[0]
    facedetect_imgt = face_recognition.face_locations(imgt)[0]
    encode_img = face_recognition.face_encodings(img)[0]
    encode_imgt = face_recognition.face_encodings(imgt)[0]
    facedist=face_recognition.face_distance([encode_img],encode_imgt)
    v = facedist[0]
    perc = 100-(100*v)
    return perc

def face_compare_enc(img1,img2):
    a = img1
    b = img2
    img = face_recognition.load_image_file(a)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    imgt = face_recognition.load_image_file(b)
    imgt = cv2.cvtColor(imgt,cv2.COLOR_BGR2RGB)
    facedetect_img = face_recognition.face_locations(img)[0]
    encode_img = face_recognition.face_encodings(img)[0]
    #cv2.rectangle(img,(facedetect_img[3],facedetect_img[0]),(facedetect_img[1],facedetect_img[2]),(255,0,255),2)
    facedetect_imgt = face_recognition.face_locations(imgt)[0]
    encode_imgt = face_recognition.face_encodings(imgt)[0]
    #cv2.rectangle(imgt,(facedetect_imgt[3],facedetect_imgt[0]),(facedetect_imgt[1],facedetect_imgt[2]),(255,0,255),2)
    result = face_recognition.compare_faces([encode_img],encode_imgt)
    result_x = result[0]
    return result_x
            
def perc_test(srclst,destlst,x):
    print("Processing please wait.....")
    perclist = []
    perc_dict = {}
    
    for l in range(0,len(srclst)):
        temp_perc = [] 
        
        for m in range(0,len(destlst)):
            perc_res = face_compare(srclst[l],destlst[m])
            temp_perc.append(perc_res)
        perclist.append(mean(temp_perc))        

    for n in range(0, len(srclst)):
        perc_dict[srclst[n]] = perclist[n]
        
    perclist = []
    
    for o in range(0 , x):
        perc_result = max(perc_dict , key = perc_dict.get)
        perclist.append(perc_result)
        perc_dict.pop(perc_result)
        
    print("Images with highest resemblance:")
    for k in range(0, len(perclist)):
        display(Image(filename=perclist[k]))
    

def layer1(bool_list):
    a = True
    count_t = 0
    count_f = 0
    for i in range (0, len(bool_list)):
        if (bool_list[i] == True):
            count_t = count_t + 1
        else:
            count_f = count_f+1
    if (count_t > 0):
        return a
    else:
        a= False
        return a
    
def enc_test(s,d):
    enclst = []
    print("Processing Please Wait....")
    
    for i in range(0 , len(s)):
        temp_enc = []
        for j in range(0 , len(d)):
            enc_res = face_compare_enc(s[i],d[j])
            temp_enc.append(enc_res)
        temp_result = layer1(temp_enc)
        enclst.append(temp_result)
        
    print("Images with higest resemblance:") 
    for k in range(0,len(srclst)):
        if(enclst[k]== True):
            display(Image(filename=srclst[k]))
            
def combine_test(srclst,destlst):
    enclst = []
    translst =[]
    print("Processing Please Wait....")
    
    for i in range(0 , len(srclst)):
        temp_enc = []
        for j in range(0 , len(destlst)):
            enc_res = face_compare_enc(srclst[i],destlst[j])
            temp_enc.append(enc_res)
        temp_result = layer1(temp_enc)
        enclst.append(temp_result)
    for v in range(0,len(enclst)):
        if(enclst[v]==True):
            translst.append(srclst[v])
        
    perclist = []
    perc_dict = {}
    
    for l in range(0,len(translst)):
        temp_perc = [] 
        
        for m in range(0,len(destlst)):
            perc_res = face_compare(translst[l],destlst[m])
            temp_perc.append(perc_res)
        perclist.append(mean(temp_perc))        

    for n in range(0, len(translst)):
        perc_dict[translst[n]] = perclist[n]
        
    perclist = []
    
    for o in range(0 , 1):
        perc_result = max(perc_dict , key = perc_dict.get)
        perclist.append(perc_result)
        perc_dict.pop(perc_result)
        
    print("Image with highest resemblance:")
    for k in range(0, len(perclist)):
        display(Image(filename=perclist[k]))    
        
        
#main

sourceimg = input("Enter the path of the image folder:")
destimg = input("Enter the path of Subject folder:")
srclst = []
destlst = []

for filename in os.listdir(sourceimg):
    sourcepath = os.path.join(sourceimg,filename)
    source_img = cv2.imread(sourcepath)
    if source_img is not None:
        srclst.append(sourcepath)

for files in os.listdir(destimg):
    destpath = os.path.join(destimg,files)
    dest_img = cv2.imread(destpath)
    if dest_img is not None:
        destlst.append(destpath)
#print(srclst,destlst)        
        
print("Enter 1 for Encoding Based Test")
print("Enter 2 for Distance Based Test")
#print("Enter 3 for Combined Test")
n = "y"
while(n == "y"):
    choice = int(input("Enter Your Choice:"))
    if choice == 1:
        enc_test(srclst,destlst)
    elif choice == 2:
        noc = input("Enter the number of candidates that needs to be shortlisted")
        if noc<len(srclst):
            perc_test(srclst,destlst)
        else:
            print("Number entered is greater than total number of candidates")
    elif choice == 3:
        combine_test(srclst,destlst)
        
        
    s = input("Do you want to continue(y/n)")
    if(s=="n"):
        break    
    


