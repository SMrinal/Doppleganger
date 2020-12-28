import cv2
import os
import numpy as np
import dlib
import face_recognition
from IPython.display import Image, display
from statistics import mean
import random

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

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
    att = []
    mtt = []
    f_count = 0
    print("Encoding Test Initiated")
    print("Please Wait....")
    for i in range(0 , len(s)):
        temp_enc = []
        for j in range(0 , len(d)):
            enc_res = face_compare_enc(s[i],d[j])
            temp_enc.append(enc_res)
        temp_result = layer1(temp_enc)
        enclst.append(temp_result)
    for ml in range(0,len(enclst)):
        if(enclst[ml] == False):
            f_count = f_count+1
    if f_count == len(enclst):
        print("Encoding Test Yeilded No result")
        print("________________________________________________________________________________________________________________")
        print("Face Distance Test Initiating")
        c = perc_test(s ,d , 2);
        for l in range(0, len(c)):
            display(Image(filename=c[l]))
            mtt.append(c[l])
        print("________________________________________________________________________________________________________________")
        print("Initiating Morph Test Phase")    
        morph_test(mtt,d)    
            
    else:
        for k in range(0,len(s)):
            if(enclst[k]== True):
                att.append(s[k])
                display(Image(filename=s[k]))
        print("________________________________________________________________________________________________________________")        
        print("Face Distance Test Initiating.....")        
        c = perc_test(att ,d , 2);
        print("Images with highest resemblance:")
        for l in range(0, len(c)):
            display(Image(filename=c[l]))
            mtt.append(c[l])
        print("________________________________________________________________________________________________________________")
        print("Initiating Morph Test Phase")    
        morph_test(mtt,d)          
        
    
    
def perc_test(srclst,destlst,x):
    
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
    return perclist 

def morph_test(a , lst):
    #x = input("Enter the path of the Image of the Subject")
    f = ".jpeg"
    o = {}
    g = []
    sourceimg = "Desktop/test"
    sourceimg_1 = "Desktop/test1"
    n = random.randint(0,(len(lst)+1))
    s1 = []
    
    for i in range(0,2):
        s = []
        for l in range(0,len(lst)):
            h = morph(a[i],lst[l])
            cv2.imwrite(("Desktop/test/"+str(i)+str(l)+f) , h)
            cv2.imwrite(("Desktop/test1/"+str(i)+str(l)+f) , h)
        for filename in os.listdir(sourceimg):
            sourcepath = os.path.join(sourceimg,filename)
            source_img = cv2.imread(sourcepath)
            if source_img is not None:
                s.append(sourcepath)        
        v = perc_test(s,lst,1)
        for j in range(0,len(s)):
            os.remove(s[j])
        g.append(v[0])
    for p in range(0,len(g)):
        ab = g[p]
        cd = ab.replace("test" , "test1")
        g[p] = cd
        
    for k in range(0,len(g)):
        o[a[k]] = g[k]    
    z = perc_test(g,lst,1)
    for key ,value in o.items() :
        if(value == z[0]):
            print("Best Candidate For Face Morphing is :")
            display(Image(filename=key))
            print("Morphed Image:")
            display(Image(filename = value))
    for fi in os.listdir(sourceimg_1):
        sourcepath_1 = os.path.join(sourceimg_1,fi)
        source_img_1 = cv2.imread(sourcepath_1)
        if source_img_1 is not None:
            s1.append(sourcepath_1)        
    for w in range(0,len(s1)):
        os.remove(s1[w])
    os.rmdir('Desktop/test')
    os.rmdir('Desktop/test1')

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

def morph(a,b):
    img = cv2.imread(a)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_gray)
    img2 = cv2.imread(b)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    height, width, channels = img2.shape
    img2_new_face = np.zeros((height, width, channels), np.uint8)





    faces = detector(img_gray)
    for face in faces:
        landmarks = predictor(img_gray, face)
        landmarks_points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))



        points = np.array(landmarks_points, np.int32)
        convexhull = cv2.convexHull(points)
        
        cv2.fillConvexPoly(mask, convexhull, 255)

        face_image_1 = cv2.bitwise_and(img, img, mask=mask)

 
        rect = cv2.boundingRect(convexhull)
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(landmarks_points)
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32)

        indexes_triangles = []
        for t in triangles:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])


            index_pt1 = np.where((points == pt1).all(axis=1))
            index_pt1 = extract_index_nparray(index_pt1)

            index_pt2 = np.where((points == pt2).all(axis=1))
            index_pt2 = extract_index_nparray(index_pt2)

            index_pt3 = np.where((points == pt3).all(axis=1))
            index_pt3 = extract_index_nparray(index_pt3)

            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                triangle = [index_pt1, index_pt2, index_pt3]
                indexes_triangles.append(triangle)



    # Face 2
    faces2 = detector(img2_gray)
    for face in faces2:
        landmarks = predictor(img2_gray, face)
        landmarks_points2 = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points2.append((x, y))


        points2 = np.array(landmarks_points2, np.int32)
        convexhull2 = cv2.convexHull(points2)

    lines_space_mask = np.zeros_like(img_gray)
    lines_space_new_face = np.zeros_like(img2)
    
    for triangle_index in indexes_triangles:
        tr1_pt1 = landmarks_points[triangle_index[0]]
        tr1_pt2 = landmarks_points[triangle_index[1]]
        tr1_pt3 = landmarks_points[triangle_index[2]]
        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)


        rect1 = cv2.boundingRect(triangle1)
        (x, y, w, h) = rect1
        cropped_triangle = img[y: y + h, x: x + w]
        cropped_tr1_mask = np.zeros((h, w), np.uint8)


        points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                           [tr1_pt2[0] - x, tr1_pt2[1] - y],
                           [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_tr1_mask, points, 255)


        cv2.line(lines_space_mask, tr1_pt1, tr1_pt2, 255)
        cv2.line(lines_space_mask, tr1_pt2, tr1_pt3, 255)
        cv2.line(lines_space_mask, tr1_pt1, tr1_pt3, 255)
        lines_space = cv2.bitwise_and(img, img, mask=lines_space_mask)

 
        tr2_pt1 = landmarks_points2[triangle_index[0]]
        tr2_pt2 = landmarks_points2[triangle_index[1]]
        tr2_pt3 = landmarks_points2[triangle_index[2]]
        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)


        rect2 = cv2.boundingRect(triangle2)
        (x, y, w, h) = rect2

        cropped_tr2_mask = np.zeros((h, w), np.uint8)

        points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                            [tr2_pt2[0] - x, tr2_pt2[1] - y],
                            [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

        
        points = np.float32(points)
        points2 = np.float32(points2)
        M = cv2.getAffineTransform(points, points2)
        warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

       
        img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
        img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
        _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

        img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
        img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area



  
    img2_face_mask = np.zeros_like(img2_gray)
    img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
    img2_face_mask = cv2.bitwise_not(img2_head_mask)


    img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
    result = cv2.add(img2_head_noface, img2_new_face)

    (x, y, w, h) = cv2.boundingRect(convexhull2)
    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

    seamlessclone = cv2.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)
    



    return seamlessclone
    
        
#Main
sourceimg = ''
destimg = ''
while((sourceimg == '') or (destimg =='')):
    sourceimg = input("Enter the path of the image folder:")
    destimg = input("Enter the path of Subject folder:")
    if((sourceimg=='') or (destimg=='')):
        print("Invalid Input, Try Again")
        print("________________________________________________________________________________________________________________")
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
createFolder('Desktop/test')
createFolder('Desktop/test1')
print("________________________________________________________________________________________________________________")
enc_test(srclst,destlst)        
    
