# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 15:18:04 2019

@author: sankar.chakraborty
"""

import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime

####################################################################
#               GLOBALS                                            #
####################################################################
MAX_FEATURES = 1000
GOOD_MATCH_PERCENT = 0.15
font = cv2.FONT_HERSHEY_SIMPLEX
AlignedForm_Colored=[]
X=0
Y=0
W=0
H=0

#Load Precompiled Model
model = tf.keras.models.load_model('../Trained-Models/LeNet5_MNIST_ImageGenerator_Custom.model')    
# Read reference image
refFilename = './Inputs/Bank Slip reference.jpg'
# Read image to be aligned
imFilename = "../Inputs/input.jpg"
    


####################################################################
#               show_wait_destroy() Function for Display           #
####################################################################
def show_wait_destroy(winname, img):
    cv2.imshow(winname, img)
    cv2.moveWindow(winname, 500, 0)
    cv2.waitKey(0)
    cv2.destroyWindow(winname)

####################################################################
#               alignImages() Homography Function for Alignment    #
#################################################################### 
def alignImages(im1, im2):
 
  # Convert images to grayscale
  im1Gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
  im2Gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
   
  # Detect ORB features and compute descriptors.
  orb = cv2.ORB_create(MAX_FEATURES)
  keypoints1, descriptors1 = orb.detectAndCompute(im1Gray, None)
  keypoints2, descriptors2 = orb.detectAndCompute(im2Gray, None)
   
  # Match features.
  matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
  matches = matcher.match(descriptors1, descriptors2, None)
   
  # Sort matches by score
  matches.sort(key=lambda x: x.distance, reverse=False)
 
  # Remove not so good matches
  numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
  matches = matches[:numGoodMatches]
 
  # Draw top matches
  imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
  cv2.imwrite("matches.jpg", imMatches)
   
  # Extract location of good matches
  points1 = np.zeros((len(matches), 2), dtype=np.float32)
  points2 = np.zeros((len(matches), 2), dtype=np.float32)
 
  for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt
   
  # Find homography
  h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
 
  # Use homography
  height, width, channels = im2.shape
  im1Reg = cv2.warpPerspective(im1, h, (width, height))
   
  return im1Reg, h

####################################################################
#                  HandWrittenDigitRecognition()                   #
#           Function to Find Contours and predict Digits           #
#################################################################### 

def HandWrittenDigitRecognition(image):
    morph = image.copy()
    hi,wi= image.shape
    hi=hi

    
    # smooth the image with alternative closing and opening
    # with an enlarging kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
    contours, hierarchy= cv2.findContours(morph,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    show_wait_destroy("morph", morph)

    
    recognizedDigit=""
    ContourPadding=1
    digitsCountours=[]
    
    #Get the Digits Contours Seperated for Recognition
    selected_contour=[]
    
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        x1, y1, w1, h1 = cv2.boundingRect(approx)
        print(" Selected Contour:"+",x="+str(x1)+",y="+str(y1)+",w="+str(w1)+",h="+str(h1))
        if h1>15: 
            selected_contour.append(contour)
            print(" Selected Contour:"+",x="+str(x1)+",y="+str(y1)+",w="+str(w1)+",h="+str(h1))
        else:
            print(" Excluded Contour:"+",x="+str(x1)+",y="+str(y1)+",w="+str(w1)+",h="+str(h1))
        
    
    
    for contour in selected_contour:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        x1, y1, w1, h1 = cv2.boundingRect(approx)
        IsInside=False
        for c in selected_contour:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            x2, y2, w2, h2 = cv2.boundingRect(approx)
            if x1>=x2 and x1<=x2+w2 and y1>y2 and y1<y2+h2:
                IsInside=True
                break    
        print("Contour InInside:"+str(IsInside))
        if IsInside==False:
            cv2.rectangle(AlignedForm_Colored, (X+x1-ContourPadding, Y+y1-ContourPadding), (X+x1+w1+ContourPadding, Y+y1+h1+ContourPadding), (0, 0, 255), 1);
            digitsCountours.append([x1-ContourPadding,y1-ContourPadding,w1+ContourPadding,h1+ContourPadding])
        else:
            cv2.rectangle(AlignedForm_Colored, (X+x1-ContourPadding, Y+y1-ContourPadding), (X+x1+w1+ContourPadding, Y+y1+h1+ContourPadding), (255, 0, 0), 1);
            
    
    #Sort the Digit Countours for Sequential Recognition
    digitsCountours=sorted(digitsCountours, key = lambda x:x[0])        
    
    for c in digitsCountours: 
        x1,y1,w1,h1=c       
        #cv2.rectangle(image, (x1-ContourPadding, y1-ContourPadding), (x1+w1+ContourPadding, y1+h1+ContourPadding), (0, 0, 255), 1);
        digit_array=morph[y1:y1+h1,x1:x1+w1]
        x_pad=int((w1+h1)/4)
        digit_array=np.pad(digit_array, (((x_pad,x_pad),(x_pad,x_pad))), 'constant')
        
        digit_array=cv2.resize(digit_array,(32,32),interpolation = cv2.INTER_AREA)
        
        prediction_array=digit_array.reshape(-1,32,32,1)
        prediction=model.predict(prediction_array)
        recognizedDigit=recognizedDigit+str(np.argmax(prediction))
        print("Recognized:"+str(np.argmax(prediction))+",x="+str(x1)+",y="+str(y1)+",w="+str(w1)+",h="+str(h1),"x_pad="+str(x_pad))
        cv2.putText(AlignedForm_Colored,str(np.argmax(prediction)),(X+x1+int(w1/2),Y+y1-ContourPadding-10), font, 1,(0,0,255),2,cv2.LINE_AA)

    return recognizedDigit




###########################################################################
#               MAIN MODULE                                               #
###########################################################################
if __name__ == '__main__':
    
    print("Reading reference image : ", refFilename)
    imReference = cv2.imread(refFilename, cv2.IMREAD_COLOR)
     
    print("Reading image to align : ", imFilename);  
    Filled_Form_Colored = cv2.imread(imFilename, cv2.IMREAD_COLOR)
       
    print("Aligning images ...")
    # Registered image will be resotred in imReg. 
    # The estimated homography will be stored in h. 
    AlignedForm_Colored, h = alignImages(Filled_Form_Colored, imReference) 
    ############################################################
    #               Image Pre-Processing                       #
    ############################################################
    #HSV Way
    # Convert BGR to HSV
    hsv = cv2.cvtColor(AlignedForm_Colored, cv2.COLOR_BGR2HSV)
    
    # define range of black color in HSV
    lower_val = np.array([60,45,0],np.uint8)
    upper_val = np.array([150,255,255],np.uint8)
    
    # Threshold the HSV image to get only black colors
    mask = cv2.inRange(hsv, lower_val, upper_val)
    # with an enlarging kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    show_wait_destroy("mask", mask)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(AlignedForm_Colored,AlignedForm_Colored, mask= mask)
    # invert the mask to get black letters on white background
    res2 = cv2.bitwise_not(mask)
    Processed_Form=cv2.bitwise_not(res2)
    
    
    ############################################################
    #               Crop By Region and Detect Digits           #
    ############################################################    
    
    #A/c Number Rectangle x=182,y=695,w=473, h=95
    X,Y,W,H=165,700,493,90
    #X,Y,W,H=180,650,485,150
    AccountNumberImage=Processed_Form[Y:Y+H,X:X+W]
    accountNo=HandWrittenDigitRecognition(AccountNumberImage)
    print("Account Number:",accountNo)
    
    #Employer's Code Rectangle x=30,y=435,w=420, h=70
    X,Y,W,H=35,430,420,80
    EmployerCodeImage=Processed_Form[Y:Y+H,X:X+W]
    EmployerCode=HandWrittenDigitRecognition(EmployerCodeImage)
    print("Employee Code:",EmployerCode)   
     
    #SOCSO RM Rectangle x=182,y=695,w=473, h=95
    X,Y,W,H=1260,110,300,60
    SOCSOImage=Processed_Form[Y:Y+H,X:X+W]
    SOCSO=HandWrittenDigitRecognition(SOCSOImage)
    SOCSO=str(float(SOCSO)/100.0)
    print("SOCSO:RM ",SOCSO) 
    
    #Cheque amount Rectangle x=182,y=695,w=473, h=95
    X,Y,W,H=1258,220,300,60
    ChequeAmountImage=Processed_Form[Y:Y+H,X:X+W]
    ChequeAmount=HandWrittenDigitRecognition(ChequeAmountImage)
    print("Cheque Amount:",ChequeAmount) 
    
    #Cheque Number Rectangle x=182,y=695,w=473, h=95
    X,Y,W,H=1035,220,230,60
    ChequeNumberImage=Processed_Form[Y:Y+H,X:X+W]
    ChequeNumber=HandWrittenDigitRecognition(ChequeNumberImage)
    print("ChequeNumber:",ChequeNumber)
    
    #Cheque Number Rectangle x=182,y=695,w=473, h=95
    X,Y,W,H=1255,300,300,120
    JUMLAHAmountImage=Processed_Form[Y:Y+H,X:X+W]
    JUMLAHAmount=HandWrittenDigitRecognition(JUMLAHAmountImage)
    print("JUMLAH Amount:RM",JUMLAHAmount)

    show_wait_destroy("AlignedForm_Colored",AlignedForm_Colored)
    now = datetime.now()
    dt_string = now.strftime("%d-%B-%Y_%H-%M-%S")
    cv2.imwrite('./Output_'+dt_string+'.jpg',AlignedForm_Colored)
    
    