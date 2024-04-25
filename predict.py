import torch
import pywt
import cv2
import numpy as np
import base64

MODEL=torch.load('models/entire_model_wavelet')
device='cuda'
face_cascade = cv2.CascadeClassifier('opencv/haarcascades/haarcascade_frontalface_default.xml')

def get_cropped_face(image):
    if image is not None:
        gray_img=cv2.cvtColor( image,cv2.COLOR_BGR2GRAY )
        faces = face_cascade.detectMultiScale(gray_img, 1.4, 5)
        if  faces.any():
            for (x,y,w,h) in faces:
                roi_gray = gray_img[y:y+h, x:x+w]
                return roi_gray

def w2d(img, mode='db1', level=5):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    # imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)   
    imArray /= 255;
    # compute coefficients 
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0;  

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)

    return imArray_H

def predict_age(image):
    crop_img=get_cropped_face(image)
    if crop_img.any():
        res_img=cv2.resize(crop_img,(50,50))
        wt_img=w2d(res_img)
        inp=torch.tensor(wt_img.reshape(1,50,50),dtype=torch.float32).to(device)
        op=MODEL(inp.unsqueeze(0))
        return torch.argmax(op).item()
