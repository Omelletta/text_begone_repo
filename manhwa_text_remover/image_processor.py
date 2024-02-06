import cv2
import numpy as np

def process_image(img):
   
    p_img = cv2.resize(img, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)   
    p_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    p_img = cv2.medianBlur(p_img, 5)
    p_img = cv2.threshold(p_img, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    #p_img = cv2.GaussianBlur(p_img, (3, 3), 1)
    
    #p_img = cv2.adaptiveThreshold(p_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    p_img = cv2.dilate(p_img, kernel, iterations=1)
    
    return p_img
