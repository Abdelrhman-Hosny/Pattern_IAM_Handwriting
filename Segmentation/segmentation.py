import cv2
import numpy as np
import glob
import time
import skimage.io as io
from skimage.filters import *
import matplotlib.pyplot as plt
from skimage.morphology import *
from skimage.exposure import equalize_hist
from skimage.filters import gaussian,median
import math
from scipy  import signal 

def binarize_gray_img(gs_img):
    # this function return binary image and invert image 
    blur_gauss = cv2.GaussianBlur(gs_img,(5,5),0)
    _, binary = cv2.threshold(blur_gauss , 150 , 255 , cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    return binary

def extract_lines(bin_img):
    # this function return image with only horizontale lines  
    n = 3
    s_e1 = np.zeros((n, int(0.3 * bin_img.shape[1])))
    s_e1[n//2] += 1
    closed = binary_opening(bin_img, s_e1)
    return closed

def remove_duplicates(lines_img):
    # this function remove duplicates 
    column = lines_img[:, lines_img.shape[1]//2]
    line_indexes = np.array(np.where(column == 1))
    unique_line_indexes = []
    maxThick = 0
    line_thickness = 1

    for i, line_index in np.ndenumerate(line_indexes):
        if(not line_index == line_indexes[0, (i[1]-1)] + 1) and (not line_index == line_indexes[0, (i[1]-1)] + 2):
            if (maxThick < line_thickness):
                maxThick = line_thickness
            unique_line_indexes.append(line_index)
            line_thickness = 1
        else:
            line_thickness = line_thickness + 1
    return np.array(unique_line_indexes), maxThick

    
def get_lines_positions(bin_img):
    # this function return unique indexes of the horizontal lines  
    staff_lines_img = extract_lines(bin_img)
    unique_indexes, _ = remove_duplicates(staff_lines_img)
    return unique_indexes

def get_writtig_area(bin_img):
    #this function assume perfect input only 3 lines are in the image and crop
    lines  =get_lines_positions(bin_img)
    print(lines)
    croped =bin_img[lines[1]+10:lines[2]-3,: ]
    return croped

def get_writing_lines_limits(croped):
    h_proj = np.sum(croped,axis=1)
    mean = np.mean(h_proj)
    h_proj[h_proj<mean]=0
    busy_indexes =np.where(h_proj!=0)[0]
    diff =np.diff(busy_indexes)
    gaps  =np.where(diff>50)[0]
    limits = [busy_indexes[0]]
    for gap in gaps :
        limits.append(busy_indexes[gap])
        limits.append(busy_indexes[gap+1])
    limits.append(busy_indexes[-1])
    line_valid_limits =[]
    for i  in range (0,len(limits),2) :
        tot_height =croped.shape[0]
        height =limits[i+1]-limits[i]
        start= max(limits[i]-height//4,0)
        end= min(limits[i+1]+height//4,tot_height)
        if (end > start):
            line_valid_limits.append((start,end))
            # io.imshow(croped[start:end,:])
    return line_valid_limits
  