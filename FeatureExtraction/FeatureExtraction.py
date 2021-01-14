import cv2 
import numpy as np



def extract_writer_width(line , f2):

    # From Paper : Writer Identification Using Text Line Based Features
    # Authors : U.-V. Marti, R. Messerli and H. Bunke


    # Input has to be thresholded
    _ , threshed = cv2.threshold(line , 200 , 255 , cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

    projection = np.sum(threshed/np.max(threshed),1)

    # diff(x) = [x(1)-x(0)  x(2)-x(1) .......  x(n)-x(n-1)]
    diff = np.diff(threshed/np.max(threshed),axis = 1)

    # Sum of absolute value of diff return 1 if a change happened from one pixel to the next 
                                # returns 0 if no change happens from one pixel to the next
    Changes  = np.sum(np.abs(diff),axis=1)

    # MaxChangesIndex picks the row with the most ones (i.e most changes)
    MaxChangesIndex = np.argmax(Changes)

    # gets the indices that are not zeros in the row of max changes
    # and subtracts those indices from each other to get the width of the lines
        
    WidthEachLine = np.diff( np.nonzero(diff[MaxChangesIndex, :]))

    #the median of the difference between  the changes from white to black then to white again
    f1 = np.median(WidthEachLine)

    # If we finish the first function we will divide f2 (from paper) by width
    # to account for small spaces.

    # f3 = f2/f1

    return  f1  #, f3



