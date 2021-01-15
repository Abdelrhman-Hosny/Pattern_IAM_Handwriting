import cv2 
import numpy as np

# I understood something wrong about this function , but just in case that what I did was right
# I will keep it commented and test both of them

# def extract_writer_width(line , f2):

#     # From Paper : Writer Identification Using Text Line Based Features
#     # Authors : U.-V. Marti, R. Messerli and H. Bunke


#     # Input has to be thresholded
#     _ , threshed = cv2.threshold(line , 200 , 255 , cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

#     projection = np.sum(threshed/np.max(threshed),1)

#     # diff(x) = [x(1)-x(0)  x(2)-x(1) .......  x(n)-x(n-1)]
#     diff = np.diff(threshed/np.max(threshed),axis = 1)

#     # Sum of absolute value of diff return 1 if a change happened from one pixel to the next 
#                                 # returns 0 if no change happens from one pixel to the next
#     Changes  = np.sum(np.abs(diff),axis=1)

#     # MaxChangesIndex picks the row with the most ones (i.e most changes)
#     MaxChangesIndex = np.argmax(Changes)

#     # gets the indices that are not zeros in the row of max changes
#     # and subtracts those indices from each other to get the width of the lines
        
#     WidthEachLine = np.diff( np.nonzero(diff[MaxChangesIndex, :]))

#     #the median of the difference between  the changes from white to black then to white again
#     f1 = np.median(WidthEachLine)

#     # If we finish the first function we will divide f2 (from paper) by width
#     # to account for small spaces.

#     f3 = f2/f1

#     return  f1  , f3

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
    
    # gets the indices that are 1 in the row of max changes which means they changed from white to black to white again
    # and subtracts those indices from each other to get the width of the lines
    IndicesRowChange = np.argwhere(diff[MaxChangesIndex, :]  == 1).T[0]
    
    WidthEachLine = np.diff(IndicesRowChange)

    #the median of the difference between  the changes from white to black then to white again
    f1 = np.median(WidthEachLine)

    # If we finish the first function we will divide f2 (from paper) by width
    # to account for small spaces.

    f3 = f2/f1

    return  f1  , f3

def extract_base_line(lineImg):

    # From Paper : Writer Identification Using Text Line Based Features
    # Authors : U.-V. Marti, R. Messerli and H. Bunke

    # Line image has to be threshed in order to reduce noise
    _ , threshed = cv2.threshold(lineImg , 200 , 255 ,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

    # Image is horizontally projected 
    projection = np.sum(threshed/np.max(threshed),1)

    #if you check the histogram in Figure 2 in the paper ,
    # you'll see that the histogram can be approximated as 
    # a normal distribution that has upper baseline = mu -std
    # and lower baseline = mu + std

    ProjectionRows = np.arange(0,projection.shape[0])

    mu = np.average(ProjectionRows,weights=projection)
    std_dev= np.sqrt(np.average(np.square(ProjectionRows - mu),weights=projection))

    

    # Variables are named according to the paper so if you dont understand
    # feel free to have a look ;)

    upper_baseline = int(mu - std_dev)
    lower_baseline = int(mu + std_dev)


    ProjNotZero = np.nonzero(projection)


    topline = np.min(ProjNotZero)
    bottomline = np.max(ProjNotZero)



    f1 = abs(topline - upper_baseline)
    f2 = abs(upper_baseline - lower_baseline)
    f3 = abs(lower_baseline - bottomline)

    f4 = f1/f2
    f5 = f1 / f3
    f6 = f2/f3

    return np.array([f1,f2,f3,f4,f5,f6])


def extract_slant_features (line ):
    ######## These set of features are extracted based on This Paper : ##########
    ####### Writer Identification Using Edge-Based Directional Features #########


    #apply canny filter to get image edges
    edges = 255 - cv2.Canny(line, 0, 250)
        
    # plt.figure(figsize=(10,20))
    # io.imshow(cv2.Canny(line, 0, 250),cmap="gray")
    # plt.figure(figsize=(10,20))
    # io.imshow(edges,cmap="gray")
    line_inverted = 1 - (edges / np.max(line))
    features = []
    
    process_line = line_inverted [2:line.shape[0]-2,2:line.shape[1]-2 ]
    # plt.figure(figsize=(10,20))
    # io.imshow(line_inverted,cmap="gray")
    # plot_h_proj(line_inverted)
    # get black pixels
    y_mask, x_mask = np.where(process_line == 1)
    # get slant features
    
    features.append(np.sum(line_inverted[y_mask +1, x_mask]))
    features.append(np.sum(line_inverted[y_mask +2, x_mask]))

    features.append(np.sum(line_inverted[y_mask - 1, x_mask]))
    features.append(np.sum(line_inverted[y_mask - 2, x_mask]))
    # print(line_inverted[y_mask - 1, x_mask])
    features.append(np.sum(line_inverted[y_mask + 1, x_mask + 1]))
    features.append(np.sum(line_inverted[y_mask + 1, x_mask + 2]))
    features.append(np.sum(line_inverted[y_mask + 2, x_mask + 1]))
    features.append(np.sum(line_inverted[y_mask + 2, x_mask + 2]))

    features.append(np.sum(line_inverted[y_mask - 1, x_mask + 1]))
    features.append(np.sum(line_inverted[y_mask - 1, x_mask + 2]))
    features.append(np.sum(line_inverted[y_mask - 2, x_mask + 1]))
    features.append(np.sum(line_inverted[y_mask - 2, x_mask + 2]))
  
    return np.array(features/np.sum(features))



def extract_all_features(line,n_features):

    # X = np.empty((0,n_features),dtype=np.float64)

    X = extract_base_line(line)
    X = np.hstack((X,extract_writer_width(line,X[1])))
    X = np.hstack((X,extract_slant_features(line)))

    return X