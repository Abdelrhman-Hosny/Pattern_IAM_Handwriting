
import os
import numpy as np
import time


# Dummy functions until the real ones are created
def dummy_extract_feature(image):
    return None
def Model_fit(X,y):
    pass
def Model_predict(X):
    pass

#End of Dummy Functions


def createFileFromArr(arr,path):
    f = open(path,"w+")

    for elem in arr:
        f.write('{} \n'.format(round(elem,2)))

    f.close()

def RunModel(path,model,**kwargs): #kwargs is for model hyperparameters
    
    #in case this is a neural network , we would have to add the compiling and the creating
    # other models from sklearn , we will jsut create a new object and 

        
    # creating variables
    Time_Pred = []
    Test_Pred = []
    y_train = np.array([[1],    #hard coded as all the files will have the same format
                        [1],
                        [2],
                        [2],
                        [3],
                        [3]])



    path_dir = os.path.join(path,'data')
    root , subdir , subfiles = next(os.walk(path_dir))

    for directory in subdir:
        root2 , ClassesDirectories , test_image = next(os.walk(os.path.join(root,directory)))
        test_image_path = os.path.join(root2,test_image[0])
    
        # Setting start time for each directory
        start_time = time.time()
        # A new feature vector
        X = dummy_extract_feature(test_image_path)
        # ClassesDirectories is the file that has the directories 1 , 2 ,3 and test_image.png
        # so each ClassDir contains 2 images and
        for WriterID , ClassDir in enumerate(ClassesDirectories , start = 1):
            # each subdir has the 
            root3 , _ , images = next(os.walk(os.path.join(root2,ClassDir)))
            
            for image in images:
                img_features = dummy_extract_feature(image)
                np.vstack((X,img_features))

        #At the end of this loop 
        # X should have at index 0 , features of test case
        # X[1:,:] has the features of writer 1,2,3
        
        Model_fit(X[1:,:],y_train)

        # Adding prediction and time to each of their respective arrays
        Test_Pred.append( Model_predict(X[0]) )
        Time_Pred.append(time.time() - start_time)

    # Print Time and Results to a file
    createFileFromArr(Time_Pred,os.path.join(path,'time.txt'))
    createFileFromArr(Test_Pred,os.path.join(path,'results.txt'))
