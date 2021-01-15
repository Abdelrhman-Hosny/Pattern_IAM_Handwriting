
import numpy as np
import os
import time
from segmentation import *
from FeatureExtraction import extract_all_features
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import mode




class Pipeline:

    def __init__(self,path,num_features,model):
        
        # Benchmarks arrays
        self.Time_Pred = []
        self.Test_Pred = []

        # num of features extracted by model
        self.num_features = num_features

        # path to run the model on
        self.path = path

        # set model that'll be run in here
        self.model = model


    def createFileFromArr(self):
        # Write the time of each iteration
        f = open(os.path.join(self.path,'time.txt'),"w+")

        for elem in self.Time_Pred:
            f.write('{} \n'.format(round(elem,2)))

        f.close()

        #Write the result of each iteration

        f = open(os.path.join(self.path,'results.txt'),"w+")

        for elem in self.Test_Pred:
            f.write('{} \n'.format(round(elem,2)))

        f.close()

    def ExtractFeaturesFromDirectory(self,rootDir,ClassesDirectories,test_image_path):

        # time starts before preprocessing and ends after prediction
        self.start_time = time.time()

        X = np.empty((0,self.num_features))
        y_train = np.empty((0,1))

        # Extract X from test_image

        # Preprocessing Phase
        Lines , n_line_test = self.ImgToLines(os.path.join(rootDir,test_image_path))

        # Feature Extraction phase
        for line in Lines:
            X = np.vstack((X , extract_all_features(line)))


        # Repeat the following for the train data
        for WriterID , ClassDir in enumerate(ClassesDirectories,start=1):
            root3 , _ , images = next(os.walk(os.path.join(rootDir,ClassDir)))

            for image in images:
                Lines , n_lines = self.ImgToLines(os.path.join(root3,image))
                for line in Lines:
                    X = np.vstack((X,extract_all_features(line)))

                    y_train = np.vstack(   (y_train ,np.ones((n_lines,1)) * WriterID )  )

            X_train = X[n_line_test:,:]
            X_test = X[:n_line_test,:]

            return X_train  , y_train , X_test


        

    
    def LoopOverDirectories(self):
        
        path_dir = os.path.join(self.path,'data')
        root , subdir , subfiles = next(os.walk(path_dir))

        for directory in subdir:
            root2 , ClassesDirectories , test_image = next(os.walk(os.path.join(root,directory)))
            
            X_train , y_train , X_test = self.ExtractFeaturesFromDirectory(root2,ClassesDirectories,test_image[0])

            pred = self.FitAndPredict( X_train , y_train , X_test)

            self.Test_Pred.append(pred)
            self.Time_Pred.append(time.time() - self.start_time)

        self.createFileFromArr()

    def ImgToLines(self,imgPath):
        
        return None, None
    
    def FitAndPredict(self,X_train,y_train,X_test):

        self.model.fit(X_train,y_train)

        pred = mode(self.model.predict(X_test))

        return pred