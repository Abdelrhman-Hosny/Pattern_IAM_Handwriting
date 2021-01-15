from segmentation import *
import numpy as np
import os
import time
# from segmentation import *
from FeatureExtraction import extract_all_features
from scipy.stats import mode
import cv2
from sklearn.preprocessing import StandardScaler



class Pipeline:

    def __init__(self,path,num_features,model):
        
        # Benchmarks arrays
        self.Time_Pred = []
        self.Test_Pred = []

        self.modelSave = []
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
            f.write('{} \n'.format(elem[0]))

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
            X = np.vstack((X , self.extract_all_feat(line)))


        # Repeat the following for the train data
        for WriterID , ClassDir in enumerate(ClassesDirectories,start=1):
            root3 , _ , images = next(os.walk(os.path.join(rootDir,ClassDir)))

            for image in images:
                Lines , n_lines = self.ImgToLines(os.path.join(root3,image))
                y_train = np.vstack(   (y_train ,np.ones((n_lines,1)) * WriterID )  )
                for line in Lines:
                    X = np.vstack((X,self.extract_all_feat(line)))

                    

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
        return self.modelSave

    def ImgToLines(self,imgPath):
        
        gs_img = cv2.imread(imgPath,0)
        gs_img_croped_v = gs_img[:,gs_img.shape[1]//4: gs_img.shape[1]-gs_img.shape[1]//4 ]
        
        b_img = binarize_gray_img(gs_img)

        b_img_cropped_h = get_writtig_area(gs_img_croped_v,b_img)

        line_limits = get_writing_lines_limits(b_img_cropped_h)

        Lines  = []
        for limit in line_limits:

           Lines.append(b_img_cropped_h[limit[0]:limit[1] ,:])
    
        return Lines , len(Lines)

        
    
    def FitAndPredict(self,X_train,y_train,X_test):

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        self.model.fit(X_train,np.ravel(y_train))


        pred = mode(self.model.predict(scaler.transform(X_test)))

        

        return pred

    def extract_all_feat(self,line):
        return extract_all_features(line,self.num_features)