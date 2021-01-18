from Pipeline import *
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
pipe = Pipeline(r'F:\Pattern_Proj_Dataset\Form\Tests',17,lr)


pipe.LoopOverDirectories()
