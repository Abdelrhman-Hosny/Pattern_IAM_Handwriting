from Pipeline import *
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
pipe = Pipeline(r'F:\Pattern_Proj_Dataset\Form\Tests',17,rfc)


pipe.LoopOverDirectories()
