from Pipeline import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
pipe = Pipeline(r'F:\Test Set Sample',20,lr)


model_coef = pipe.LoopOverDirectories()

for i in model_coef:
    print(i)