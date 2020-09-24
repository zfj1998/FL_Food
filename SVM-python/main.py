from SVR import OnlineSVR
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from score import get_metrics

iris = datasets.load_iris()
x = iris.data
y = iris.target
x_train,x_test,y_train,y_test = train_test_split(x,y)

iris_svr = OnlineSVR(numFeatures=x.shape[1], C=1, eps=0.1, kernelParam=30)
for i in range(x_train.shape[0]):
    print('%%%%%%%%%%%%%%% Data point {0} %%%%%%%%%%%%%%%'.format(i))
    iris_svr.learn(x_train[i, :], y_train[i])

y_predict = iris_svr.predict(x_test)
score = get_metrics(y_test,y_predict)
print(score)
print('y_predict:',y_predict,y_test)