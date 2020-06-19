import tensorflow as tf
import sklearn
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data=pd.read_csv("student-mat.csv", sep=";")

data=data[["G1","G2","G3","studytime","failures","absences"]]

predict="G3"
Z=np.array(data.drop([predict],1))
W=np.array(data[predict])
x_tr,x_ts,y_tr,y_ts=sklearn.model_selection.train_test_split(Z,W,test_size=0.17)

best=0
for _ in range(35):
    x_tr,x_ts,y_tr,y_ts=sklearn.model_selection.train_test_split(Z,W,test_size=0.17)

    linear=linear_model.LinearRegression()
    linear.fit(x_tr,y_tr)
    ac=linear.score(x_ts,y_ts)
    print(ac)
    if ac>best:
        best=ac
        with open("Studentmodel.pickle","wb") as f:
            pickle.dump(linear, f)

pic_in=open("Studentmodel.pickle", "rb")
linear=pickle.load(pic_in)

print("Co: ",linear.coef_)
print("Intercept: ",linear.intercept_)

predictions=linear.predict(x_ts)

for x in range(len(predictions)):
    print(predictions[x],x_ts[x],y_ts[x])

p="G1"
style.use("ggplot")
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Grade")
pyplot.show()
