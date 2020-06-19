import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data=pd.read_csv("car.data")

pe=preprocessing.LabelEncoder()
buy=pe.fit_transform(list(data["buying"]))
maint=pe.fit_transform(list(data["maint"]))
door=pe.fit_transform(list(data["door"]))
persons=pe.fit_transform(list(data["persons"]))
lug=pe.fit_transform(list(data["lug_boot"]))
safety=pe.fit_transform(list(data["safety"]))
cls=pe.fit_transform(list(data["class"]))

predict="class"

x=list(zip(buy,maint,door,persons,lug,safety))
y=list(cls)

x_tr,x_ts,y_tr,y_ts=sklearn.model_selection.train_test_split(x,y,test_size=0.1)

model= KNeighborsClassifier(n_neighbors=9)
model.fit(x_tr,y_tr)
ac=model.score(x_ts,y_ts)
print(ac)

predicted=model.predict(x_ts)
names=["unacc","acc","good","vgood"]
for i in range(len(predicted)):
    print("Prediction: ",names[predicted[i]],"Data ",x_ts[i],"Actual: ",names[y_ts[i]])
    n=model.kneighbors([x_ts[i]],9,True)
    print("N: ",n)
