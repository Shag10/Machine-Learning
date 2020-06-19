import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer=datasets.load_breast_cancer()
'''print(cancer.feature_names)
print(cancer.target_names)'''

x=cancer.data
y=cancer.target

x_tr,x_ts,y_tr,y_ts=sklearn.model_selection.train_test_split(x,y,test_size=0.2)

classes=['malignant' 'benign']
clf=svm.SVC(kernel="linear",C=2)
clf.fit(x_tr,y_tr)
y_predict=clf.predict(x_ts)

ac=metrics.accuracy_score(y_ts,y_predict)
print(ac)
