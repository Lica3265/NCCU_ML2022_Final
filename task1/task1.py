import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns  # visualization tool
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
# Load data
data_train = pd.read_csv('./train_nov28_task1.csv')
data_test = pd.read_csv('./test_nov28_task1_only_features.csv')
#data_train.info()
#data_train.shape

#y_train = data_train['label'].values
#X_train = data_train.drop("label", axis=1)
#X_train = X_train.values

#print(data_test[['feature0','feature1','feature2','feature3','feature4']].values[:3])
#print(data_train[['class']].values[:3])
X_train = data_train[['feature0','feature1','feature2','feature3','feature4']].values
y_train = data_train['class'].values

#X_train = X._train.values

#y_test = data_test['label'].values
#X_test = data_test.drop("label", axis=1)
#X_test = X_test.values

X_test = data_test[['feature0','feature1','feature2','feature3','feature4']].values
#y_test = data_test['class'].values

#knn_clf = KNeighborsClassifier(n_neighbors=5)
#knn_clf.fit(X_train, y_train)

RF_clf = RandomForestClassifier(n_estimators=100,random_state= 42)
RF_clf.fit(X_train, y_train)
#knn_clf.predict(X_test).to_csv(index_label="id")
#predict = knn_clf.predict(X_test)
predict = RF_clf.predict(X_test)
pd.DataFrame(
        {
            "id": list(range(1,len(predict)+1)),
            "Category":predict
        }
        ).to_csv('submission.csv',index=False , header=True)
#baseline_accuracy = knn_clf.score(X_test, y_test)
#print(baseline_accuracy)
