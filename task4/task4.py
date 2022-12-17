import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns  # visualization tool
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.impute import SimpleImputer

# Load data
data_train = pd.read_csv('./archive/train_dec08_task4_missing.csv')
data_test = pd.read_csv('./archive/test_dec08_task4_missing_only_features.csv')
# Deal with missing value
#data_train = data_train.dropna()


X_train = data_train[['feature0','feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9']].values
y_train = data_train['class'].values
#X_train = data_train[['feature0','feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature11','feature12','feature13']].values


X_test = data_test[['feature0','feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9']].values
## Plot
#sns.pairplot(data_train)
#plt.show()


# Create our imputer to replace missing values with the mean e.g.
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
#imp = SimpleImputer(missing_values=np.nan, strategy='median')
#imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

imp = imp.fit(X_train)
X_train_imp = imp.transform(X_train)
X_test_imp = imp.transform(X_test) 
# Train
clf = RandomForestClassifier()
#clf = svm.SVC()
clf.fit(X_train_imp, y_train)
#clf.fit(X_train, y_train)
predict = clf.predict(X_test_imp)
# CSV out
pd.DataFrame(
        {
            "id": list(range(1,len(predict)+1)),
            "Category":predict
        }
        ).to_csv('submission.csv',index=False , header=True)
