import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns  # visualization tool
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import RandomForestClassifier
# Load data
data_train = pd.read_csv('./train_dec08_task3.csv')
data_test = pd.read_csv('./test_dec08_task3_only_features.csv')
X_train = data_train[['feature0','feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9']].values
y_train = data_train['class'].values


X_test = data_test[['feature0','feature1','feature2','feature3','feature4','feature5','feature6','feature7','feature8','feature9']].values
## Plot
#sns.pairplot(data_train)
#plt.show()

# CSV out
RF_clf = RandomForestClassifier()
RF_clf.fit(X_train, y_train)
predict = RF_clf.predict(X_test)
pd.DataFrame(
        {
            "id": list(range(1,len(predict)+1)),
            "Category":predict
        }
        ).to_csv('submission.csv',index=False , header=True)
