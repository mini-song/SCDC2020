# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import sys   
!{sys.executable} -m pip install boruta
import csv
import pandas as pd
import numpy as np
import statsmodels.api as sm
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
from sklearn import tree
import csv
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
import platform
from matplotlib import font_manager, rc
from sklearn.ensemble import RandomForestClassifier
from boruta import boruta_py  #모듈을 _py 식으로 직접 호출하면 aaa.aa(dd,dd) 이런식으로 사용해야된다.!! 
import pandas as pd
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

ata =[]
df = pd.read_csv('./정제.csv')
df = df.sample(frac=1)
df0 = df['MRC_ID_DI'] != -1

df2=df[df0]
print(df2)
dfy = df2['MRC_ID_DI']
data_ratio=[]
for i in df['MRC_ID_DI']:
    if i != -1:
        data_ratio.append(i)

for i in range(0, 11):
    print("%-4d %-4d %.1f%%" % (i, data_ratio.count(i), data_ratio.count(i)/len(data_ratio) * 100))

dfy.to_csv('./임시.csv')
y = dfy.iloc[0:]
del df2['MRC_ID_DI']
del df2['cst_id_di']
df2.to_csv('dummy_.csv')
X=df2.iloc[0:]
featureColumns = df2.columns.tolist()
del featureColumns[0]

# +
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3)
y_new=y
X_new=X
y = pd.read_csv('./임시.csv', header=None, index_col=0).values
y = y.ravel()
X.to_csv('./임시2.csv')
X = pd.read_csv('./임시2.csv', index_col=0).values
y=np.delete(y,0)



# +
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

feat_selector = boruta_py.BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)

feat_selector.fit(X, y)

feat_selector.support_

feat_selector.ranking_

X_filtered = feat_selector.transform(X)



# +
from sklearn.metrics import accuracy_score
Feature=feat_selector.support_
Feature_list=Feature.tolist()
Var_Num=[]
Var_Data=[]
for i in range(0,len(Feature_list)):
    if Feature_list[i] == True:
        Var_Num.append(i)
for i in range(0,len(Var_Num)):
    print(df2.columns[Var_Num[i]])
    Var_Data.append(df2.columns[Var_Num[i]])
print(Var_Data)


X_new2 = X_new[Var_Data]
X_train, X_test, Y_train, Y_test = train_test_split(X_new2, y_new, test_size=0.3)
model =LogisticRegression()
model.fit(X_train,Y_train)
ax2 = plt.subplot(212)
predict_result=model.predict(X_test)
pd.DataFrame(predict_result, columns=["prediction"]).plot(marker='o', ls="", ax=ax2,label=[0,1,2,3,4,5,6,7,8,9,10])
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11])
print(predict_result)
plt.title("Prediction Result")
plt.tight_layout()
plt.show()

x=[0,1,2,3,4,5,6,7,8,9,10]
# -




