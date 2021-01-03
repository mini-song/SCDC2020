# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
from sklearn import tree
import matplotlib.pyplot as plt
import platform
from matplotlib import font_manager, rc
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
df=pd.read_csv('./data.csv',encoding='cp949')
df=df.replace('low',0)
df=df.replace('mid',1)
df=df.replace('high',2)
df.to_csv('./정제된데이터.csv', encoding='cp949',index=False)
df=df.replace('A',0) #범주형 자료 해석 x ->  sklearn.preprocessing.OneHotEncoder 쓰거나 dummy encoding을 활용
df=df.replace('B',1)
df=df.replace('C',2)
df=df.replace('D',3)
df=df.replace('E',4)
df=df.replace('F',5)
df=df.replace('G',6)
df=df.replace('H',7)
df=df.replace('디저트',0)
df=df.replace('호텔/숙박',1)
df=df.replace('항공/여행사',2)
df=df.replace('취미',3)
df=df.replace('할인점',4)
df=df.replace('뷰티',5)
df=df.replace('면세점',6)
df=df.replace('종합몰',7)
df=df.replace('오픈마켓/소셜',8)
df=df.replace('전문몰',9)
print(df)

# +


Y=df['MRC_ID_DI']
del df['MRC_ID_DI']
X =df

clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=5)
clf = clf.fit(X, Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
clf.fit(X_train,Y_train)
print(round(clf.score(X_test,Y_test),2)*100,"%")
print("특성 중요도 : \n{}".format(clf.feature_importances_))
print(clf.feature_importances_)
list(df.columns.array)


imp = clf.feature_importances_
if platform.system() == 'Windows':
    font_name= font_manager.FontProperties(fname="C:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)
else:
    rc('font', family='AppleGothic')

# -
plt.barh(range(len(imp)), imp) 
#plt.yticks(range(len(imp)), list(df.columns.array)) 
plt.show()

a=imp.tolist()

for i in range(0,len(a)):
    if a[i]>0.05:
        print('VAR' + str(i+1) , a[i])


