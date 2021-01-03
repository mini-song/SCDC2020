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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import statsmodels.api as sm
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm

data1 = pd.read_csv('[Track1_데이터3] samp_cst_feat.csv',encoding = 'euc-kr')
data2 = pd.read_csv('[Track1_데이터2] samp_train.csv',encoding = 'euc-kr')
data1["MRC_ID_DI"] = data2["MRC_ID_DI"]

data1["MRC_ID_DI"] = data2["MRC_ID_DI"]

categories = ['VAR007','VAR015','VAR018','VAR026','VAR059',
              'VAR066','VAR067','VAR070','VAR077','VAR078',
              'VAR094','VAR096','VAR097','VAR098','VAR107',
              'VAR111','VAR124','VAR127','VAR143','VAR144',
              'VAR145','VAR148','VAR165','VAR177','VAR179',
              'VAR199','VAR208',"MRC_ID_DI"]

data1[categories] = data1[categories].astype("int64")

data1.groupby(["MRC_ID_DI"]).size()

# #### 온라인 마켓 사용, 미사용으로 분류

data1["MRC_ID_DI"] = data1["MRC_ID_DI"].replace(range(1,11),1)

data1 = data1.drop(['cst_id_di'],axis = 1)

samsung = sm.add_constant(data1, has_constant = 'add')
samsung.head()

feature_columns = list(samsung.columns.difference(["MRC_ID_DI"]))
X = samsung[feature_columns]
y = samsung["MRC_ID_DI"]
print(y)

x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                   train_size = 0.7, test_size = 0.3,
                                                   random_state = 100) #set_seed
print("x_train.shape = {}, x_test.shape = {}, y_train.shape = {}, y_test.shape = {}".format(x_train.shape, x_test.shape,
                                                                                            y_train.shape, y_test.shape))

model = sm.Logit(y_train, x_train)
results = model.fit(method = "newton")

results.summary()

results.params

np.exp(results.params)

results.aic

y_pred = results.predict(x_test)
y_pred


# +
def PRED(y, threshold):
    Y = y.copy()
    Y[Y > threshold] = 1
    Y[Y <= threshold] = 0
    return(Y.astype(int))

Y_pred = PRED(y_pred,0.5)
Y_pred
# -

# ### 오분류표

cfmat = confusion_matrix(y_test, Y_pred)
def acc(cfmat) :
    acc = round((cfmat[0,0]+cfmat[1,1])/np.sum(cfmat),3)
    return(acc)
acc(cfmat) # accuracy == 0.863

pca = PCA(n_components = 10)
pca.fit(X)

PCscore = pca.transform(X)
PCscore[:,0:5]

eigens_vector = pca.components_.transpose()
eigens_vector

# +
mX = np.matrix(X)

(mX * eigens_vector)[:, 0:5]
# -

print(PCscore)
plt.scatter(PCscore[:, 0], PCscore[:, 1], c = y)
print(PCscore[:,0])
plt.show()

# +
distortions = []

for i in range(1, 11) :
    km = KMeans(n_clusters = i, random_state = 102)
    km.fit(X)
    distortions.append(km.inertia_)

plt.plot(range(1, 11), distortions, marker = 'o')
plt.xlabel("# of clusters")
plt.ylabel("Distortion")
plt.show()

# +
lr_clf = LogisticRegression(max_iter = 10000)
lr_clf.fit(x_train, y_train)

pred_lr = lr_clf.predict(x_test)
print(accuracy_score(y_test, pred_lr))
print(mean_squared_error(y_test, pred_lr))
# -

bag_clf = BaggingClassifier(base_estimator = lr_clf,
                           n_estimators = 5,
                           verbose = 1)
lr_clf_bag = bag_clf.fit(x_train, y_train)
pred_lr_bag = lr_clf_bag.predict(x_test)
pred_lr_bag

print(accuracy_score(y_test, pred_lr_bag))
print(mean_squared_error(y_test, pred_lr_bag))

# +
from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier()
dt_clf.fit(x_train, y_train)
pred_dt = dt_clf.predict(x_test)

print(accuracy_score(y_test, pred_dt))
print(mean_squared_error(y_test, pred_dt))
# -

rf_clf = RandomForestClassifier(n_estimators = 5, 
                                max_depth = 3,
                               random_state = 103, 
                               verbose = 1)
rf_clf.fit(x_train, y_train)
pred = rf_clf.predict(x_test)
print(accuracy_score(y_test, pred))

rf_clf = RandomForestClassifier(n_estimators = 500, 
                                max_depth = 3,
                               random_state = 103, 
                               verbose = 1)
rf_clf.fit(x_train, y_train)
pred = rf_clf.predict(x_test)
print(accuracy_score(y_test, pred))

rf_clf = RandomForestClassifier(n_estimators = 500, 
                                max_depth = 10,
                               random_state = 103, 
                               verbose = 1)
rf_clf.fit(x_train, y_train)
pred = rf_clf.predict(x_test)
print(accuracy_score(y_test, pred))

rf_clf4 = RandomForestClassifier()

# +
params = { 'n_estimators' : [10, 100, 500, 1000],
           'max_depth' : [3, 5, 10, 15]}

rf_clf4 = RandomForestClassifier(random_state = 103, 
                                n_jobs = -1,
                                verbose = 1)
grid_cv = GridSearchCV(rf_clf4,
                      param_grid = params,
                      n_jobs = -1,
                      verbose = 1)
grid_cv.fit(x_train, y_train)

print('최적 하이퍼 파라미터: ', grid_cv.best_params_)
print('최고 예측 정확도: {:.4f}'.format(grid_cv.best_score_))

# +
test_acc = []

for n in range(1, 11):
    clf = KNeighborsClassifier(n_neighbors = n)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    test_acc.append(accuracy_score(y_test, y_pred))
    
    print("k : {}, 정확도 : {}".format(n, accuracy_score(y_test, y_pred)))
# -

test_acc

plt.figure()
plt.plot(range(1, 11), test_acc, label = 'test')
plt.xlabel("n_neighbors")
plt.ylabel("accuracy")
plt.xticks(np.arange(0, 11, step = 1))
plt.legend()
plt.show()

# +
clf_lin = svm.LinearSVC()
clf_lin.fit(x_train, y_train)

y_pred_lin = clf_lin.predict(x_test)

print(confusion_matrix(y_test, y_pred_lin))
print(accuracy_score(y_test, y_pred_lin))
# -

# #### 0(미사용), 1,6,8 Group shaping

group0, group1 = data1[data1["MRC_ID_DI"]==0], data1[data1["MRC_ID_DI"]==1]
group6, group8 = data1[data1["MRC_ID_DI"]==6], data1[data1["MRC_ID_DI"]==8]

print("group0.shape = {}, group1.shape = {}, group6.shape = {}, group8.shape = {}".format(group0.shape, group1.shape,
                                                                                          group6.shape, group8.shape))

group0, group1, group6, group8 = pd.get_dummies(group0), pd.get_dummies(group1), pd.get_dummies(group6), pd.get_dummies(group8)

# #### Dummy 변수 생성, group by shape

print("group0.shape = {}, group1.shape = {}, group6.shape = {}, group8.shape = {}".format(group0.shape, group1.shape,
                                                                                          group6.shape, group8.shape))

group0 = group0.T.drop(["MRC_ID_DI"]).T
group0.index = range(1,len(group0)+1)
group0

group0.corr(method = 'pearson')

for a in categories:
    data1[a].value_counts().plot(kind= 'bar')
    plt.title(a)
    plt.show()


