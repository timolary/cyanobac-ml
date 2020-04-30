import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os 
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn import svm
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import recall_score, make_scorer
from sklearn.model_selection import cross_val_score, cross_val_predict

ds1 = pd.read_csv('data/AllSites.csv')
threshold = np.where(ds1['NP_Cya_bio'] >= 4e8)
target = np.zeros(len(ds1['NP_Cya_bio']))
target[threshold] = 1
ds1['target'] = pd.Series(target)
ds1 = ds1.dropna(axis=0, how='any')

ds2 = ds1.drop(['Station', 'Stratum','Date','StationID','Time'], axis=1)
# ds2
ds2['Depth'] = ds1['Depth'].astype(str).str.extract('([-+]?\d*\.\d+|\d+)').astype(float)
ds2['TP'] = ds1['TP'].astype(str).str.extract('([-+]?\d*\.\d+|\d+)').astype(float)
ds2['Cl'] = ds1['Cl'].astype(str).str.extract('([-+]?\d*\.\d+|\d+)').astype(float)
ds2['DP'] = ds1['DP'].astype(str).str.extract('([-+]?\d*\.\d+|\d+)').astype(float)
ds2['TN'] = ds1['TN'].astype(str).str.extract('([-+]?\d*\.\d+|\d+)').astype(float)
ds2['TempC'] = ds1['TempC'].astype(str).str.extract('([-+]?\d*\.\d+|\d+)').astype(float)
ds2['Chla'] = ds1['Chla'].astype(str).str.extract('([-+]?\d*\.\d+|\d+)').astype(float)
ds2['Chla'] = ds1['Chla'].astype(str).str.extract('([-+]?\d*\.\d+|\d+)').astype(float)
ds2['Secchi'] = ds1['Secchi'].astype(str).str.extract('([-+]?\d*\.\d+|\d+)').astype(float)
ds2['date'] = ds1['Date'].astype(str).str.extract('(\d)').astype(int) # This is just the month number
ds2 = ds2.drop(['NP_Cya_bio'], axis=1)
y = np.array(ds2['target'])
X = np.array(ds2.drop(['target'], axis=1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True)
y_test_pos= np.where(y_test == 1)
assert len(y_test_pos[0]) >= 5, "Need at least 5 positive samples in training set"

loo = LeaveOneOut()
X_train_normalized = preprocessing.normalize(X_train) # normalize X for processing
svm_classifer = svm.SVC()
kernels = ['linear', 'rbf', 'sigmoid']
scorer = make_scorer(recall_score, zero_division=0)
for kernel in kernels:
    print(f'KERNEL: {kernel}')
    svm_classifer = svm.SVC(kernel=kernel)
    distros = dict(C=np.logspace(-2, 10, 10), gamma=np.logspace(-9, 3, 10))
    print('optimizing hyperparams...')
    search = GridSearchCV(svm_classifer, distros, scoring=scorer, verbose=50, cv=loo, n_jobs=4)
    search = search.fit(X_train_normalized, y_train)
    params = search.best_params_
    print(f'hyperparamters: {params}')
    print('testing optimized hyperparams...')
    X_test_normalized = preprocessing.normalize(X_test)
    svm_classifer = svm.SVC(**params)
    y_predicted = cross_val_predict(svm_classifer, X_test_normalized, y_test, cv=20, verbose=1, n_jobs=4, pre_dispatch='2*n_jobs')
    print(metrics.classification_report(y, y_predicted))
    print(metrics.confusion_matrix(y, y_predicted))
    print('\n')

