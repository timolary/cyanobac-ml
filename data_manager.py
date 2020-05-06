import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_data():
    
    ds1 = pd.read_csv('data/AllSites.csv')
    threshold = np.where(ds1['NP_Cya_bio'] >= 4e8)
    target = np.zeros(len(ds1['NP_Cya_bio']))
    target[threshold] = 1
    ds1['target'] = pd.Series(target)
    # ds1 = ds1.dropna(axis=0, how='any')
    ds1 = ds1.dropna(axis=0, subset = ['NP_Cya_bio'])
    
    ds2 = ds1.drop(['Station', 'Stratum','Date','StationID','Time', 'Depth'], axis=1)

    # ds2
    #ds2['Depth'] = ds1['Depth'].astype(str).str.extract('([-+]?\d*\.\d+|\d+)').astype(float)
    ds2['TP'] = ds1['TP'].astype(str).str.extract('([-+]?\d*\.\d+|\d+)').astype(float)
    ds2['Cl'] = ds1['Cl'].astype(str).str.extract('([-+]?\d*\.\d+|\d+)').astype(float)
    ds2['DP'] = ds1['DP'].astype(str).str.extract('([-+]?\d*\.\d+|\d+)').astype(float)
    ds2['TN'] = ds1['TN'].astype(str).str.extract('([-+]?\d*\.\d+|\d+)').astype(float)
    ds2['TempC'] = ds1['TempC'].astype(str).str.extract('([-+]?\d*\.\d+|\d+)').astype(float)
    ds2['Chla'] = ds1['Chla'].astype(str).str.extract('([-+]?\d*\.\d+|\d+)').astype(float)
    ds2['Secchi'] = ds1['Secchi'].astype(str).str.extract('([-+]?\d*\.\d+|\d+)').astype(float)
    ds2['N:P'] = ((ds2['TN']*1e-3)/14.007)/((ds2['TP']*1e-6)/30.974) #for TN in ds2['TN'] for TP in ds2['TP']]
    ds2['Month'] = ds1['Date'].astype(str).str.extract('(\d+)').astype(int) # This is just the month number
    
#    ds2 = ds2.drop(['NP_Cya_bio'], axis=1)
#    ds2 = ds2.fillna(value=0)

#    y = np.array(ds2['target'])
#    X = np.array(ds2.drop(['target'], axis=1))
#    for i, x in enumerate(X.T):
#        if i == 8:
#            X[:,i]=np.nan_to_num(x,nan=0,posinf=0,neginf=0)
#        else:
#            X[:,i]=np.nan_to_num(x,nan=np.nanmean(x),posinf=np.nanmean(x),neginf=np.nanmean(x))
#    # X=np.nan_to_num(X,nan=0,posinf=0,neginf=0)
#    ds2.drop(['target'], axis=1)

    #New ds3 with missing values filled in. 
    ds3 = ds2.copy()
    for colname in ds3:
        ds3[colname] = ds3[colname].fillna(ds3.groupby('Month')[colname].transform('mean'))

    #Delete rows with missing values from ds2
    ds2 = ds2.dropna(axis=0, how='any')

    return ds2, ds3
    
def split_data(X,y, num_pos):
    '''
   Ensure there are at least num_pos samples in the testing set 
    '''
    ytp=0
    while ytp < num_pos:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True)
        y_train_pos = np.where(y_train == 1)
        y_test_pos= np.where(y_test == 1)
        ytp = len(y_test_pos[0])
    assert len(y_test_pos[0]) >= num_pos, "Need at least 5 positive samples in training set"
    return X_train, X_test, y_train, y_test