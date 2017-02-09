import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from sklearn import gaussian_process as gp
from sklearn.metrics import mean_squared_error as mse
from sklearn.gaussian_process.kernels import *
from sklearn.gaussian_process import GaussianProcessRegressor

def prep_batches(TS,batch_size):
    batches = [TS[i*batch_size:(i+1)*batch_size,] for i in range(int((TS.shape[0] - TS.shape[0] % batch_size)/batch_size))]
    if TS.shape[0] % batch_size != 0:
        i = int((TS.shape[0] - TS.shape[0] % batch_size)/batch_size)
        batch_end = TS[i*batch_size:TS.shape[0],]
        return np.array(batches),np.array(batch_end),True
    return np.array(batches),None,False
    
def prep_data(window,window_size):
    X = np.array([window[i:window_size+i,] for i in range(window.shape[0] - window_size)])
    y = window[window_size:window.shape[0],].reshape([window.shape[0] - window_size,1])
    return X,y   
        
def prep_input(batch_1,batch_2,window_size):
    window = np.concatenate((batch_1,batch_2),axis = 0)
    return np.r_[np.array([batch_1[k:k+window_size] for k in range(batch_1.shape[0] - window_size)]),np.array([window[batch_1.shape[0] - window_size + k : batch_1.shape[0] + k] for k in range(batch_2.shape[0])])]
    
    
def prep_input_pred(batch_1,batch_2,window_size):
    window = np.concatenate((batch_1,batch_2),axis = 0)
    return np.array([window[batch_1.shape[0] - window_size + k : batch_1.shape[0] + k] for k in range(batch_2.shape[0])])
    
def GPR(TS,batch_size,window_size,kernel,standardization):
    if standardization:
        scaler = MinMaxScaler(feature_range=(0, 1))
        TS_standardized = scaler.fit_transform(TS)
    else:
        TS_standardized = TS
    batches, batch_end,bool_end = prep_batches(TS_standardized,batch_size)
    GPR = gp.GaussianProcessRegressor(kernel=kernel,alpha = 0)
    window = np.array([])
    preds = np.array([])
    params = [kernel,batch_size,window_size]
    for i in range(len(batches) - 1):
        batch_1 = batches[i]
        batch_2 = batches[i + 1]
        X, y = prep_data(batch_1,window_size)
        GPR.fit(X,y)
        
        if i != 0:
            X_to_predict = prep_input_pred(batch_1,batch_2,window_size)
        else:
            X_to_predict = prep_input(batch_1,batch_2,window_size)
            
        preds = np.concatenate((preds,GPR.predict(X_to_predict).reshape([X_to_predict.shape[0],])),axis=0)
    
    if bool_end:
        batch_1 = batches[-1]
        X, y = prep_data(batch_1,window_size)
        GPR.fit(X,y)
        X_to_predict = prep_input_pred(batch_1,batch_end,window_size)
        preds = np.concatenate((preds,GPR.predict(X_to_predict).reshape([X_to_predict.shape[0],])),axis=0)
    
    if standardization:
        preds = scaler.inverse_transform(preds)
        
    mse = np.sum((TS[window_size:].reshape([TS[window_size:].shape[0],1]) - np.array(preds).reshape([TS[window_size:].shape[0],1]))**2)/TS[window_size:].shape[0]
    
    return TS[window_size:],preds,mse,params

