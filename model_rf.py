from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.utils import column_or_1d
import numpy as np
import pandas as pd
import os
import joblib
from scipy import io


n_inputs=10
for count in range(1990,2020):
    # import data
    file=io.loadmat('train_'+str(count)+'.mat')
    x_train=file['X_train']
    y_train=file['y_train']
    del file
    x_train=x_train[0:len(y_train),[0,1,2,3,4,5,6,7,8,9]]
    x_train=x_train.reshape(len(x_train),n_inputs)
    
    X_train=x_train[:,:]
    y_train=y_train[:,:]
    
    # model
    rfr = RandomForestRegressor(n_estimators=200,bootstrap=True,oob_score=True,
                                min_weight_fraction_leaf=0,min_samples_split=3,min_samples_leaf=3
                                )
    
    y_train = column_or_1d(y_train, warn=True)
    
    
    # train
    rfr.fit(x_train[:,:], y_train)
    random_forest_importance=rfr.feature_importances_
    # save
    joblib.dump(rfr,'rf_ws_'+str(count)+'.pkl')
    io.savemat('importance_'+str(count)+'.mat',{'importance':random_forest_importance})
    del random_forest_importance,rfr,X_train,y_train,x_train
    
# import testing set
file=io.loadmat('data_2020_2022.mat')

x=file['testing_var'] #ws,qa,sst,ta,lon,lat,year,month
del file

for i in range(0,len(x[:,1])):
    if np.isnan(x[i,0]):
        x[i,0]=999
    if np.isnan(x[i,1]):
        x[i,1]=999
    if np.isnan(x[i,2]):
        x[i,2]=999
    if np.isnan(x[i,3]):
        x[i,3]=999
    if np.isnan(x[i,4]):
        x[i,4]=999
    if np.isnan(x[i,5]):
        x[i,5]=999
    if np.isnan(x[i,6]):
        x[i,6]=999
    if np.isnan(x[i,7]):
        x[i,7]=999
    if np.isnan(x[i,8]):
        x[i,8]=999
    if np.isnan(x[i,9]):
        x[i,9]=999

# import model
for count in range(1990,2020):
    #rfr = joblib.load('G:\\CMIP6\\LHF_new\\model_rf_access\\rfr_lhf_'+str(count)+'.pkl')
    rfr = joblib.load('rfr_ws_'+str(count)+'.pkl')
    pred1 = rfr.predict(x[:,[0,1,2,3,4,5,6,7,8,9]])
    io.savemat('ws_test_'+str(count)+'.mat',{'pred1':pred1}) 
    del rfr,pred1