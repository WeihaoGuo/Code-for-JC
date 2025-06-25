import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import os
import numpy as np
import pandas as pd
from scipy import io


# import data

file=io.loadmat('input_data.mat')
x=file['a'] #ws_icoads,lon,lat,year,month,ws_all

year=file['year'] #ws_icoads,lon,lat,year,month,ws_all
#year=year[0:1243430,:]
#year=year.reshape(1243430,)
year=year[8981280:,:]
year=year.reshape(2993760,)
del file
#标准化
x1=x[:,0]
x2=x[:,1]
x3=x[:,2]
x4=x[:,3]
x5=x[:,4]
x6=x[:,5]
x7=x[:,6]
x8=x[:,7]
x9=x[:,8]
x10=x[:,9]

x1=pd.Series(x1)
x2=pd.Series(x2)
x3=pd.Series(x3)
x4=pd.Series(x4)
x5=pd.Series(x5)
x6=pd.Series(x6)
x7=pd.Series(x7)
x8=pd.Series(x8)
x9=pd.Series(x9)
x10=pd.Series(x10)

x1 = x1.sub(x1.min()).div((x1.max() - x1.min()))
x2 = x2.sub(x2.min()).div((x2.max() - x2.min()))
x3 = x3.sub(x3.min()).div((x3.max() - x3.min()))
x4 = x4.sub(x4.min()).div((x4.max() - x4.min()))
x5 = x5.sub(x5.min()).div((x5.max() - x5.min()))
x6 = x6.sub(x6.min()).div((x6.max() - x6.min()))
x7 = x7.sub(x7.min()).div((x7.max() - x7.min()))
x8 = x8.sub(x8.min()).div((x8.max() - x8.min()))
x9 = x9.sub(x9.min()).div((x9.max() - x9.min()))
x10 = x10.sub(x10.min()).div((x10.max() - x10.min()))

x[:,0]=np.array(x1)
x[:,1]=np.array(x2)
x[:,2]=np.array(x3)
x[:,3]=np.array(x4)
x[:,4]=np.array(x5)
x[:,5]=np.array(x6)
x[:,6]=np.array(x7)
x[:,7]=np.array(x8)
x[:,8]=np.array(x9)
x[:,9]=np.array(x10)

del x1
del x2
del x3
del x4
del x5
del x6
del x7
del x8
del x9
del x10

x#=x[0:1243430,:]
x=x[8981280:,:]
y=x[:,10]

for i in range(0,len(y)):
    if np.isnan(x[i,0]):
        y[i]=np.nan
    if np.isnan(x[i,1]):
        y[i]=np.nan
    if np.isnan(x[i,2]):
        y[i]=np.nan
    if np.isnan(x[i,3]):
        y[i]=np.nan
    if np.isnan(x[i,4]):
        y[i]=np.nan
    if np.isnan(x[i,5]):
        y[i]=np.nan
    if np.isnan(x[i,6]):
        y[i]=np.nan
    if np.isnan(x[i,7]):
        y[i]=np.nan
    if np.isnan(x[i,8]):
        y[i]=np.nan
    if np.isnan(x[i,9]):
        y[i]=np.nan
    if np.isnan(y[i]):
        x[i,:]=np.nan

for count in range(1990,2020):
    X_train=x[np.where(year!=count)]
    y_train=y[np.where(year!=count)]

    #去除nan
    y_train=pd.DataFrame(y_train)
    y_train=y_train.dropna()
    y_train=y_train.values
    X_train=pd.DataFrame(X_train)
    X_train=X_train.dropna()   
    X_train=X_train.values
    #x = np.nan_to_num(x)
    #y = np.nan_to_num(y)
    io.savemat('train_'+str(count)+'.mat',{'X_train':X_train,'y_train':y_train})
    del X_train,y_train