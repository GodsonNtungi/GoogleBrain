# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 20:08:47 2022

@author: ASUS
"""

import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import GridSearchCV
from keras.constraints import maxnorm
from keras.wrappers.scikit_learn import KerasRegressor
import matplotlib.pyplot as plt
#%%


data=pd.read_csv('train.csv')

test=pd.read_csv('test.csv')

#%%


print(data.head())


#%%

X= data.copy()
y=X.pop('pressure')
print(X.shape)
x=X.drop(columns=['id', 'breath_id','time_step'])

feature_col=['id', 'breath_id', 'R', 'C', 'time_step', 'u_in', 'u_out']
feature_col2=['breath_id', 'R', 'C', 'time_step', 'u_in', 'u_out']




#%%

x.u_in=x.u_in/100
x.R=x.R/50
x.C=x.C/50


print(x.head())
X_train,X_valid,y_train,y_valid=train_test_split(x,y,train_size=0.8)

#%%

print(x.shape)
#%%

X_grid,x_not_use,y_grid,y_not_use =train_test_split(X_train,y_train,train_size=0.0004)


def defn_model(neurons=10,weight_constraint=0,dropout_rate=0):
    model=keras.Sequential([
        layers.Dense(units=neurons,activation='relu',input_shape=[7],
                     kernel_constraint=maxnorm(weight_constraint)),
        layers.Dropout(dropout_rate),
        layers.BatchNormalization(),
        layers.Dense(neurons,'relu',kernel_constraint=maxnorm(weight_constraint)),
        layers.Dropout(dropout_rate),
        layers.BatchNormalization(),
        layers.Dense(neurons,'relu',kernel_constraint=maxnorm(weight_constraint)),
        layers.Dropout(dropout_rate),
        layers.BatchNormalization(),
         layers.Dense(neurons,'relu',kernel_constraint=maxnorm(weight_constraint)),
        layers.Dropout(dropout_rate),
        layers.BatchNormalization(),
        
        
        
        layers.Dense(1)])
    
    
    .
    model.compile(loss='mae',optimizer='adam')
    return model



model1=KerasRegressor(build_fn=defn_model)

dropout_rate=[0.0]
weight_constraint=[1]
neurons=[80]
batch_size=[20,50,60,70,100]
epochs=[35]

param_grid=dict(dropout_rate=dropout_rate,weight_constraint=weight_constraint,neurons=neurons,
                batch_size=batch_size,epochs=epochs)

#%%

grid=GridSearchCV(estimator=model1,param_grid=param_grid,n_jobs=3,cv=3)

grid_result=grid.fit(X_grid,y_grid)

print('Best %f of %s'%(grid_result.best_score_,grid_result.best_params_))



#%%

X_train,X_valid,y_train,y_valid=train_test_split(x,y,train_size=0.01,test_size=0.01)

#%%

test_model=keras.Sequential([
    layers.Dense(16,activation='relu',input_shape=[4]),
    layers.Dense(1)])

test_model.compile(optimizer='adam',loss='mae',metrics=['mse'])
test_model.summary()
#%%


with tf.device('/CPU:0'):
    history_test=test_model.fit(X_train,y_train,validation_data=(X_valid,y_valid),epochs=600,batch_size=60,callbacks=earlystopping)
#%%
with tf.device('/CPU:0'):
    y_pred=test_model.predict(X_train)
#%%
plt.Figure(figsize=(100,10))
value=int(len(data.index)*0.01)
e=range(value)

lt.plot(e[60300:],y_train[60300:],'.r')

plt.plot(e[60300:],y_pred[60300:],'.y')

#%%
model=keras.Sequential([layers.Dense(80,activation='relu',input_shape=[4],kernel_constraint=maxnorm(1)),
                        layers.BatchNormalization(),
                        layers.Dense(80,'relu',kernel_constraint=maxnorm(1)),
                        layers.BatchNormalization(),
                        layers.Dense(80,'relu',kernel_constraint=maxnorm(1)),
                        layers.BatchNormalization(),
                        layers.Dense(80,'relu',kernel_constraint=maxnorm(1)),
                        layers.BatchNormalization(),
                     
                        
                        layers.Dense(1)])

model.compile(loss='mae',optimizer='adam',metrics=['mae'])#tf.keras.metrics.MeanAbsolutePercentageError()])
earlystopping=EarlyStopping(min_delta=0.001,patience=20,restore_best_weights=True)
#%%
  
with tf.device('/CPU:0'):
    history=model.fit(X_train,y_train,validation_data=(X_valid,y_valid),epochs=100,batch_size=60,callbacks=earlystopping)


#%%
prediction=model.predict(Xtest)



#%%
output=pd.DataFrame({'id':test.id,'pressure':prediction1})

output.to_csv('submission.csv',index=False)

#%%











