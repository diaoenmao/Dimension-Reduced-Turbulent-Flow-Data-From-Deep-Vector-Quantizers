#!/usr/bin/env python
# coding: utf-8

# In[1]:


# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# We need to import key libraries that we're going to use.  
# For now this is just numpy, which is our linear algebra library
import numpy as np
import pandas as pd
# to make this notebook's output stable across runs, we are going to see the random seed
np.random.seed(42)

# To plot pretty figures
import matplotlib.pyplot as plt
# %matplotlib inline 
# This command figures show up in the notebook.  It's a "magic" command...
# Typically, this now happens by default so it is often an unnecessary command, but is good for standardization.
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15

# Read HDF5
import h5py

from decimal import Decimal

from numpy import zeros

from sklearn import preprocessing
from sklearn import pipeline # 


from scipy import stats


from mpl_toolkits.mplot3d import Axes3D # this part works well in py3
from matplotlib import cm

# from mpl_toolkits import mplot3d
# import pylab as p

from IPython.display import Image

from sklearn.metrics import mean_squared_error,r2_score

from sklearn import model_selection

import time

from sklearn import linear_model

import pickle

std_scaler = preprocessing.StandardScaler()


# In[2]:


PROJECT_ROOT_DIR = "."
PROJECT_SAVE_DIR = "Reza_ML_Saved_PHProject"
# makes the directory if it doesn't exist.
import os
if not (os.path.isdir(PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR)):
    print('Figure directory didn\'t exist, creating now.')
    os.mkdir(PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR)
else:
    print('Figure directory exists.') 


# In[3]:


#a simple defined helper function.
def savepdf(fig,name):
    fig.savefig(PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR+'/'+name+'.pdf',transparent=True)


# In[4]:


def savemodel_pkl(model,pkl_filename):
    # Save to file in the current working directory
    # pkl_filename = "pickle_model.pkl"  
    with open(PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR+'/'+pkl_filename+'.pkl', 'wb') as file:  
        pickle.dump(model, file)


# In[5]:


def loadmodel_pkl(pkl_filename):
    # Save to file in the current working directory
    # pkl_filename = "pickle_model.pkl"  
    with open(PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR+'/'+pkl_filename+'.pkl', 'rb') as file:  
        b=pickle.load(file)
        return b


# # User Input 

# In[6]:


Features_list=['R_Q_A_A2'] ## Features_list_possible=['R_Q_A','R_Q_A2','R_Q_A_A2']

### change accordingly ########################################
model_no='10'
MLP_Model_param={'hidden_layer_sizes':[300,300,300,300,300],'max_iter':1000, 'activation':'tanh','tol':1e-6,'learning_rate_init':2e-4,'random_state':42,'early_stopping':True}

N=128; # number of grid points in the DNS setup
DNSdata_time_Step=8000
TestSet_Percentage=10/100

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ done code input $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


# import the data

# In[7]:



f = h5py.File(PROJECT_ROOT_DIR+'/'+'DNS_Data'+'/'+'Q_R_A_A2_H_data_Ts_'+str(DNSdata_time_Step)+'.h5','r')
# $$$ note how the X data is constructed (Q,R,A,A2)
X_1D=np.concatenate([ f['dataset_Q_R_data'][:] , f['dataset_Atot'][:] , f['dataset_A2_tot'][:]  ],axis=1)
Y_1D=f['dataset_Htot'][:]
f.close
print(type(X_1D),X_1D.shape,Y_1D.shape)


# In[8]:


# This is fixed by the way we constructed X_TrainValid_1D
RQ_Index=np.array(range(0,2))
A_Index=np.array(range(2,11))
A2_Index=np.array(range(11,20))

# all the possible combinations of input features
Features_list_possible=['R_Q_A','R_Q_A2','R_Q_A_A2']

R_Q_A_Index=np.concatenate([RQ_Index,A_Index],axis=0)
R_Q_A2_Index=np.concatenate([RQ_Index,A2_Index],axis=0)
R_Q_A_A2_Index=np.concatenate([RQ_Index,A_Index,A2_Index],axis=0)

print(R_Q_A_Index, R_Q_A2_Index, R_Q_A_A2_Index)


# # Reserve a Test set and Split the Training and Validation sets

# In[ ]:





# In[9]:



TrainValidSets_Percentage=1-TestSet_Percentage
TestSet_NDataPoints=np.int_(TestSet_Percentage*(N*N*N))
TrainValidSets_NDataPoints=np.int_(TrainValidSets_Percentage*(N*N*N))
# print(TestSet_NDataPoints,TrainValidSets_NDataPoints)
np.random.seed(42)
NDataPoints_random_numbers=np.random.randint(low=0,high=N*N*N-1,size=[N*N*N])
# print(NDataPoints_random_numbers.shape,NDataPoints_random_numbers[:5])
TestSet_IDs=NDataPoints_random_numbers[0:TestSet_NDataPoints]
print(TestSet_IDs.shape,TestSet_IDs[:5])
TrainValidSets_IDs=NDataPoints_random_numbers[TestSet_NDataPoints:TestSet_NDataPoints+TrainValidSets_NDataPoints]
# print(TrainValidSets_IDs.shape,TrainValidSets_IDs[:5])
# print(TestSet_IDs[-1],TrainValidSets_IDs[0])


# In[10]:


X_test=X_1D[TestSet_IDs,:] 
Y_test=Y_1D[TestSet_IDs,:] 
print("X_test.shape:{} and Y_test.shape:{}".format(X_test.shape,Y_test.shape))

X_train,X_valid,Y_train,Y_valid=    model_selection.train_test_split(X_1D[TrainValidSets_IDs,:],Y_1D[TrainValidSets_IDs,:],random_state=42)
print("X_train.shape:{} , X_valid.shape :{}, Y_train.shape :{}, Y_valid.shape :{}".      format(X_train.shape, X_valid.shape, Y_train.shape, Y_valid.shape))

# just to check 
# print(X_train[2,RQ_Index])
# print(X_train[2,A_Index])
# print(X_train[2,A2_Index])

# print(X_train[2,RQ_A2_Index])

# print(X_train[2,:])


# In[11]:


# print(type(Features_list),type(Features_list_possible))
if Features_list[0]==Features_list_possible[0]:    
    Feature_Index=R_Q_A_Index
    print(Features_list[0],Feature_Index)
elif Features_list[0]==Features_list_possible[1]:
    Feature_Index=R_Q_A2_Index
    print(Features_list[0],Feature_Index)
else:
    Feature_Index=R_Q_A_A2_Index
    print(Features_list[0],Feature_Index)


# In[12]:


# print(('MLP_SkL_XF_'+str(Features_list[0])+'_M_'+model_no))
# print(str(Features_list[0]),Features_list[0])


# # MLP with SkitLearn

# In[13]:


df_MLP_SkL = pd.DataFrame([])


# In[14]:


# build and fit the model
from sklearn import neural_network
our_regressor=neural_network.MLPRegressor(**MLP_Model_param)
Model_WithScaling=pipeline.Pipeline([('scaler',std_scaler),('regr',our_regressor)])
# with scaling
start_fit= time.time()
Model_WithScaling.fit(X_train[:,Feature_Index], Y_train)
end_fit= time.time()
duration_fit=end_fit - start_fit
################################################
# pickle the model 
# save the model to disk
pkl_filename= ('MLP_SkL_XF_'+Features_list[0]+'_M_'+model_no)
savemodel_pkl(Model_WithScaling,pkl_filename)
################################################
# compute the score/MSE
start_score= time.time()
Score_scaled_training=Model_WithScaling.score(X_train[:,Feature_Index], Y_train)
Score_scaled_validation=Model_WithScaling.score(X_valid[:,Feature_Index], Y_valid)

MSE_scaled_training_IndividualOutputs=mean_squared_error(y_true=Y_train,                                                         y_pred=Model_WithScaling.predict(X_train[:,Feature_Index]),multioutput='raw_values')
MSE_scaled_validation_IndividualOutputs=mean_squared_error(y_true=Y_valid,                                                           y_pred=Model_WithScaling.predict(X_valid[:,Feature_Index]),multioutput='raw_values')

AvgMSE_scaled_training=mean_squared_error(y_true=Y_train,y_pred=Model_WithScaling.predict(X_train[:,Feature_Index]),multioutput='uniform_average')
AvgMSE_scaled_validation=mean_squared_error(y_true=Y_valid,y_pred=Model_WithScaling.predict(X_valid[:,Feature_Index]),multioutput='uniform_average')

loss=Model_WithScaling.named_steps['regr'].loss_
n_iter=Model_WithScaling.named_steps['regr'].n_iter_
end_score= time.time()
duration_score=end_score - start_score
################################################
# store the score/MSE in dataframe
df_MLP_SkL = df_MLP_SkL.append    (pd.DataFrame({'Number_of_features': Features_list[0],                   'Model_No': model_no,                   'Model_loss': loss ,                    'Model_n_iter': n_iter ,                    'Model_duration_fit': duration_fit ,                    'Model_duration_score': duration_score ,                    'Score_scaled_training':                    Score_scaled_training,                  'Score_scaled_validation':                    Score_scaled_validation,                  'Average_MSE_scaled_training':                   AvgMSE_scaled_training,                  'Average_MSE_scaled_validation':                   AvgMSE_scaled_validation,                  'MSE_scaled_training':                   [MSE_scaled_training_IndividualOutputs],                  'MSE_scaled_validation':                   [MSE_scaled_validation_IndividualOutputs]                 }                , index=[model_no]))

################################################
# pickle and save the dataframe
df_MLP_SkL.to_csv(PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR+'/'+'df_'+'MLP_SkL_XF_'+Features_list[0]+'_M_'+model_no+'.csv')
df_MLP_SkL.to_pickle(PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR+'/'+'df_'+'MLP_SkL_XF_'+Features_list[0]+'_M_'+model_no+'.pkl')


# In[ ]:





# In[ ]:




