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


# In[6]:


def batch_generator(X, y, batch_size):
    total_batches = len(X) // batch_size
    current_batch = 0
    while True:
        start = batch_size * current_batch
        end = start + batch_size
        yield (X[start:end], y[start:end])
        current_batch = (current_batch + 1) % total_batches


# # User Input 1

# In[7]:


Features_list=['R_Q_A_A2'] ## Features_list_possible=['R_Q_A','R_Q_A2','R_Q_A_A2']

### change accordingly ########################################
model_no='10'


batch_size_t=100

# Parameters
learning_rate = 2e-4
display_step = 1

N_epochs=1000
max_diff_R2score=0.001
max_diff_loss=0.01

# Network Parameters
n_fixed_neurons=300
# MLP_Model_param={'hidden_layer_sizes':[1000,2500],'max_iter':1000, 'activation':'tanh',\
#                 'tol':1e-6,'learning_rate_init':2e-4,'random_state':42,'early_stopping':True}

N=128; # number of grid points in the DNS setup
DNSdata_time_Step=8000
TestSet_Percentage=10/100

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ done code input $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


# import the data

# In[8]:



f = h5py.File(PROJECT_ROOT_DIR+'/'+'DNS_Data'+'/'+'Q_R_A_A2_H_data_Ts_'+str(DNSdata_time_Step)+'.h5','r')
# $$$ note how the X data is constructed (Q,R,A,A2)
X_1D=np.concatenate([ f['dataset_Q_R_data'][:] , f['dataset_Atot'][:] , f['dataset_A2_tot'][:]  ],axis=1)
Y_1D=f['dataset_Htot'][:]
f.close
print(type(X_1D),X_1D.shape,Y_1D.shape)


# In[9]:


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





# In[10]:



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


# In[11]:


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


# In[12]:


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


# In[13]:


# print(('MLP_SkL_XF_'+str(Features_list[0])+'_M_'+model_no))
# print(str(Features_list[0]),Features_list[0])


# # MLP with TensorFlow

# In[14]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


# Let's first get a new, empty graph to work with.
# 
# 

# In[15]:


tf.reset_default_graph()
g = tf.get_default_graph()
g.get_operations()


# In[16]


Number_of_features=len(Feature_Index)
Number_of_outputs=len(Y_1D[0])
X=tf.placeholder(tf.float32)
X.set_shape((None,Number_of_features))
Y=tf.placeholder(tf.float32)
Y.set_shape((None,Number_of_outputs))


# ### prepare data

# In[17]:


std_scaler.fit(X_train[:,Feature_Index])
train_X=std_scaler.transform(X_train[:,Feature_Index])
train_y=Y_train
valid_X=std_scaler.transform(X_valid[:,Feature_Index])
valid_y=Y_valid
total_batches_t = len(train_X) // batch_size_t
training_generator = batch_generator(train_X, train_y, batch_size=batch_size_t)


# In[31]:


model_Inputs_PH=Sequential() # initialize the model
model_Inputs_PH.add(layers.Dense(n_fixed_neurons,activation='tanh',input_shape=(Number_of_features,),kernel_initializer='glorot_uniform' ))
model_Inputs_PH.add(layers.Dense(n_fixed_neurons,activation='tanh',kernel_initializer='glorot_uniform' ))
model_Inputs_PH.add(layers.Dense(n_fixed_neurons,activation='tanh',kernel_initializer='glorot_uniform' ))
model_Inputs_PH.add(layers.Dense(n_fixed_neurons,activation='tanh',kernel_initializer='glorot_uniform' ))
model_Inputs_PH.add(layers.Dense(n_fixed_neurons,activation='tanh',kernel_initializer='glorot_uniform' ))
model_Inputs_PH.add(layers.Dense(Number_of_outputs))
model_Inputs_PH.summary()


# ### Computing the loss and creating the optimizer

# In[32]:


predicted_y=model_Inputs_PH(X)
avg_loss = tf.reduce_mean(tf.squared_difference(predicted_y, Y))

global_step=tf.train.get_or_create_global_step()
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(avg_loss,global_step=global_step)


# ### Initializing variables and training
# Finally, it's time for training. Let's add an op to the graph that initializes all variables, then start a session and run the training code.

# In[33]:


# This piece is only necessary so as not to use up an ungodly amount of GPU memory:
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


# In[34]:


init_all_vars = tf.global_variables_initializer()
saver = tf.train.Saver()


# In[38]:


validation_feed_dict = {X: valid_X, Y: valid_y}


# In[39]:


best_loss = np.infty
best_R2score =-np.infty
df_TF_ANN = pd.DataFrame([])
FName=('MLP_TF_XF_'+Features_list[0]+'_M_'+model_no)


# In[40]:


with tf.Session(config=config) as sess:
    start_fit= time.time()
    sess.run(init_all_vars)

        
    for i in range(N_epochs):
        for v in range(total_batches_t):                       
            X_batch, y_batch = next(training_generator)
            feed_dict = {X: X_batch, Y: y_batch}

            _, loss= sess.run([optimizer, avg_loss], feed_dict=feed_dict)
        if i % 100 == 0:
            valid_loss, predicted_y_valid = sess.run([avg_loss,predicted_y], feed_dict=validation_feed_dict)
            r2_score_valid=r2_score(valid_y,predicted_y_valid)
            train_loss, predicted_y_train = sess.run([avg_loss,predicted_y], feed_dict={X: train_X, Y: train_y})
            r2_score_train=r2_score(train_y,predicted_y_train)
            
            MSE_scaled_training_IndividualOutputs=mean_squared_error(y_true=train_y,y_pred=predicted_y_train,multioutput='raw_values')
            
            MSE_scaled_validation_IndividualOutputs=mean_squared_error(y_true=valid_y,y_pred=predicted_y_valid,multioutput='raw_values')
            print("Iter {}: training loss = {}, validation loss = {}, r2_score_valid = {}, train_loss = {}, r2_score_train = {}".format                  (i, loss, valid_loss,r2_score_valid,train_loss,r2_score_train))
            
            
            if best_loss-valid_loss> max_diff_loss and r2_score_valid-best_R2score>max_diff_R2score:
                save_path = saver.save(sess, PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR+'/'+FName+'.ckpt')
                best_loss = valid_loss
                best_R2score=r2_score_valid
                checks_without_progress = 0
            else:
                save_path = saver.save(sess, PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR+'/'+FName+'.ckpt')
                print("Early stopping! where")
                print("Iter {}: training loss = {}, validation loss = {},(best_loss-valid_loss) {}, (r2_score_valid-best_R2score) {} = "                      .format(i,loss, valid_loss,best_loss-valid_loss,r2_score_valid-best_R2score))
                end_fit= time.time()
                duration_fit=end_fit - start_fit
                # ################# save dataframe ##################
                df_TF_ANN = df_TF_ANN.append                (pd.DataFrame({'Number_of_features': Number_of_features,                   'Model_No': model_no,                   'Model_n_iter': i ,                    'Model_training_duration': duration_fit ,                    'Score_scaled_training':                    r2_score_train,                  'Score_scaled_validation':                    r2_score_valid,                  'Average_MSE_scaled_training':                   train_loss,                  'Average_MSE_scaled_validation':                   valid_loss,                  'MSE_scaled_training':                   [MSE_scaled_training_IndividualOutputs],                  'MSE_scaled_validation':                   [MSE_scaled_validation_IndividualOutputs]                 }                , index=[model_no]))
                
                df_TF_ANN.to_csv(PROJECT_ROOT_DIR+'/'+PROJECT_SAVE_DIR+'/'+'df_'+FName+'.csv')
                # ################# done with saving ###########################
                break


# In[ ]:





# In[ ]:





# In[ ]:




