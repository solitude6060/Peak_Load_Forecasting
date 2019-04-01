#!/usr/bin/env python
# coding: utf-8

# # DSAI HW1 Peak Load Forecasting
# 請根據台電歷史資料，預測未來七天的"電力尖峰負載"(MW)

# In[434]:


import numpy as np
import pandas as pd
from keras import backend as K
from keras.models import Sequential 
from keras.layers import Dense


# ## Data Preprocessing

# Read data

# In[421]:


def readfile(path):
    file = open(path, "r+")
    f = file.readlines()
    return_tp_list = []
    test_list = []
    for i in f[1:366]:
        #print(i)
        day_list = i[:-1].split(",")
        temp_list = []
        for d in day_list[:2]:
            temp_list.append(d)
        for d in day_list[3:5]:
            temp_list.append(d)
        test_list.append(day_list[2])
        #print(temp_list)
        #print(len(temp_list))
        return_tp_list.append(temp_list)
    
    return return_tp_list, test_list


# In[422]:


#tp_2017  = pandas.read_csv('data/taipower_2017.csv')[:365]
#tp_2018  = pandas.read_csv('data/taipower_2018.csv')[:365]
tp_2017, target_2017_list = readfile('data/taipower_2017.csv')
tp_2018, target_2018_list = readfile('data/taipower_2018.csv')

#print(tp_2018)


# Parser data to get the same week in last year and last week 

# In[423]:


def sliceData(tp_data_2017, tp_data_2018):
    #index = 1
    con_list = []
    data_2017_list = []
    for t in range(len(tp_data_2017[:-1])):
        if (t+1) % 7 is not 0:
            con_list += tp_data_2017[t][1:]
        else:
            #if t is 0:
            con_list += tp_data_2017[t][1:]
            data_2017_list.append(con_list)
            con_list = []
    
    con_list = []
    data_2018_list = []
    for t in range(len(tp_data_2018[:-1])):
        if (t+1) % 7 is not 0:
            con_list += tp_data_2018[t][1:]
            #print(con_list)
        else:
            #if t is 0:
            con_list += tp_data_2018[t][1:]    
            data_2018_list.append(con_list)
            con_list = []
    
    #print("2017", data_2017_list[0])
    #print("2018", data_2018_list[0])
    train_list = []
    temp_list =[]
    for i in range(len(data_2017_list)):
        if i == 0:
            temp_list = data_2017_list[i] + data_2017_list[-1]
            #print("i = 0", temp_list)
        else:
            temp_list = data_2017_list[i] + data_2017_list[i]
            #print("i != 0", temp_list)
        train_list.append(temp_list)
    
    return train_list


# In[424]:


train_X = sliceData(tp_2017, tp_2018)
print(train_X[0])
print(len(train_X[0]))


# In[425]:


def getTrainY(target_list):
    train_y_list = []
    con_list = []
    for i in range(len(target_list[:-1])):
        if (i+1) % 7 is not 0:
            con_list.append(target_list[i])
        else:
            con_list.append(target_list[i])
            train_y_list.append(con_list)
            con_list = []
    return train_y_list


# In[426]:


train_Y = getTrainY(target_2018_list)
#print(train_Y)
np_train_X = np.array(train_X)
np_train_Y = np.array(train_Y)

# np_train_X = np_train_X[:,:,np.newaxis]
# np_train_Y = np_train_X[:,:,np.newaxis]

#np_train_X  = np.reshape(np_train_X , (1, np_train_X.shape[0], np_train_X.shape[1]))
#np_train_Y  = np.reshape(np_train_Y , (1, np_train_Y.shape[0], np_train_Y.shape[1]))
#n = np_train_X.reshape(52, 1, 42)
#print(n.shape)
#print(np_train_X.shape)


# In[427]:


def bulidModel(x_shape):
    model = Sequential()
    model.add(Dense(units=256, input_shape=(x_shape[1], ), activation='linear'))
    model.add(Dense(units=7,activation='linear'))
    model.summary()
    return model


# In[428]:


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


# In[429]:


LinearModel = bulidModel(np_train_X.shape)
LinearModel.compile(loss=root_mean_squared_error, optimizer='rmsprop')


# In[430]:


train_history = LinearModel.fit(np_train_X , np_train_Y , batch_size=64 , epochs=1000 , verbose=1, validation_split=0.2)


# In[431]:


def getTestX(tp_2018):
    findDate = 0
    index = 0
    test_list = []
    for d in tp_2018:
        if d[0] == '20180402':
            findDate = 1
            index += 1
            test_list += d[1:]
        if findDate is 1:
            index += 1
            test_list += d[1:]
            #test_list.append(d)
        if index is 7:
            break
    return test_list


# In[432]:


testList = getTestX(tp_2018)
test_2019_list = ['28756', '1887', '6.56', '29140', '1933', '6.63', '30093', '1892', '6.29', '29673', '2054', '6.92', '25810', '2155', '8.35', '24466', '2298', '9.39', '23895', '1655', '6.92']
test_X = testList + test_2019_list
np_test_X = np.array(test_X)
print(test_X)
print(len(test_X))


# In[433]:


#print(np_test_X.shape)
test = np_test_X.reshape(1, 42)
#print(test.shape)
test_Y = LinearModel.predict(test)
print(test_Y[0])


# In[436]:


def saveCSV(test_Y):
    date = 20180402
    d_list = []
    temp_list = []
    index = 0
    for t in test_Y:
        temp_list.append(str(date+index))
        temp_list.append(t)
        d_list.append(temp_list)
        temp_list = []
        index += 1
    df = pd.DataFrame(d_list, columns=['date', 'peak_load(MW)'])
    return df


# In[437]:


df = saveCSV(test_Y[0])
df.to_csv('submission.csv', index=0)