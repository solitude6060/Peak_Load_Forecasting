#!/usr/bin/env python
# coding: utf-8

# # DSAI HW1 Peak Load Forecasting
# 請根據台電歷史資料，預測未來七天的"電力尖峰負載"(MW)

# In[6]:


import numpy as np
import pandas as pd
from keras import backend as K
from keras.models import Sequential 
from keras.layers import Dense


# ## Data Preprocessing

# Read data and parser - 
# 將CSV資料讀入後以逗號拆開，只取包含時間前4個feature

# In[7]:


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


# 取2017和2018年的資料

# In[8]:


tp_2017, target_2017_list = readfile('data/taipower_2017.csv')
tp_2018, target_2018_list = readfile('data/taipower_2018.csv')

#print(tp_2018)


# Parser data ，取去年同一時間和上一週的資料

# In[12]:


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


# In[13]:


train_X = sliceData(tp_2017, tp_2018)
#print(train_X[0])
#print(len(train_X[0]))


# 取得Training target，也就是training data中實際的尖峰負載預測值

# In[14]:


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


# In[15]:


train_Y = getTrainY(target_2018_list)
np_train_X = np.array(train_X)
np_train_Y = np.array(train_Y)


# # Bulid Model
# 以Keras開發，用NN概念實現Linear regression
# 以多層Dense，最後一層output 1x7 向量代表整週的預測

# In[17]:


def bulidModel(x_shape):
    model = Sequential()
    model.add(Dense(units=256, input_shape=(x_shape[1], ), activation='linear'))
    model.add(Dense(units=7,activation='linear'))
    model.summary()
    return model


# # Loss function
# root_mean_squared_error

# In[18]:


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


# In[19]:


LinearModel = bulidModel(np_train_X.shape)
LinearModel.compile(loss=root_mean_squared_error, optimizer='rmsprop')


# In[20]:


train_history = LinearModel.fit(np_train_X , np_train_Y , batch_size=128 , epochs=1000 , verbose=1, validation_split=0.2)


# ## 產生 Predict 2019/04/02 ～ 2019/04/08 所需的Testing feature
# 取 2018/04/02 ～ 2018/04/08 以及網站上的 2019/04/02前一週的資料
# 但沒有2019/04/01所以用2018/04/01代替

# In[23]:


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


# In[24]:


testList = getTestX(tp_2018)
test_2019_list = ['28756', '1887', '6.56', '29140', '1933', '6.63', '30093', '1892', '6.29', '29673', '2054', '6.92', '25810', '2155', '8.35', '24466', '2298', '9.39', '23895', '1655', '6.92']
test_X = testList + test_2019_list
np_test_X = np.array(test_X)
#print(test_X)
#print(len(test_X))


# In[25]:


#print(np_test_X.shape)
test = np_test_X.reshape(1, 42)
#print(test.shape)
test_Y = LinearModel.predict(test)
print(test_Y[0])


# In[26]:


def saveCSV(test_Y):
    date = 20190402
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


# In[27]:


df = saveCSV(test_Y[0])
df.to_csv('submission.csv', index=0)


# 2019.04.01
# P76071200
# 資工所 馬崇堯

# In[ ]:




