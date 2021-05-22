#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.metrics import mean_squared_error


# In[2]:


n=1000 #number of data points


# In[3]:


mean=[0,0,0]  #mean-given
cov=[[1,0.8,0.8],[0.8,1,0.8],[0.8,0.8,1]]


# In[4]:


data = np.random.multivariate_normal(mean, cov, n)
print(data)


# In[5]:


print(len(data))


# In[6]:


train_sample=data[0:800]
test_sample=data[800:]


# In[7]:


print(len(train_sample))
print(len(test_sample))


# In[8]:


max=np.amax(train_sample,axis=0)
max_train=np.amax(test_sample,axis=0)

min=np.amin(train_sample,axis=0)
min_train=np.amin(test_sample,axis=0)


# In[9]:


print(max)
print(min)


# In[10]:


train_sample_normalised=(train_sample-min)/(max-min)
test_sample_normalised=(test_sample-min_train)/(max_train-min_train)


# In[11]:


train_sample_normalised


# In[12]:


#2


# In[13]:


learning_rate=0.01
W1=[[1 ,1, 1],
 [1,1 ,1]]
W2=[[1,1],[1,1],[1,1]]
bias1=[[0],[0]]
bias2=[[0],[0],[0]]


# In[14]:


def sig(val):
    res=1/(1+math.exp(-val))
    return res


# In[15]:


def difference(list1,list2):
    difference = []
    zip_object = zip(list1, list2)
    for list1_i, list2_i in zip_object:
        difference.append(list1_i-list2_i)
    return difference    


# In[16]:


def forwardi(X):
    
    #print(X)
    input1=np.dot(W1,np.vstack(X))+bias1
    #print(input1)
    output1 = list(map(sig, input1))
    #print(output1)
    input2=np.dot(W2,np.vstack(output1) )+bias2
    #print("input2"+str(input2))
    output2 = list(map(sig, input2))
    #print("OUTPUT2"+str(output2))
    X_hat=output2
    return X_hat


# In[17]:


def weight_update(X):
    #X=train_sample_normalised[1]
    #print(X)
    print()
    input1=np.dot(W1,np.vstack(X))+bias1
    #print(input1)
    output1 = list(map(sig, input1))
    #print(output1)
    input2=np.dot(W2,np.vstack(output1) )+bias2
    #print("input2"+str(input2))
    output2 = list(map(sig, input2))
    #print("OUTPUT2"+str(output2))
    X_hat=output2
    diff=difference(X_hat,X)
    #print(diff)
    
    sigmai2=list(map(sig, input2))
    diff_o2wrti2=np.vstack(np.array(list(map(sig, input2))*np.array(difference([1,1,1],sigmai2)))) # 3x1
    prod=np.vstack(2*np.array(diff))*diff_o2wrti2 #3x1
    out=[]
    out.append(output1)
    updateW2=np.dot(prod,out)
    W2_star=W2-learning_rate*(updateW2)
    ####
    diff_i2wrto1=W2
    sigmai1=list(map(sig, input1))
    diff_o1wrti1=(np.array(sigmai1)*np.array(difference([1,1,1],sigmai1))) #1x2  
    diff_i1wrtw1=X
    first_three=np.dot(2*np.array(diff)*np.array(list(map(sig, input2))*np.array(difference([1,1,1],sigmai2))),W2) #1 x 2
                    
    intermediate_prod=first_three*np.array(diff_o1wrti1)
    
    diff_i1wrtw1_2D=[]
    diff_i1wrtw1_2D.append(diff_i1wrtw1)
    W1_star=W1-learning_rate*(np.dot(np.vstack(intermediate_prod),diff_i1wrtw1_2D))
    bias1_star=bias1-learning_rate*(np.vstack(intermediate_prod)*[[1],[1]])
    #print("bias1_str"+str(bias1_star))
    bias2_star=bias2-learning_rate*(prod*[[1],[1],[1]])
    #print("bias1_str"+str(bias1_star))
    
    return W1_star,W2_star,bias1_star,bias2_star

    


# In[18]:


loss=[]
loss_train=[]
loss_test=[]
epoch=[]
for i in range(150):
    epoch.append(i)
    x_train_pred=[]
    x_pred=[]
    for i in train_sample_normalised:
        
        #print("FORWARD-TEST")
        y_test=forwardi(i)
        W1,W2,bias1,bias2=weight_update(i)
        x_train_pred.append(y_test)
    loss_train.append(mean_squared_error(train_sample_normalised, x_train_pred))
    for i in test_sample_normalised:
        x=forwardi(i)
        x_pred.append(x)
    loss_test.append(mean_squared_error(test_sample_normalised, x_pred))
        
  
     
        
    


# In[19]:


plt.plot(epoch,loss_train)
plt.show()


# In[20]:


plt.plot(epoch,loss_test)
plt.show()


# In[21]:


plt.plot(epoch,loss_train,label='train') #blue -- training
plt.plot(epoch,loss_test,label='test',color='black') #green -- test
plt.xlabel('epochs')
plt.ylabel('error')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




