#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from numpy import array
import random
# import csv
import math
TRAIN_FILE="mnist_train.csv"
TEST_FILE="mnist_test.csv"
num_layers=2
num_itr=10
neuron_num=784;
hidden_layer_neuron = 100;
result_class=10;
step = 0.5;
bias=0;
model_file="model.csv"

# In[2]:


train_data = np.genfromtxt((TRAIN_FILE), delimiter=',');


# In[5]:


def filldefault(file,layers):
    with open(file,'w') as f:
        for layernum in range(0,layers):
            for i in range(0,neuron_num*hidden_layer_neuron):
                f.write(str(random.uniform(0,1))+"\t");
            f.write('\n');
        f.close()


# In[ ]:


def activation_function(value):  #implement sigmoid for better results
    return 1/(1+math.exp(-(1e-3)*value))


# In[ ]:


def derivative(value):
    return (1-value)*value*(1e-3);


# In[ ]:


def front_prop(inputs,model,layers):
    input_vals = inputs
    # print(input_vals)
    output_vals = np.array([])
    f = np.vectorize(activation_function, otypes=[np.float])
    for layer in range(0,layers):
#         print(layer)
        input_n = len(input_vals)  # number of inputs of this layer
        if(layer == layers - 1):
            output_n = result_class;
        else:
            output_n = hidden_layer_neuron;
        # print(input_n)
        # print(output_n)
        layerweights = model[layer,:input_n*output_n].reshape(output_n,input_n);
        # print(layerweights.shape)
        mult = np.dot(layerweights,input_vals)
        # print(mult.shape)
        outputs = f(mult)
        # print(outputs.shape)
        output_vals = np.append(output_vals,outputs);
        input_vals = outputs;
    return (output_vals); # giving it as tup as final output will have less size and will cause problems.


def delta(outputs,targets,model):
    layers = len(model);
    deltas =np.empty([layers,len(model[0])]); # make a copy for structure
    final_out = outputs[2]
    inputs = outputs[1]
#     print(final_out)
    prev_layer = np.subtract(final_out,targets)
#     print(prev_layer)
    hidden_out = outputs[1]
    for i in range(0,layers):
        newprev = np.zeros(neuron_num)
        prev_out=hidden_out[-(i+1),:]; # outputs of layer prev to this 
        if ( i == 0 ):
            this_out=final_out
            for j in range(0,result_class):
                for k in range(0,neuron_num):
                    p = prev_layer[j]*derivative(this_out[j])*prev_out[k];
                    deltas[-(i+1),k+neuron_num*j] = p;
                    newprev[k]+=prev_layer[j]*derivative(this_out[j])*model[-(i+1),k+neuron_num*j];
            prev_layer = newprev;
        else:
            this_out = hidden_out[-i,:];
            for j in range(0,neuron_num):
                for k in range(0,neuron_num): # wt for k -> j
                    p = prev_layer[j]*derivative(this_out[j])*prev_out[k];
                    deltas[-(i+1),k+neuron_num*j] = p;
                    newprev[k]+=prev_layer[j]*derivative(this_out[j])*model[-(i+1),k+neuron_num*j];
            prev_layer = newprev;
    return deltas;


#%%
def back_prop(outputs,targets,model):
    deltas = delta(outputs,targets,model);
#     print(np.sum(np.subtract(model,deltas)))
    return np.subtract(model,step*deltas);


#%%
def write_back(file,value):
    with open(file,'w') as f:
        for i in range(0,len(value)):
            for j in range(0,len(value[i]) - 1):
                f.write(str(value[i,j])+"\t")
            f.write(str(value[i,-1])+"\n")
        f.close();


#%%
def loss_fn(output,target):
    return 0.5*np.sum(np.power(np.subtract(output,target),2));


#%%
def out(outputs):
    return np.where(outputs==outputs.max())[0][0]

# def mapit()

#%%
def train(iterations,layer,model_param,file):
    itr = iterations
    inputs = train_data[:,1:]
    train_values = train_data[:,0]
    while(itr > 0):
        print('Iteration '+str(iterations-itr + 1))
        outputs = np.apply_along_axis( front_prop, axis=1, arr=inputs,model=model_param,layers=layer );
        print(outputs.shape)
        itr = itr - 1 
    print('Writing trained model to file!')
    # write_back('t.csv',model_param)

# In[ ]:
filldefault('model.csv', 2)
model = np.loadtxt('model.csv')
print(front_prop(train_data[0,1:],model,2)[1].shape)
