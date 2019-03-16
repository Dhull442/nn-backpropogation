#!/usr/bin/env python
# coding: utf-8

# In[42]:


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
result_class=10;
step = 0.5;
bias=0;
model_file="model.csv"


# In[3]:


train_data = np.genfromtxt((TRAIN_FILE), delimiter=',');


# In[21]:


print(train_data.shape)


# `The biases are always 1`

# In[14]:


def filldefault(file,layers):
    with open(file,'w') as f:
        for layernum in range(0,layers):
            if( layernum < layers -1 ):
                # fill initial defaults
                for i in range(0,neuron_num*neuron_num-1):
                    f.write(str(random.uniform(0,1))+",");
                f.write(str(random.uniform(0,1))+"\n");
            else:
                # fill initial defaults
                for i in range(0,neuron_num*result_class):
                    f.write(str(random.uniform(0,1))+",");
                for i in range((neuron_num)*result_class,neuron_num*(neuron_num) -1):
                    f.write("0,")
                f.write("0\n")
        f.close()


# The weights are from first class to next one. first val ~= input1 -> hidden(1,1) . second val ~= input2 -> hidden(1,1)
# Biases come after all the weights

# In[66]:


def activation_function(value):  #implement sigmoid for better results
    return 1/(1+math.exp(-(1e-3)*value))


# In[67]:


def derivative(value):
    return (1-value)*value*(1e-3);


# In[75]:


def front_prop(inputs,model,layers):
    input_vals = inputs
    output_vals = np.array([input_vals])
    for layer in range(0,layers):
#         print(layer)
        input_n = len(input_vals)  # number of inputs of this layer
        if(layer == layers - 1):
            output_n = result_class;
        else:
            output_n = neuron_num;  # number of outputs of this layer
        outputs = np.zeros(output_n) # outputs of this particular layer
        for i in range(0,output_n):
            start = i*input_n;
            end = (i+1)*input_n;
            weights = model[layer,start:end];
            net = np.sum(np.multiply(input_vals,weights)) + bias;
            outputs[i] = activation_function(net);
        if(layer < layers -1):
            output_vals = np.vstack([output_vals,np.array([outputs])]);
        input_vals = outputs;
    return (output_vals,outputs); # giving it as tup as final output will have less size and will cause problems.


# In[74]:


def delta(outputs,targets,model):
    layers = len(model);
    deltas =np.empty([layers,len(model[0])]); # make a copy for structure
    final_out = outputs[1]
#     print(final_out)
    prev_layer = np.subtract(final_out,targets)
#     print(prev_layer)
    hidden_out = outputs[0]
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


# In[44]:


def back_prop(outputs,targets,model):
    deltas = delta(outputs,targets,model);
#     print(np.sum(np.subtract(model,deltas)))
    return np.subtract(model,step*deltas);


# In[10]:


def write_back(file,value):
    with open(file,'w') as f:
        for i in range(0,len(value)):
            for j in range(0,len(value[i]) - 1):
                f.write(str(value[i,j])+",")
            f.write(str(value[i,-1])+"\n")
        f.close();


# In[11]:


def loss_fn(output,target):
    return 0.5*np.sum(np.power(np.subtract(output,target),2));


# In[12]:


def out(outputs):
    return np.where(outputs==outputs.max())


# In[86]:


def train(itr,layer,model_param,file):
    while(itr > 0):
        print('Iteration '+str(itr))
#         a = np.zeros(len(train_data))
        for sample in range(0,len(train_data)):  # for testing only :C
            targets = np.zeros(result_class)
            targets[train_data[sample,0].astype(int)] = 1;
            outputs = front_prop(train_data[sample,1:],model_param,layer)   # outputs is a tuple
            net_error = loss_fn(outputs[1],targets)
            new_model = back_prop(outputs,targets,model_param)
            model_param = new_model
#             a[sample] = net_error
            print('ITR '+str(itr)+', ERROR FOR SAMPLE NO. '+str(sample)+' : ' + str(net_error))
        itr = itr -1;
        write_back('trained'+str(itr)+'.csv',model_param)
    print('Writing trained model to file!')


# Model file contains all rows having same number of entries, in last row junk entries are fed!

# In[32]:


print(model_param.shape)


# In[29]:


filldefault("model.csv",num_layers);  ## Reset model to initial


# In[87]:


model_param = np.genfromtxt(model_file,delimiter=",");


# In[77]:


len(train_data)


# In[ ]:


train(10,2,model_param,model_file)


# In[290]:


def test(tests,layer,model,outFile):
    with open(outFile,'w')  as f:
        for i in range(0,len(tests)):
            result = out(front_prop(tests[i,:],model,layer)[1])
            f.write(str(i+1)+","+str(result)+"\n");
        f.close()


# In[291]:


test_file = np.genfromtxt(TEST_FILE,delimiter=",");


# In[292]:


model=np.genfromtxt(model_file,delimiter=",")


# In[293]:


test(test_file,2,model,'submit.csv')


# In[22]:


1e-7

