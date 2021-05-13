#!/usr/bin/env python
# coding: utf-8

# In[3]:
import math
from torch import empty
from module import Linear, Sequential
from activation_functions import ReLU, Tanh, LeakyReLU, Sigmoid
from losses import MSE

class SGD():
    def __init__(self,layers,learning_rate,momentum = 0):
        self.layers = layers
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = list((0,0) for i in range(0, len(layers)))

    def step(self):
        for i in range(len(self.layers)):
            if self.layers[i].param() is not None:
                if type(self.layers[i]) == type(Linear(1,1)):
                    #print(self.momentum * self.velocity[i][1] + self.layers[i].param()[1][1])
                    #print(self.layers[i].param()[1][1])
                    if self.momentum != 0:
                        self.velocity[i] = (self.momentum * self.velocity[i][0] + self.layers[i].param()[0][1],self.momentum * self.velocity[i][1] + self.layers[i].param()[1][1])
                        self.layers[i].step(self.learning_rate*self.velocity[i][0],self.learning_rate*self.velocity[i][1])
                    else:
                        self.layers[i].step(self.learning_rate,self.learning_rate)
                        #print("velocity")
                        #print(self.velocity)


# In[ ]:

def createData(nb_data = 1000):
    # train data
    train_input = empty((nb_data,2)).uniform_(0,1)
    tmp = train_input.clone()
    
    train_target = tmp.apply_(lambda x: math.pow(x-0.5,2)).sum(1)
    train_target[train_target > 1/(2*math.pi)] = 5
    train_target[train_target <= 1/(2*math.pi)] = 1
    train_target[train_target == 5] = 0
    
    one_hot_train_target = empty((nb_data,2)).fill_(0)
    
    for i in range(train_target.shape[0]):
        if train_target[i] == 1:
             one_hot_train_target[i][1] = 1
        else:
             one_hot_train_target[i][0] = 1

    # test data
    test_input = empty((nb_data,2)).uniform_(0,1)
    tmp = test_input.clone()
    
    test_target = tmp.apply_(lambda x: math.pow(x-0.5,2)).sum(1)
    test_target[test_target > 1/(2*math.pi)] = 5
    test_target[test_target <= 1/(2*math.pi)] = 1
    test_target[test_target == 5] = 0
    
    one_hot_test_target = empty((nb_data,2)).fill_(0)
    
    for i in range(test_target.shape[0]):
        if test_target[i] == 1:
            one_hot_test_target[i][1] = 1
        else:
            one_hot_test_target[i][0] = 1

    return train_input,one_hot_train_target,test_input,one_hot_test_target


def train_model(model,lr,batch_size,momentum,epochs,train_input,train_target, loss_log = False):
    losses = []
    optimizer = SGD(model.layers,learning_rate = lr, momentum = momentum)
    criterion = MSE()
    batch_size = batch_size
    
    for epoch in range(epochs):
        acc_loss = 0
        
        for b in range(0, train_input.size(0), batch_size):
            if b + batch_size > train_input.size(0):
                mini_batch_size = train_input.size(0) - b
            else:
                mini_batch_size = batch_size
            
            pred = model.forward(train_input.narrow(0, b, mini_batch_size))
            loss = criterion.compute_loss(pred,train_target.narrow(0, b, mini_batch_size))
            acc_loss += loss.item()
            grad_wrt_outputs = criterion.backward()
            model.backward(grad_wrt_outputs)
            optimizer.step()
            model.zero_grad()
    
        batches = int(train_input.size(0)/ mini_batch_size) if train_input.size(0)% mini_batch_size == 0 else int(train_input.size(0)/ mini_batch_size)+1
        if loss_log: 
            print(epoch, acc_loss/batches)
            losses.append(acc_loss/batches)
    return losses

def computeAccuracy(model, input, target):
    pred = (model.forward(input)).argmax(dim = 1)
    actual = target.argmax(dim = 1)
    return (pred[pred == actual].shape[0]/pred.shape[0])

def parseModel(dictionary, crossval = False):
    layers = []
    for element in dictionary["model"]:
        if "Linear" in element:
            in_size,out_size = element[7:-1].split(",")
            in_size = int(in_size)
            out_size = int(out_size)
            if crossval == False:
                layers.append(Linear(in_size,out_size, dictionary["xavierGain"]))
            else: 
                layers.append(Linear(in_size,out_size))
        elif "LeakyReLU" in element:
            layers.append(LeakyReLU())
        elif "ReLU" in element:
            layers.append(ReLU())
        elif "Tanh" in element:
            layers.append(Tanh())
        elif "Sigmoid" in element:
            layers.append(Tanh())
        else:
            print(f'{elem} is an invalid argument')
    return Sequential(layers)