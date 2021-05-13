#!/usr/bin/env python
# coding: utf-8

# In[3]:

from torch import empty
import math

class MSE():
    def __init__(self):
        self.diff = 0

    def compute_loss(self,pred,actual):
        n = actual.shape[0]*actual.shape[1]
        self.diff = (pred - actual).clone()
        #print("max")
        #print(torch.max(self.diff))
        loss = ((pred - actual).apply_(lambda x: math.pow(x,2)).sum())/(n)
        return loss

    def backward(self):
        #print("shape")
        #print((2*self.diff/self.diff.shape[0]).shape)
        n = self.diff.shape[0]*self.diff.shape[1]
        return 2*self.diff/n

