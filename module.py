#!/usr/bin/env python
# coding: utf-8

# In[2]:


from torch import empty
import math



class Module(object):
    """
    Module class - interface that all other model architecture classes in the framework         should inherit
    """

    def __init__(self):
         pass

    def forward(self, *arguments):
         raise NotImplementedError

    def backward(self, *gradwrtoutput):
         raise NotImplementedError

    def param(self):
         return []

    def step(self,lr):
         raise NotImplementedError

    def zero_grad(self):
         raise NotImplementedError

    def reset_weights(self,*parameters):
         raise NotImplemented



class Linear(Module):
    ''' Linear fully-connected layer class '''

    def __init__(self, in_size, out_size,gain = 1):
        ''' Linear Constructor 

        :param in_size: size of the input, positive integer
        :param out_size: size of the output, positive integer

        :raises ValueError if: 
          - in_size is not a positive integer
          - out_size is not a positive integer
        '''

        if not isinstance(in_size, int) or in_size <= 0:
             raise ValueError("The input size must be a positive integer")

        if not isinstance(out_size, int) or out_size <= 0:
             raise ValueError("The output size must be a positive integer")

        self.weights = empty((out_size,in_size))
        self.bias = empty((out_size))

        ##Xavier weight initialization
        variance = 2. / (in_size + out_size) #from dlc-handout-5-5-initialization.pdf of course slide 13
        std = gain * math.sqrt(variance)
        self.weights.uniform_(-std, std)
        self.bias.uniform_(-std, std)

        self.in_value = empty(1)

        self.gradWeights = empty((out_size,in_size)).fill_(0.)
        self.gradBias = empty((out_size)).fill_(0.)

    def forward(self, x):
        ''' Linear fully connected forward pass

        :param x: output of the previous layer, torch.Tensor
        :returns: Linear()

        '''
        self.in_value = x.clone()

        if x.shape[1] != self.weights.shape[1]:
            print(f'input dimension of x (dim = {x.shape[0]}) does not match weight matrix (dim = {self.weights.shape[1]} ')
            raise
        return (x @ self.weights.T) + self.bias

    def backward(self, *gradwrtoutput):
        ''' Linear fully connected backward pass
        :param x: gradient of the next layer, torch.Tensor
        :returns: Linear()
        '''

        self.gradWeights = self.gradWeights+ gradwrtoutput[0].T @ self.in_value
        self.gradBias = self.gradBias + gradwrtoutput[0].sum(0)
        return  gradwrtoutput[0] @ self.weights

    def param(self): 
        '''
        Retrieves parameters from all layers 

        :returns: list of Module.param
        ''' 
        return [(self.weights,self.gradWeights),(self.bias,self.gradBias)]

    def step(self,lr_weight,lr_bias):
        self.weights = self.weights - lr_weight
        self.bias = self.bias - lr_bias

    def zero_grad(self):
        self.gradWeights.fill_(0)
        self.gradBias.fill_(0)

    def reset_weights(self,*parameters):
        in_size = self.weights.shape[1]
        out_size = self.weights.shape[0]
        self.weights = empty((out_size,in_size))
        self.bias = empty((out_size))
        gain = parameters[0]
        ##Xavier weight initialization
        variance = 2. / (in_size + out_size) #from dlc-handout-5-5-initialization.pdf of course slide 13
        std = gain * math.sqrt(variance)
        self.weights.uniform_(-std, std)
        self.bias.uniform_(-std, std)

        self.in_value = empty(1)

        self.gradWeights = empty((out_size,in_size)).fill_(0.)
        self.gradBias = empty((out_size)).fill_(0.)



class Sequential(Module):
    def __init__(self,layers_list):
        self.layers = layers_list
        self.output = empty(1)


    def forward(self,x):
        output = x.clone()
        for layer in self.layers:
             output = layer.forward(output)
        return output

    def backward(self, *gradwrtoutput):
        grad = gradwrtoutput[0].clone()
        for layer in self.layers[::-1]:
             grad = layer.backward(grad)

    def param(self):
        parameters=[]
        for layer in self.layers:
             parameters.append(layer.param())
        return parameters

    def step(self,lr):
        return

    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()

    def reset_weights(self,*parameters):
        for layer in self.layers:
            layer.reset_weights(parameters[0])
        return 
