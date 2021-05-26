

from module import Module
from torch import empty
import math


class Tanh(Module):
    ''' Class performing Tanh activation '''

    def __init__(self):
        '''
            Tanh constructor
        '''

        self.in_value = empty(1)
        self.grad_value = empty(1)

    def derivative(self, x):
        return (1.0-math.pow(math.tanh(x),2))

    def forward(self, x) :
        '''
        Tanh forward pass

        :pararm x: output from the previous layer, torch.Tensor

        :returns:  
        '''
        #tmp = x.clone()
        self.in_value = x.clone()
        tmp = x.clone()
        return tmp.apply_(math.tanh)

    def backward (self, *gradwrtoutput):
        '''
        Tanh backward pass

        :param gradwrtoutput: gradient of the next layer, torch.Tensor

        :returs: 
        '''
        der = self.in_value.clone()
        der = der.apply_(self.derivative)
        return der * gradwrtoutput[0]

    def param ( self ) :
        return []

    def step(self,lr):
        return

    def zero_grad(self):
        return

    def reset_weights(self,*parameters):
        return

class ReLU(Module) :
    ''' Class performing ReLU activation '''

    def __init__(self):
        ''' 
        ReLU constructor
        '''

        self.in_value = empty(1)

    def forward(self, x) :
        ''' 
        ReLU forward pass

        :param x: output of the previous layer, torch.Tensor

        :returns: 
        '''
        #tmp = x.clone()
        self.in_value = x.clone()
        tmp = x.clone()
        tmp[tmp < 0] = 0.0
        return tmp

    def backward ( self , * gradwrtoutput ):
        ''' 
        ReLU backward pass

        :param gradwrtoutput: gradient of the next layer, torch.Tensor

        :returns: 
        '''

        der = self.in_value.clone()
        der[der < 0] = 0
        der[der > 0] = 1
        return der * gradwrtoutput[0]

    def param ( self ) :
        return []

    def step(self,lr):
        return

    def zero_grad(self):
        return

    def reset_weights(self,*parameters):
        return

class LeakyReLU(Module) :
    ''' Class performing LeakyReLU activation '''

    def __init__(self, alpha = 0.01):
        ''' 
        LeakyReLU constructor
        '''

        self.in_value = empty(1)
        self.alpha = alpha

    def forward(self, x) :
        ''' 
        LeakyReLU forward pass

        :param x: output of the previous layer, torch.Tensor

        :returns: 
        '''
        #tmp = x.clone()
        self.in_value = x.clone()
        tmp = x.clone()
        tmp[tmp < 0] =self.alpha * tmp[tmp < 0]
        return tmp

    def backward ( self , * gradwrtoutput ):
        ''' 
        LeakyReLU backward pass

        :param gradwrtoutput: gradient of the next layer, torch.Tensor

        :returns: 
        '''

        der = self.in_value.clone()
        der[der < 0] = self.alpha
        der[der > 0] = 1
        return der * gradwrtoutput[0]

    def param ( self ) :
        return []

    def step(self,lr):
        return

    def zero_grad(self):
        return

    def reset_weights(self,*parameters):
        return

class Sigmoid(Module):
    ''' Class performing sigmoid activation '''

    def __init__(self):
        ''' Sigmoid constructor '''

        self.in_value = empty(1)
        self.grad_value = empty(1)

    def sigmoid(x):
        return 1/(1+math.exp(-x))

    def derivative(x):
        return (sigmoid(x) * (1 - sigmoid(x)))

    def forward(self, x):
        '''
        Sigmoid forward pass

        :param x: output from the previous layer, torch.Tensor

        :returns: 
        '''
        self.in_value = x.clone()
        return tmp.apply_(sigmoid)

    def backward(self, *gradwrtoutput):
        '''
        Sigmoid backward pass

        :param gradwrtoutput: gradient of the next layer, torch.Tensor

        :returns: 
        '''

        der = self.in_value.clone()
        der = der.apply_(derivative)
        return der * gradwrtoutput[0]
    def param():
        return []

    def step(self,lr):
        return

    def zero_grad(self):
        return

    def reset_weights(self,*parameters):
        return