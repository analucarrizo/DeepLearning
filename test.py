


from os import system
from torch import empty
import random
import math
from module import Linear, Sequential
from activation_functions import ReLU, LeakyReLU, Tanh, Sigmoid
from losses import MSE
from train import SGD, createData, computeAccuracy, parseModel, train_model, compute_stats, run_stats_model
from cross_validation import crossValidation




def run_model(dictionary):    
    print("************************************************")
    print("************************************************")
    train_input, train_target, test_input, test_target = createData()
    model = parseModel(dictionary)
    train_model(model, dictionary["lr"], dictionary["batch_size"], dictionary["momentum"], dictionary["epochs"], train_input,train_target,test_input,test_target,loss_log = True)
    print(f'train accuracy: {computeAccuracy(model, train_input, train_target)}')
    print(f'test accuracy: {computeAccuracy(model, test_input, test_target)}')
    print("************************************************")
    print("************************************************")



if __name__ == '__main__':
    # model with ReLU activation function and best parameters obtained with cross validation
    models_results = dict()
    print("ReLU model")
    dictionary = dict()
    dictionary["model"] = ["Linear(2,25)","ReLU","Linear(25,25)","ReLU","Linear(25,25)", "ReLU","Linear(25,2)"]
    dictionary["lr"] = 0.01
    dictionary["batch_size"] = 16
    dictionary["momentum"] = 0.8
    dictionary["xavierGain"] = 1
    dictionary["epochs"] = 100
    models_results['ReLU model'] = run_model(dictionary)



    # model with Tanh activation function
    print("Tanh model")
    dictionary["model"] = ["Linear(2,25)","Tanh","Linear(25,25)","Tanh","Linear(25, 25)", "Tanh", "Linear(25,2)"]
    dictionary["lr"] = 0.1
    dictionary["batch_size"] = 50
    dictionary["momentum"] = 0.9
    dictionary["xavierGain"] = 1
    dictionary["epochs"] = 100
    models_results['Tanh model'] = run_model(dictionary)





    # model with Sigmoid activation function
    print("Sigmoid model")
    dictionary["model"] = ["Linear(2,25)","Sigmoid","Linear(25,25)","Sigmoid","Linear(25,25)","Sigmoid", "Linear(25,2)"]
    dictionary["lr"] = 0.1
    dictionary["batch_size"] = 100
    dictionary["momentum"] = 0.9
    dictionary["xavierGain"] = 1.0
    dictionary["epochs"] = 100
    models_results['Sigmoid model'] = run_model(dictionary)






