
import math
from torch import empty
from module import Linear, Sequential
from activation_functions import ReLU, Tanh, LeakyReLU, Sigmoid
from losses import MSE

#SGD class
class SGD():
    def __init__(self,layers,learning_rate,momentum = 0):
        #initilization: layers is all the layers of the model
        self.layers = layers
        self.learning_rate = learning_rate
        self.momentum = momentum
        # tuple for update of weights and bias in linear levels
        self.velocity = list((0,0) for i in range(0, len(layers)))

    def step(self):
        #update of velocity for parameters that require a gradient (therefore only the Linear levels)
        for i in range(len(self.layers)):
            if self.layers[i].param() is not None:
                if type(self.layers[i]) == type(Linear(1,1)):
                    if self.momentum != 0:
                        # velocity update is the same as specified on pytorch:
                        #https://pytorch.org/docs/stable/optim.html 
                        #Velocity_(t+1) = momentum * Velocity_(t) + gradient of layer
                        #Parameters_(t+1) = Parameters_(t) - lr * Velocity_(t+1)-->> update done within class of layer (calling step)
                        self.velocity[i] = (self.momentum * self.velocity[i][0] + self.layers[i].param()[0][1],self.momentum * self.velocity[i][1] + self.layers[i].param()[1][1])
                        self.layers[i].step(self.learning_rate*self.velocity[i][0],self.learning_rate*self.velocity[i][1])
                    else:
                        self.layers[i].step(self.learning_rate,self.learning_rate)

#function generating data
def createData(nb_data = 1000):
    # train data
    train_input = empty((nb_data,2)).uniform_(0,1)
    tmp = train_input.clone()

    train_target = tmp.apply_(lambda x: math.pow(x-0.5,2)).sum(1)
    train_target[train_target > 1/(2*math.pi)] = 5
    train_target[train_target <= 1/(2*math.pi)] = 1
    train_target[train_target == 5] = 0

    one_hot_train_target = empty((nb_data,2)).fill_(0)

    #one hotting targets
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
    #one hotting targets
    for i in range(test_target.shape[0]):
        if test_target[i] == 1:
            one_hot_test_target[i][1] = 1
        else:
            one_hot_test_target[i][0] = 1

    return train_input,one_hot_train_target,test_input,one_hot_test_target


def train_model(model,lr,batch_size,momentum,epochs,train_input,train_target,test_input,test_target, loss_log = False):
    losses = []
    validation_input = test_input
    valdidation_target = test_target
    validation_loss = []
    train_loss = []
    validation_acc = []
    train_acc = []
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
            #model forward pass
            pred = model.forward(train_input.narrow(0, b, mini_batch_size))
            ## computing loss
            loss = criterion.compute_loss(pred,train_target.narrow(0, b, mini_batch_size))
            acc_loss += loss.item()
            #compute gradient of loss
            grad_wrt_outputs = criterion.backward()
            #backpropagate with gradient of loss
            model.backward(grad_wrt_outputs)
            #update parameters of model
            optimizer.step()
            #reset gradient matrices to 0
            model.zero_grad()
    
        batches = int(train_input.size(0)/ mini_batch_size) if train_input.size(0)% mini_batch_size == 0 else int(train_input.size(0)/ mini_batch_size)+1
        #keeping track of loss accuracy after one epoch
        if loss_log: 
            print(epoch, acc_loss/batches)
            pred_train = model.forward(train_input)
            pred_test = model.forward(test_input)
            train_loss.append(MSE().compute_loss(pred_train,train_target).item())
            validation_loss.append(MSE().compute_loss(pred_test,test_target).item())
            train_acc.append(computeAccuracy(model,train_input,train_target))
            validation_acc.append(computeAccuracy(model,test_input,test_target)) 
            model.zero_grad()
    return train_loss,validation_loss,train_acc,validation_acc

#computes the accuracy 
def computeAccuracy(model, input, target):
    pred = (model.forward(input)).argmax(dim = 1)
    actual = target.argmax(dim = 1)
    return (pred[pred == actual].shape[0]/pred.shape[0])


#description: given a dictionary containing a field 'model', this function parses the model and creates it
## input: dictionary: dictionary containing field model which contains a list of strings describing the model
                            ## example: if given :dictionary['model'] = ["Linear(2,25),"ReLU","Linear(25,2)"],
                                    ##  The function will parse this list of strings to transform it into a model and then return it
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

### function to compute std and mean of losses and accuracies after having trained a model 10 times
## input: results of multiple runs of a model:
            ##train_losses and test_loss: list of losses after every epoch. Each element of the list is the sum 
                                            #of the losses over 10 trainings of the model for a given epoch
        
            ## train_accuracies and test_accuracies: list of accuracies after every epoch. Each element of the list is the sum 
                                            #of the losses over 10 trainings of the model for a given epoch
                
            ##train_accuracy and test_accuracy: list containing the final accuracy of the model after each training
            
            #final_train_loss and final_test_loss: list containing the final loss of the model after each training
def compute_stats(train_losses,test_losses,train_accuracies,test_accuracies,train_accuracy,test_accuracy,final_train_loss,final_test_loss,validation_loss):
    avg_train_losses = list(map(lambda x: x/10,train_losses))
    avg_test_losses = list(map(lambda x: x/10,test_losses))
    avg_train_accuracies = list(map(lambda x: x/10,train_accuracies))
    avg_test_accuracies = list(map(lambda x: x/10,test_accuracies))

    avg_train_accuracy = sum(train_accuracy)/10
    avg_test_accuracy = sum(test_accuracy)/10
    avg_final_train_loss = sum(final_train_loss)/10
    avg_final_test_loss = sum(final_test_loss)/10

    final_train_loss_std =  sum(list(map(lambda x: (x-avg_final_train_loss)**2,final_train_loss)))/10
    final_test_loss_std =  sum(list(map(lambda x: (x-avg_final_test_loss)**2,final_test_loss)))/10
    final_train_acc_std = sum(list(map(lambda x: (x-avg_train_accuracy)**2,train_accuracy)))/10
    final_test_acc_std = sum(list(map(lambda x: (x-avg_test_accuracy)**2,test_accuracy)))/10
    return (avg_train_losses,avg_test_losses,avg_train_accuracies,avg_test_accuracies,avg_train_accuracy,avg_test_accuracy,avg_final_train_loss,avg_final_test_loss,final_train_loss_std,final_test_loss_std,final_train_acc_std,final_test_acc_std)


### function to compute std and mean of losses and accuracies after having trained a model 10 times (complement of compute_stats),
##runs model 10 times and then returns std and mean of losses and accuracies.
        ##input: dictionary containing lr,epochs,batch size and momentum and xavier gain and a list of strings describing the model
def run_stats_model(dictionary):
    train_accuracy = []
    test_accuracy = []
    train_losses = []
    final_train_loss = []
    final_test_loss = []
    train_accuracies = []
    test_accuracies = []
    for i in range(10):
        #create new data
        train_input, train_target, test_input, test_target = createData()
        #create model based on list of strings given in field dictionary['model]
        model = parseModel(dictionary)
        #train model
        train_loss,validation_loss,train_acc,validation_acc = train_model(model,dictionary["lr"],dictionary["batch_size"],dictionary["momentum"], dictionary["epochs"],train_input,train_target,test_input,test_target, loss_log = True)
        
        #compute final accuracies
        tr_accuracy = computeAccuracy(model, train_input, train_target)
        te_accuracy = computeAccuracy(model, test_input, test_target)
        
        #compute final losses
        fi_train_loss = train_loss[-1]
        fi_test_loss = validation_loss[-1]
        
        train_accuracy.append(tr_accuracy)
        test_accuracy.append(te_accuracy)
        final_train_loss.append(fi_train_loss)
        final_test_loss.append(fi_test_loss)
        ##creating lists necessary for calling compute_stats function
        if i == 0:
            train_losses = train_loss.copy()
            test_losses = validation_loss.copy()
            train_accuracies = train_acc.copy()
            test_accuracies = validation_acc.copy()
        else:
            for j in range(len(train_loss)):
                train_losses[j] += train_loss[j]
                test_losses[j] += validation_loss[j]
                train_accuracies[j] += train_acc[j]
                test_accuracies[j] += validation_acc[j]
    avg_train_losses,avg_test_losses,avg_train_accuracies,avg_test_accuracies,avg_train_accuracy,avg_test_accuracy,avg_final_train_loss,avg_final_test_loss,final_train_loss_std,final_test_loss_std,final_train_acc_std,final_test_acc_std = compute_stats(train_losses,test_losses,train_accuracies,test_accuracies,train_accuracy,test_accuracy,final_train_loss,final_test_loss,validation_loss)
    return {'avg_train_losses': avg_train_losses,'avg_test_losses': avg_test_losses,'avg_train_accuracies':avg_train_accuracies,'avg_test_accuracies':avg_test_accuracies,'avg_train_accuracy': avg_train_accuracy,'avg_test_accuracy':avg_test_accuracy,'avg_final_train_loss':avg_final_train_loss,'avg_final_test_loss':avg_final_test_loss,'final_train_loss_std':final_train_loss_std,'final_test_loss_std':final_test_loss_std,'final_train_acc_std':final_train_acc_std,'final_test_acc_std': final_test_acc_std}
        
