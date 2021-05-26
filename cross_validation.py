from module import Linear, Sequential
from activation_functions import ReLU, Tanh, LeakyReLU, Sigmoid 
from train import parseModel, createData, train_model, computeAccuracy
from torch import empty

## function used to cross validate (3 folds)
        ##input: dictionary containg all lr,batch_sizes,momentums,xavierGain,epochs the user wants to cross validate on
def crossValidation(dictionary):
    if dictionary.get("model", None) == None:
        print("arguments missing (model)")
        return
    if dictionary.get("lr", None) == None:
        print("arguments missing (lr)")
        return
    if dictionary.get("batch_size") == None:
        print("arguments missing (batch_size)")
        return
    if dictionary.get("momentum") == None:
        print("arguments missing (momentum)")
        return
    if dictionary.get("xavierGain",None) == None:
        print("arguments missing (xavierGain)")
        return
    if dictionary.get("epochs",None) == None:
        print("arguments missing (epochs)")
        return

    model = parseModel(dictionary,crossval= True)
    
    for layer in model.layers:
        print(layer)
    
    lrs = dictionary["lr"]
    batch_sizes = dictionary["batch_size"]
    momentums = dictionary["momentum"]
    xavierGains = dictionary["xavierGain"]
    epochs = dictionary["epochs"]
    
    best_param = {'lr': None,'batch_size': None,'momentum': None,'xavierGain': None, 'epochs': None ,'accuracy': 0.0}
    
    fold1_input,fold1_target,fold2_input,fold2_target = createData(nb_data = 500)
    fold3_input,fold3_target,_,_ = createData(nb_data=500)
    folds_inputs = [fold1_input,fold2_input,fold3_input]
    folds_targets = [fold1_target,fold2_target,fold3_target]
    
    for epoch in epochs:
        for lr in lrs:
            for batch_size in batch_sizes:
                for momentum in momentums:
                    for xavierGain in xavierGains:
                        mean_accuracy = 0
                        k=0
                        print(f'epoch: {epoch}, lr:{lr} batch_size: {batch_size} momentum: {momentum} xavierGain:{xavierGain}')
                        for i in range(len(folds_inputs)):
                            for j in range(i+1,len(folds_inputs)):
                                model.reset_weights(xavierGain)
                                train_input = empty((folds_inputs[i].shape[0]*2,folds_inputs[i].shape[1]))
                                train_input[:folds_inputs[i].shape[0]] = folds_inputs[i]
                                train_input[folds_inputs[i].shape[0]:] = folds_inputs[j]

                                train_target = empty((folds_targets[i].shape[0]*2,folds_targets[i].shape[1]))
                                train_target[:folds_targets[i].shape[0]] = folds_targets[i]
                                train_target[folds_targets[i].shape[0]:] = folds_targets[j]

                                if i == 0 and j == 1:
                                    k = 2
                                elif i==0 and j==2:
                                    k = 1
                                else:
                                    k = 0

                                test_input = folds_inputs[k].clone()
                                test_target = folds_targets[k].clone()
                                train_model(model,lr,batch_size,momentum,epoch,train_input,train_target)

                                accuracy = computeAccuracy(model,test_input,test_target)
                                print("********************************************")
                                print(f'fold accuracy: {accuracy}')
                                print("********************************************")
                                mean_accuracy += accuracy
                            
                            mean_accuracy = mean_accuracy/3
                            if mean_accuracy > best_param['accuracy']:
                                best_param = {'lr': lr,'batch_size': batch_size,'momentum': momentum,'xavierGain': xavierGain, 'epochs': epoch ,'accuracy': mean_accuracy}
    return best_param