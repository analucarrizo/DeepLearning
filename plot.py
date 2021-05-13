#!/usr/bin/env python
# coding: utf-8

# In[1]:


import plotly.express as px
import pandas as pd


# In[11]:


# input_: train_input or test_input
# target_: train_target or test_target
# type_: string, either train or test
def visualize_model(model, input_, target_, type_):
    pred = (model.forward(input_)).argmax(dim = 1)
    actual = target_.argmax(dim = 1)
    print(pred[pred == actual].shape[0]/pred.shape[0])
    d = {'x': input_[:,0], 'y':  input_[:,1],'pred': pred,'actual': actual}
    df = pd.DataFrame(data=d)
    fig = px.scatter(df, x=df.x, y=df.y, color=df.pred, title = "Visualization of the: " + type_+ " set")
    fig.update_xaxes(type='linear')
    fig.update_yaxes(type='linear')
    fig.update_yaxes(tick0=0.0, dtick=0.5)
    fig.update_xaxes(tick0=0.0, dtick=0.5)
    fig.update_layout(width = 800, height = 800)
#     fig.title("Visualization of the: ", type_, "set")
    fig.show()


# In[9]:


# pred = model.forward(test_input).argmax(dim = 1)
# actual = test_target.argmax(dim = 1)
# print(pred[pred == actual].shape[0]/pred.shape[0])
# d = {'x': train_input[:,0], 'y':  train_input[:,1],'pred': pred,'actual': actual}
# df = pd.DataFrame(data=d)
# fig = px.scatter(df, x=df.x, y=df.y, color=df.pred)
# fig.update_xaxes(type='linear')
# fig.update_yaxes(type='linear')
# fig.update_yaxes(tick0=0.0, dtick=0.5)
# fig.update_xaxes(tick0=0.0, dtick=0.5)
# fig.update_layout(width = 800, height = 800)
# fig.show()

