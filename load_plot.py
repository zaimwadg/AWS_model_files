import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_TS(name_file,name_TS):
    data = pd.read_csv(name_file)
    return data[name_TS].values

def plot_TS(TS,title,save_fig=False):
    fig = plt.figure()
    fig.set_size_inches( ( 15.4,7) )
    x = range(len(TS))
    y = TS
    plt.axis((-20,len(y)+20,min(y) - 100,max(y) + 100))
    plt.plot(x,y, color='blue')
    plt.ylabel('Value')
    plt.xlabel('Index')
    plt.title(title)
    if save_fig:
        fig.savefig(title+'.pdf', format='pdf')
    plt.show()
    
def plot_values(true_values,predicted_values,title,save_bool = False):
    fig = plt.figure()
    fig.set_size_inches( ( 15.4,7) )
    x = range(len(true_values))
    y_min = min(list(true_values) + list(predicted_values))
    y_max = max(list(true_values) + list(predicted_values))
    plt.axis((-20,len(true_values)+20,y_min - 100,y_max + 100))
    plt.scatter(x,true_values, color='red', label='True values')
    plt.plot(x,predicted_values, color='blue', label='Predicted values')
    plt.legend(loc='upper left')
    plt.title(title)
    plt.ylabel('Value')
    plt.xlabel('Index')
    if save_bool == True:
        fig.savefig(title+'.pdf', format='pdf')
    plt.show()
    
def plot_absolute_error(true_values,predicted_values,title,save_bool = False):
    fig = plt.figure()
    fig.set_size_inches( ( 15.4,7) )
    x = range(len(true_values))
    y = np.abs(true_values - predicted_values)
    plt.axis((-20,len(y)+20,min(y) - 100,max(y) + 100))
    plt.plot(x,y, color='blue')
    plt.title(title)
    plt.ylabel('Absolute error')
    plt.xlabel('Index')
    if save_bool == True:
        fig.savefig(title+'.pdf', format='pdf')
    plt.show()
    
def plot_error(true_values,predicted_values,title,save_bool = False):
    fig = plt.figure()
    fig.set_size_inches( ( 15.4,7) )
    x = range(len(true_values))
    y = true_values - predicted_values
    plt.axis((-20,len(y)+20,min(y) - 100,max(y) + 100))
    plt.plot(x,y, color='blue')
    plt.title(title)
    plt.ylabel('Error')
    plt.xlabel('Index')
    if save_bool == True:
        fig.savefig(title+'.pdf', format='pdf')
    plt.show()
    
def plots_predictions_errors(true_values,predicted_values,title,save_bool = False):
    plot_values(true_values,predicted_values,title,save_bool)
    plot_error(true_values,predicted_values,'Error of ' + title,save_bool)
    plot_absolute_error(true_values,predicted_values,'Absolute error of ' + title,save_bool) 
    
  