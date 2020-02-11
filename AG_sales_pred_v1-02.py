# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 20:30:03 2020

@author: Anthony Paech 2016
"""

# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# To plot pretty figures
#%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rnn"



import numpy as np
import pandas as pd
import csv
import sys
import datetime as dt
import joblib
import pickle

import sales_regression_cfg as cfg

from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder


#import matplotlib.pyplot as plt

import timeit

from collections import Counter,OrderedDict
 


def save_model(model,filename):
    #filename = 'finalized_model.sav'
    #    joblib.dump(regressor,open("SGDRegressorNS.p","wb"))

    joblib.dump(model, filename)
    return 

def load_model(filename):
    # some time later...

    # load the model from disk

    loaded_model = joblib.load(filename)
    return loaded_model






def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)
    
    
    
def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch
        
        
        
try:
  import tensorflow.compat.v1 as tf
except Exception:
  pass

# #tf.enable_eager_execution()
# tf.compat.v1.enable_eager_execution(
#     config=None,
#     device_policy=None,
#     execution_mode=None
# )

# tf.enable_v2_behavior()



#import tensorflow as tf

#import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print("tensorflow:",tf.__version__)
   



#t_min, t_max = 0, 30
#resolution = 0.1

t_min, t_max = 0, 30
resolution = 0.1



def time_series(t):
    return t * np.sin(t) / 3 + 2 * np.sin(t*5)

def next_batch(batch_size, n_steps):
    t0 = np.random.rand(batch_size, 1) * (t_max - t_min - n_steps * resolution)
    Ts = t0 + np.arange(0., n_steps + 1) * resolution
    ys = time_series(Ts)
    return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1)








def main():


#    f=open(cfg.outfile_predict,"w")

    print("\n\nSales prediction tool using a neural network - By Anthony Paech 2/2/20\n")
#    print("Loading",sys.argv[1],"into pandas for processing.....")
   # print("Loading",cfg.datasetworking,"into pandas for processing.....")

 #   print("Results Reported to",cfg.outfile_predict)

 #   f.write("\n\nsales prediction tool - By Anthony Paech 2/2/20\n\n")
 #   f.write("Loading "+sys.argv[1]+" into pandas for processing.....\n")
#    f.write("Loading "+cfg.datasetworking+" into pandas for processing.....\n")

#    f.write("Results Reported to "+cfg.outfile_predict+"\n")

 #   df=read_excel(sys.argv[1],-1)  # -1 means all rows
    df=pd.read_excel("NAT-raw310120all.xlsx",-1)  # -1 means all rows

    if df.empty:
        #print(sys.argv[1],"Not found.")
        print(cfg.infilename,"Not found.")

        sys.exit()

 #   print(df)
    
#############################################################################3
    #  create a pivot table of code, product, day delta and predicted qty and export back to excel

    df["month"]=df.date.dt.to_period('M')

#    print(df)

    mask=(df["product"]=="SJ300")
    table = pd.pivot_table(df[mask], values='qty', index=['product'],columns=['month'], aggfunc=np.sum, margins=False, fill_value=0)   #, observed=True)
    print("\ntable=\n",table.head(5))
 #   f.write("\n\n"+table.to_string())
    print("table created.")

    tsales=table.to_numpy()
    print(tsales)
    print(tsales.shape)

    sales=tsales.ravel()
    print(sales)
    print(sales.shape)
 
    print("\nWriting to Excel: tabletest.xlsx")
 
    with pd.ExcelWriter("tabletest.xlsx") as writer:  # mode="a" for append
        table.to_excel(writer,sheet_name="Monthly unit sales")
        print("table of monthly unit sales written to excel.")

####################





##    t = np.linspace(t_min, t_max, int((t_max - t_min) / resolution))
##    print("t")
##    print(t)
##    print(t.shape)
##    
##    n_steps = 15
##    t_instance = np.linspace(12.2, 12.2 + resolution * (n_steps + 1), n_steps + 1)
##
##    print("t_instance")
##    print(t_instance)
##    print(t_instance.shape)
##    print("\n\n")
##
##    plt.figure(figsize=(11,4))
##    plt.subplot(121)
##    plt.title("A time series (generated)", fontsize=14)
##    plt.plot(t, time_series(t), label=r"$t . \sin(t) / 3 + 2 . \sin(5t)$")
##    plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "b-", linewidth=3, label="A training instance")
##    plt.legend(loc="lower left", fontsize=14)
##    plt.axis([0, 30, -17, 13])
##    plt.xlabel("Time")
##    plt.ylabel("Value")
##
##    plt.subplot(122)
##    plt.title("A training instance", fontsize=14)
##    plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
##    plt.plot(t_instance[1:], time_series(t_instance[1:]), "w*", markersize=10, label="target")
##    plt.legend(loc="upper left")
##    plt.xlabel("Time")
##
##
##    #save_fig("time_series_plot")
##    plt.show()

################################


 #   t = np.linspace(t_min, t_max, int((t_max - t_min) / resolution))

    monthno=np.arange(0,sales.shape[0]).ravel()
    print("monthno")
    print(monthno)
    print(monthno.shape)

    m_start=2
    n_steps = 15+m_start
    monthno_instance = np.arange(m_start, n_steps+1).ravel()

    print("monthno_instance")
    print(monthno_instance)
    print(monthno_instance.shape)
    print("\n\n")

    plt.figure(figsize=(11,4))
    plt.subplot(121)
    plt.title("A time series (generated)", fontsize=14)
    plt.plot(monthno, sales[monthno], label=r"$unit sales$")
    plt.plot(monthno_instance[:-1], sales[monthno_instance[:-1]], "b-", linewidth=3, label="A training instance")
    plt.legend(loc="lower left", fontsize=14)
 #   plt.axis([0, 30, -17, 13])
    plt.xlabel("Month")
    plt.ylabel("UnitSales")

    plt.subplot(122)
    plt.title("A training instance", fontsize=14)
    plt.plot(monthno_instance[:-1], sales[monthno_instance[:-1]], "bo", markersize=10, label="instance")
    plt.plot(monthno_instance[1:], sales[monthno_instance[1:]], "w*", markersize=10, label="target")
    plt.legend(loc="upper left")
    plt.xlabel("Time")


    #save_fig("time_series_plot")
    plt.show()




#################################    

    X_batch, y_batch = next_batch(1, n_steps)



    #print(np.c_[X_batch[0], y_batch[0]])




    reset_graph()

    #n_steps = 40
    n_inputs = 1
    n_neurons = 100
    n_layers=1
    n_outputs = 1
    learning_rate = 0.001


    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

    # one layer
    #cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
    #cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons, activation=tf.nn.elu)
    cell = tf.nn.rnn_cell.GRUCell(num_units=n_neurons)

    rnn_outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
    stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
    outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])


    #multi layers
    #layers=[tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons, activation=tf.nn.elu) for layer in range(n_layers)]
    ##layers=[tf.nn.rnn_cell.GRUCell(num_units=n_neurons) for layer in range(n_layers)]

    #multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell(layers)

    #outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)



    loss = tf.reduce_mean(tf.square(outputs - y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True)
    training_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


    n_iterations = 1500
    batch_size = 15

    with tf.Session() as sess:
        init.run()
        for iteration in range(n_iterations):
            X_batch, y_batch = next_batch(batch_size, n_steps)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            if iteration % 100 == 0:
                mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
                print(iteration, "\tMSE:", mse)
        
        X_new = time_series(np.array(t_instance[:-1].reshape(-1, n_steps, n_inputs)))
        y_pred = sess.run(outputs, feed_dict={X: X_new})
        
        saver.save(sess, "./my_time_series_model")
        
    print(y_pred)

    plt.title("Testing the model", fontsize=14)
    plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
    plt.plot(t_instance[1:], time_series(t_instance[1:]), "w*", markersize=10, label="target")
    plt.plot(t_instance[1:], y_pred[0,:,0], "r.", markersize=10, label="prediction")
    plt.legend(loc="upper left")
    plt.xlabel("Time")

    plt.show()



    print("\n\nSales Prediction results written to spreadsheet.\n\n")
#    f.write("\n\nSales Prediction results written to spreadsheet.\n\n")
    
   
##############################################################################


 #   f.close()
    return

    

if __name__ == '__main__':
    main()


