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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, StandardScaler


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






# def save_fig(fig_id, tight_layout=True):
#     path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
#     print("Saving figure", fig_id)
#     if tight_layout:
#         plt.tight_layout()
#     plt.savefig(path, format='png', dpi=300)
    
    
    
# def shuffle_batch(X, y, batch_size):
#     rnd_idx = np.random.permutation(len(X))
#     n_batches = len(X) // batch_size
#     for batch_idx in np.array_split(rnd_idx, n_batches):
#         X_batch, y_batch = X[batch_idx], y[batch_idx]
#         yield X_batch, y_batch
        
        
        
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
#
#t_min, t_max = 0, 30
#resolution = 0.1



#def time_series(t):
#    return t * np.sin(t) / 3 + 2 * np.sin(t*5)

# def next_batch(sales,batch_size, n_steps):
#     t0 = np.random.rand(batch_size, 1) * (t_max - t_min - n_steps * resolution)
#     print("bs=",batch_size,"t0=",t0,t0.shape)
#     Ts = t0 + np.arange(0., n_steps + 1) * resolution
#     print("Ts=",Ts,Ts.shape)
#     ys = time_series(Ts)
#     print("ys.shape=",ys.shape)
#     return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1)



def next_batch(sales,batch_size, n_steps):
    sales_size=sales.shape[0]
  #  print("sales shape[0]",sales_size,"bs=",batch_size)
    start_array=np.random.randint(sales_size-batch_size-2,size=(1,1)) #* (sales.shape[0]-n_steps)
#    print("start array",start_array)
#    print("start_array.shape=",start_array.shape)

    
    sales_series = start_array + np.arange(0, n_steps+1,dtype=int)
  #  print("sales series",sales_series)
#    print("sales_series.shape=",sales_series.shape)

    ys = sales[sales_series]

#    print("ys=",ys[:,:-1],ys.shape)
    return ys[:,:-1].reshape(-1,n_steps,1), ys[:,1:].reshape(-1,n_steps,1)







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

    # if df.empty:
    #     #print(sys.argv[1],"Not found.")
    #     print(cfg.infilename,"Not found.")

    #     sys.exit()

 #   print(df)
    
#############################################################################3
    #  create a pivot table of code, product, day delta and predicted qty and export back to excel

 #   df["month"]=df.date.dt.to_period('M')
    df["week"]=df.date.dt.to_period('W')


# #    print(df)

    mask=(df["product"]=="SJ300")
    table = pd.pivot_table(df[mask], values='qty', index=['product'],columns=['week'], aggfunc=np.sum, margins=False, fill_value=0)   #, observed=True)
    print("\ntable=\n",table.head(5))
#  #   f.write("\n\n"+table.to_string())
    print("table created.")

    tsales=table.to_numpy()

#  #   scaler=StandardScaler()
#  #   scaled_sales=scaler.fit_transform(tsales)
# #    print(scaled_sales)
# #    print(scaled_sales.shape)
# #    sales=scaled_sales.ravel()


    sales=tsales.ravel()
 #   sales=np.array([10000,11000,12000,11650,10987,11450,11002,9000,8790,8867,7356,7654,8007,7654,6543,9567,9677,6789,7890,8000,8500,9500,10500,11800,11000,10600,10000,11000,12000,11650,11450,11002,9000,8000,8500,9500,10500,11800,11000,10600,9800,9700,9900,8700,8500,8300,8790,9800,9700,9900,8700,8500,8300,8790,7678,9878,]).ravel()
  #  print(sales)
  #  print(sales.shape)
   # print("sales[:1]",sales[:1])
   # print("sales[1:]",sales[1:])
   # print("sales[-1:]",sales[-1:])
   # print("sales[:-1]",sales[:-1])
   # print("sales[-1]",sales[-1])
    lensales=sales.shape[0]
    # print("\nWriting to Excel: tabletest.xlsx")
 
    # with pd.ExcelWriter("tabletest.xlsx") as writer:  # mode="a" for append
    #     table.to_excel(writer,sheet_name="Weekly unit sales")
    #     print("table of weekly unit sales written to excel.")

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

    weekno=np.arange(0,sales.shape[0]).ravel()
 #   print("monthno")
 #   print(monthno)
 #   print(monthno.shape)

    m_start=3
    n_steps = 90
    weekno_instance = np.arange(m_start, m_start+n_steps+1).ravel()

  #  print("monthno_instance")
  #  print(monthno_instance)
  #  print(monthno_instance.shape)
  #  print("\n\n")

    plt.figure(figsize=(11,4))
    plt.subplot(121)
    plt.title("A unit sales series", fontsize=14)
    plt.plot(weekno, sales[weekno], label=r"$unit sales$")
    plt.plot(weekno_instance[:-1], sales[weekno_instance[:-1]], "b-", linewidth=3, label="A training instance")
    plt.legend(loc="best", fontsize=14)
 #   plt.axis([0, 30, -17, 13])
    plt.xlabel("Week")
    plt.ylabel("UnitSales")

    plt.subplot(122)
    plt.title("A training instance", fontsize=14)
    plt.plot(weekno_instance[:-1], sales[weekno_instance[:-1]], "bo", markersize=10, label="instance")
    plt.plot(weekno_instance[1:], sales[weekno_instance[1:]], "w*", markersize=10, label="target")
    plt.legend(loc="best")
    plt.xlabel("Week")


    #save_fig("time_series_plot")
    plt.show()




#################################    

   # X_batch, y_batch = next_batch(sales,1, n_steps)



    #print(np.c_[X_batch[0], y_batch[0]])




    reset_graph()

    #n_steps = 40
    n_inputs = 1
    n_neurons = 1000
    n_layers=1
    n_outputs = 1
    learning_rate = 0.001   #0.001


    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

    # one layer
    ##cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
    ##cell = tf.nn.rnn_cell.GRUCell(num_units=n_neurons)

    rnn_outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
    stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
    outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])


    #multi layers
    #layers=[tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu) for layer in range(n_layers)]
    #layers=[tf.nn.rnn_cell.GRUCell(num_units=n_neurons) for layer in range(n_layers)]

    #multi_layer_cell = tf.nn.rnn_cell.MultiRNNCell(layers)

    #outputs, states = tf.nn.dynamic_rnn(multi_layer_cell, X, dtype=tf.float32)



    loss = tf.reduce_mean(tf.square(outputs - y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True)
    training_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


    n_iterations = 2000
    batch_size = 90

    with tf.Session() as sess:
        init.run()
        for iteration in range(n_iterations):
            X_batch, y_batch = next_batch(sales,batch_size, n_steps)
   #         print("X_batch=\n",X_batch.shape,"\ny_batch=\n",y_batch.shape)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            if iteration % 100 == 0:
                mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
                print(iteration, "\tMSE:", mse)
        
 #       X_new = time_series(np.array(t_instance[:-1].reshape(-1, n_steps, n_inputs)))
      #  X_index=monthno_instance.reshape(-1, n_steps, n_inputs)
      #  print("X_index=",X_index)
        X_new = sales[weekno_instance[:-1]].reshape(-1, n_steps, n_inputs)
       # X_new = monthno_instance.reshape(-1, n_steps, n_inputs)
 
  #      print("X_new=",X_new)

        y_pred = sess.run(outputs, feed_dict={X: X_new})
        
        saver.save(sess, "./my_time_series_model")
        
  #  print("y_pred=",y_pred)

    plt.title("Testing the model", fontsize=14)
    plt.plot(weekno_instance[:-1], sales[weekno_instance[:-1]], "bo", markersize=10, label="instance")
    plt.plot(weekno_instance[1:], sales[weekno_instance[1:]], "w*", markersize=10, label="target")
  #  plt.plot(weekno_instance, sales[weekno_instance], "w*", markersize=10, label="target")

    plt.plot(weekno_instance[1:], y_pred[0,:,0], "r.", markersize=10, label="prediction")
    plt.legend(loc="best")
    plt.xlabel("Week")

    plt.show()

#  Creative prediction
##############################
    with tf.Session() as sess:                        # not shown in the book
        saver.restore(sess, "./my_time_series_model") # not shown
    
     #   last_sales_val=sales[n_steps-1].astype(float)
      #  sequence = [last_sales_val] * n_steps

        #sequence = [10000.] * n_steps
        sequence = sales[-n_steps:].tolist()
        #nexts=sequence[-1]+1
        #print("sq=",sequence,"nexts",nexts,"n_steps=",n_steps)

        for iteration in range(20):
            X_batch = np.array(sequence[-n_steps:]).reshape(1, n_steps, 1)
            y_pred = sess.run(outputs, feed_dict={X: X_batch})
          #  print("x_batch=\n",X_batch,"y_pred=\n",y_pred)
         #   print("i=",iteration,"y_pred=",y_pred)
            sequence.append(y_pred[0, -1, 0])
          #  monthno=np.append(monthno,monthno.shape[0]+1)
           # print("monthno=",monthno)
            
     
  #  print("sequence=\n",sequence)   
  #  lensequence=len(sequence)
  #  print("lensales=",lensales,"lensequence=",lensequence)
  #  print("y_pred=\n",y_pred[0,:,0])    
    plt.figure(figsize=(8,4))
    plt.plot(np.arange(len(sequence))+lensales, sequence, "b-", linewidth=1)
    plt.plot(weekno[:lensales], sales[weekno[:lensales]], linewidth=3,label=r"$unit sales$")
    plt.plot(weekno[:n_steps], sales[weekno[:n_steps]], linewidth=2, label=r"$unit sales$")

  #  plt.plot(monthno_instance, y_pred[0,:,0], "r.", markersize=10, label="prediction")

   # plt.plot(monthno[-lensequence:],sequence,"b-", linewidth=3)
    plt.xlabel("Week")
    plt.ylabel("Qty")
    plt.show()
        
    
 #   print("\n\nSales Prediction results written to spreadsheet.\n\n")
#    f.write("\n\nSales Prediction results written to spreadsheet.\n\n")
    
   
##############################################################################
    # with tf.Session() as sess:
    #     saver.restore(sess, "./my_time_series_model")
    
    #     sequence1 = [0. for i in range(n_steps)]
    #     for iteration in range(len(t) - n_steps):
    #         X_batch = np.array(sequence1[-n_steps:]).reshape(1, n_steps, 1)
    #         y_pred = sess.run(outputs, feed_dict={X: X_batch})
    #         sequence1.append(y_pred[0, -1, 0])
    
    #     sequence2 = [time_series(i * resolution + t_min + (t_max-t_min/3)) for i in range(n_steps)]
    #     for iteration in range(len(t) - n_steps):
    #         X_batch = np.array(sequence2[-n_steps:]).reshape(1, n_steps, 1)
    #         y_pred = sess.run(outputs, feed_dict={X: X_batch})
    #         sequence2.append(y_pred[0, -1, 0])
    
    # plt.figure(figsize=(11,4))
    # plt.subplot(121)
    # plt.plot(t, sequence1, "b-")
    # plt.plot(t[:n_steps], sequence1[:n_steps], "b-", linewidth=3)
    # plt.xlabel("Time")
    # plt.ylabel("Value")
    
    # plt.subplot(122)
    # plt.plot(t, sequence2, "b-")
    # plt.plot(t[:n_steps], sequence2[:n_steps], "b-", linewidth=3)
    # plt.xlabel("Time")
    # save_fig("creative_sequence_plot")
    # plt.show()
    

################################################################

 #   f.close()
    return

    

if __name__ == '__main__':
    main()


