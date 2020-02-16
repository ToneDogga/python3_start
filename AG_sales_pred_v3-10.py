# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 20:30:03 2020

@author: Anthony Paech 2016
"""

# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import pandas as pd
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
    

  
def next_batch(sales,product_row_no,batch_size,n_steps,n_inputs):
    return sales[product_row_no,:-1].reshape(-1,n_steps,n_inputs),sales[product_row_no,1:].reshape(-1,n_steps,n_inputs)   
   

    
def load_data(filename):   #,mask_text):   #,batch_size,n_steps,n_inputs):   
    df=pd.read_excel(filename,-1)  # -1 means all rows


    #  create a pivot table of code, product, day delta and predicted qty and export back to excel

    df["week"]=df.date.dt.to_period('W')
   

    mask=((df["product"]=="SJ300") | (df["product"]=="AJ300"))

  #  mask=(df["product"]=="SJ300")
    table = pd.pivot_table(df[mask], values='qty', index=['product'],columns=['week'], aggfunc=np.sum, margins=False, fill_value=0)   #, observed=True)
    print("\ntable=\n",table.head(5))#  #   f.write("\n\n"+table.to_string())
    print("table created.")
    
    product_names=list(table.index)   #"t"  #list(table[table["product"]])
    #print("product names=",product_names)
    sales=table.to_numpy()
    
  #  print("sales=\n",sales,sales.shape)        
    return sales,product_names



def main():  
    print("\n\nSales prediction tool using a neural network - By Anthony Paech 16/2/20\n")
       
   #    

  #  n_rows=1
    batch_size=1   #4    #the number of mini batches at each time step
    
  #  n_steps = 10     # number of time steps of a week each.
  #  n_inputs = 1     #  number of different product unit sales for that time period
    n_neurons = 105
  # batch_size= 5    #the number of mini batches at each time step
  #  n_outputs=1
    learning_rate=0.001
    n_iterations=1000
    product_row_no=0
    n_predict_steps=80
    t_start=3
    t_finish=102
     
    sales,product_names=load_data("NAT-raw310120all.xlsx")  #,batch_size,n_steps,n_inputs)
    print("All sales=\n",sales,sales.shape)  
    print("\nProduct names",product_names)
  #  sales=sales[:,-(total_size+1):]
 #   n_start_inputs=sales.shape[1]
   # total_size=n_inputs*n_steps
    n_steps=sales.shape[1]-1
    n_rows=sales.shape[0]
    
    n_inputs=2  #105
    n_outputs=1

    print("\nTotal number of time steps=",n_steps,",Total number of products",n_rows)
 
    x_axis=np.arange(0,n_steps+1)

    
    plt.figure(figsize=(11,4))
    plt.subplot(121)
    plt.title("A unit sales series", fontsize=14)
    plt.plot(x_axis, sales[product_row_no,x_axis], label=r"$unit sales$")
 #   plt.plot(x_axis[:-1], sales[product_row_no,x_axis[:-1]], "b-", linewidth=3, label="A training instance")
    plt.legend(loc="best", fontsize=14)
 #   plt.axis([0, 30, -17, 13])
    plt.xlabel("Week")
    plt.ylabel("UnitSales")

    plt.subplot(122)
    plt.title("A training instance", fontsize=14)
    plt.plot(x_axis[:-1], sales[product_row_no,:-1], "bo", markersize=10, label="instance")
    plt.plot(x_axis[1:], sales[product_row_no,1:], "w*", markersize=10, label="target")
    plt.legend(loc="best")
    plt.xlabel("Week")


    #save_fig("time_series_plot")
    plt.show()

    
    
    
    reset_graph()
    
    
    # batch_size should be n_steps
    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

    # one layer
    ##cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
    ##cell = tf.nn.rnn_cell.GRUCell(num_units=n_neurons)
    
    # cell = tf.contrib.rnn.OutputProjectionWrapper(
    #      tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu),
    #      output_size=n_outputs)

#    outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    
    
    
    
    rnn_outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

    stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
    stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
 #  outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])
    outputs = tf.reshape(stacked_outputs, [-1,n_steps, n_outputs])
  #  full_outputs = tf.reshape(stacked_outputs, [-1,n_steps, n_outputs])

 
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


    step=0
    with tf.Session() as sess:
        init.run()
        step=0
        for iteration in range(n_iterations):
            X_batch, y_batch = next_batch(sales,product_row_no,batch_size,n_steps,n_inputs)
      #      print("X_batch=\n",X_batch,"\ny_batch=\n",y_batch)
      #      print("X_batch=\n",X_batch.shape,"\ny_batch=\n",y_batch.shape)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            if iteration % 100 == 0:
                mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
                print(iteration, "\tMSE:", mse)
           #    print("X_batch=\n",X_batch,"\ny_batch=\n",y_batch)


        X_new=sales[product_row_no,1:].reshape(-1,n_steps, n_inputs)
    
#        X_new=sales[product_row_no,1:].reshape(-1, n_steps, n_inputs)
      # X_new = monthno_instance.reshape(-1, n_steps, n_inputs)
 
   #     print("X_new=",X_new,X_new.shape)

        y_pred = sess.run(outputs, feed_dict={X: X_new})
        
        saver.save(sess, "./my_time_series_model")
        
  #  print("y_pred[0,:,0]=", y_pred[0,:,0])

    plt.title("Testing the model", fontsize=14)
    
#    plt.plot(x_axis[:-1], sales[0,x_axis[:-1]], "bo", markersize=10, label="instance")
#    plt.plot(x_axis[1:], sales[0,x_axis[1:]], "w*", markersize=10, label="target")
    plt.plot(x_axis[2:], sales[product_row_no,2:], "bo", markersize=10, label="instance")
    plt.plot(x_axis[3:], sales[product_row_no,-n_steps+2:], "w*", markersize=10, label="target")

  #  print("3 y_pred=",y_pred)   #[:,-n_steps:,:]
   
    plt.plot(x_axis[3:]+1, y_pred[product_row_no,2:,0], "r.", markersize=10, label="prediction")


    plt.legend(loc="best")
    plt.xlabel("Week")

    plt.show()

###############################################################
    
#    x_axis=np.arange(0,n_steps+n_predict_steps)
   
    #  Creative prediction
##############################
    with tf.Session() as sess:                        # not shown in the book
        saver.restore(sess, "./my_time_series_model") # not shown
    
     #   last_sales_val=sales[n_steps-1].astype(float)
      #  sequence = [last_sales_val] * n_steps

        #sequence = [10000.] * n_steps
        #sequence = sales[-n_steps:].tolist()
        sequence=sales[product_row_no,x_axis].reshape(n_outputs,-1)
     #   print("start sequence=",sequence,sequence.shape)

        #nexts=sequence[-1]+1
        #print("sq=",sequence,"nexts",nexts,"n_steps=",n_steps)

        for iteration in range(0,n_predict_steps):
            X_batch=sequence[product_row_no,-n_steps:].reshape(-1, n_steps, n_inputs)
           # X_batch = np.array(sequence[0,-training_length:]).reshape(1,training_length,n_inputs)
            y_pred = sess.run(outputs, feed_dict={X: X_batch})
      #      print("x_batch=\n",X_batch,"y_pred=\n",y_pred)
     #       print("i=",iteration,"y_pred=",y_pred[0,-1,0])
            sequence=np.concatenate((sequence,y_pred[product_row_no, -1, 0].reshape(n_outputs,-1)),axis=1)
      #      print("sequence.shape=",sequence,sequence.shape)

          #  monthno=np.append(monthno,monthno.shape[0]+1)
           # print("monthno=",monthno)
            
    #sequence=sequence.ravel() 
#    print("sequence=\n",sequence)
    #lensequence=len(sequence)
 #   print("sequence.shape=",sequence.shape)
  #  print("y_pred=\n",y_pred[0,:,0])    


######
    pred_axis=np.arange(n_steps+1,n_steps+n_predict_steps+1)        
    plt.figure(figsize=(8,4))
 #  plt.plot(x_axis[:n_steps+1], sequence[product_row_no,:n_steps+1], "b-", linewidth=1)
    plt.plot(x_axis, sales[product_row_no,:], linewidth=2,label=r"$unit sales$")
    plt.plot(pred_axis, sequence[product_row_no,-n_predict_steps:], "r-",linewidth=3,label=r"$unit sales$")

  #  plt.plot(monthno_instance, y_pred[0,:,0], "r.", markersize=10, label="prediction")

   # plt.plot(monthno[-lensequence:],sequence,"b-", linewidth=3)
    plt.xlabel("Week")
    plt.ylabel("Qty")
    plt.title("Unit sales: "+str(product_names[product_row_no]),fontsize=14)
    plt.show()
        
    
    
 
        
    return

    

if __name__ == '__main__':
    main()

        
        
    
