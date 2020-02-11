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
   



def next_batch(sales,batch_size, n_steps,n_inputs):
# with the sales array the shape is 1,105
# there is 1 row, and 105 time series unit sales numbers
# this is the same as 1 instance, 105 time steps, 
# shape [30,105,1]   instances, time steps, mini-batches-length =>  [batch_size,n_steps,n_inputs]
#

# X_batch = np.array([
#         [[0, 1, 2], [9, 8, 7]], # instance 1
#         [[3, 4, 5], [0, 0, 0]], # instance 2
#         [[6, 7, 8], [6, 5, 4]], # instance 3
#         [[9, 0, 1], [3, 2, 1]], # instance 4
#     ])

# print("X_batch shape=",X_batch.shape)

# shape [4,2,3]   instances, time steps, mini-batches-length =>  [batch_size,n_steps,n_inputs]

# sequence_lwngth_batch= np.array[2,1,2,1]
    
    
 #   print("batch_size=",batch_size)    # 5
  #  print("n_steps=",n_steps)     # 10
  #  print("n_inputs=",n_inputs)   #2
 #   print("sales=",sales.shape)
      
    batch_series = np.zeros(shape=(batch_size, n_steps,n_inputs),dtype=int)
  #  print("batch series=\n",batch_series,batch_series.shape)
    indexing_series=np.zeros(dtype=int,shape=(batch_size*n_steps))
   # time_index=np.zeros(dtype=int,shape=(n_steps,batch_size))
  #  print("indexing series.shape",indexing_series.shape)
    for i in range(0,n_inputs):
       batch_series[0,:,i] = np.random.randint(0,n_steps+1-batch_size,size=(n_steps))
       #print("batch series 1=\n",batch_series,batch_series.shape)
       for j in range(0,n_steps):
            for k in range(0,batch_size):
                #  print("b=",batch_series[i,0,0])  #np.arange(0,n_steps-1)
                batch_series[k,j,i] = batch_series[0,j,i] + k  #np.arange(0,n_steps-1)
                
       new_series=batch_series[:,:,i].reshape(batch_size*n_steps)
  #     print("new series=\n",new_series,"new_series shape",new_series.shape)
       indexing_series=np.vstack((indexing_series,new_series))
    #   print("indexing series=\n",i,indexing_series,indexing_series.shape)
 #      print(i,sales[i,indexing_series])

    indexing_series=np.delete(indexing_series,0,axis=0)  # first row is all zeros
    indexing_series=indexing_series.reshape(n_inputs,batch_size,n_steps)  

    s=sales[:,indexing_series]    #.reshape(batch_size,n_inputs,-1)  
  #  print("1final sales=\n",s,s.shape)
    s=s.reshape(n_inputs,batch_size,n_steps)  
  #  print("2final sales=\n",s,s.shape)

    if s.shape[0]>1:
        s=np.delete(s,0,axis=0)  # first row is all zeros
  #  print("3final sales=\n",s,s.shape)

    s=s.reshape(n_inputs,batch_size,n_steps)  

  #  print("4final sales=\n",s,s.shape)

   # r=sales[:,time_series].reshape(batch_size,n_inputs,-1)  
   # print("final time series sales=\n",r,r.shape)
  #  print("br return X=\n",s[:, :-1])
  #  print("br return y=\n",s[:, 1:])
  #  print("return X.T=\n",s[:, :-1].T)  #reshape(-1, n_steps, n_inputs))
  #  print("return y.T=\n",s[:, 1:].T)   #reshape(-1, n_steps, n_inputs))


  #  print("return X=\n",s[:, :-1].reshape(-1, n_steps, n_inputs))
  #  print("return y=\n",s[:, 1:].reshape(-1, n_steps, n_inputs))

    return s[:, :-1].T, s[:, 1:].T






def main():


#    f=open(cfg.outfile_predict,"w")

    print("\n\nSales prediction tool using a neural network - By Anthony Paech 2/2/20\n")
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
    
    
    #mask=((df["product"]=="SJ300") | (df["product"]=="AJ300"))

    mask=(df["product"]=="SJ300")
    table = pd.pivot_table(df[mask], values='qty', index=['product'],columns=['week'], aggfunc=np.sum, margins=False, fill_value=0)   #, observed=True)
    print("\ntable=\n",table.head(5))
#  #   f.write("\n\n"+table.to_string())
    print("table created.")

    sales=table.to_numpy()



################################
    n_inputs = 1
    n_neurons = 315     # best is n_steps+1
   # n_layers=1
    n_outputs = 1
    learning_rate = 0.001   #0.001
    n_iterations =4000
    batch_size = 52
  #  m_start=10
    n_steps=104
    n_predict_steps=80

###############################
  #  sales=np.arange(0,n_steps+1)   #,size=(n_inputs,n_steps))

    sales=sales[-n_steps-1:].reshape(n_inputs,-1)
    lensales=sales.shape[1]
    
    print("sales.shape=",sales.shape)    
    
    x_axis=np.arange(0,lensales).ravel()
    
#    print("x_axis",x_axis)

    plt.figure(figsize=(11,4))
    plt.subplot(121)
    plt.title("A unit sales series", fontsize=14)
    plt.plot(x_axis, sales[0,x_axis], label=r"$unit sales$")
    plt.plot(x_axis[:-1], sales[0,x_axis[:-1]], "b-", linewidth=3, label="A training instance")
    plt.legend(loc="best", fontsize=14)
 #   plt.axis([0, 30, -17, 13])
    plt.xlabel("Week")
    plt.ylabel("UnitSales")

    plt.subplot(122)
    plt.title("A training instance", fontsize=14)
    plt.plot(x_axis[:-1], sales[0,x_axis[:-1]], "bo", markersize=10, label="instance")
    plt.plot(x_axis[1:], sales[0,x_axis[1:]], "w*", markersize=10, label="target")
    plt.legend(loc="best")
    plt.xlabel("Week")


    #save_fig("time_series_plot")
    plt.show()




#################################    

   # X_batch, y_batch = next_batch(sales,1, n_steps)



    #print(np.c_[X_batch[0], y_batch[0]])




    reset_graph()



#    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
#    y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

    X = tf.placeholder(tf.float32, [n_steps, None, n_inputs])
    y = tf.placeholder(tf.float32, [n_steps, None, n_outputs])

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
    outputs = tf.reshape(stacked_outputs, [n_steps,-1, n_outputs])

 
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



    with tf.Session() as sess:
        init.run()
        for iteration in range(n_iterations):
            X_batch, y_batch = next_batch(sales,batch_size, n_steps,n_inputs)
   #         print("X_batch=\n",X_batch.shape,"\ny_batch=\n",y_batch.shape)
  #         print("X_batch=\n",X_batch.shape,"\ny_batch=\n",y_batch.shape)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            if iteration % 100 == 0:
                mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
                print(iteration, "\tMSE:", mse)
           #    print("X_batch=\n",X_batch,"\ny_batch=\n",y_batch)
    
 #       X_new = time_series(np.array(t_instance[:-1].reshape(-1, n_steps, n_inputs)))
      #  X_index=monthno_instance.reshape(-1, n_steps, n_inputs)
      #  print("X_index=",X_index)
        X_new = sales[:,x_axis[:-1]].reshape(n_steps,-1, n_inputs)
       # X_new = monthno_instance.reshape(-1, n_steps, n_inputs)
 
  #      print("X_new=",X_new)

        y_pred = sess.run(outputs, feed_dict={X: X_new})
        
        saver.save(sess, "./my_time_series_model")
        
  #  print("y_pred=",y_pred)

    plt.title("Testing the model", fontsize=14)
    plt.plot(x_axis[:-1], sales[0,x_axis[:-1]], "bo", markersize=10, label="instance")
    plt.plot(x_axis[1:], sales[0,x_axis[1:]], "w*", markersize=10, label="target")
  #  plt.plot(weekno_instance, sales[weekno_instance], "w*", markersize=10, label="target")

    plt.plot(x_axis[1:], y_pred[:,0,0], "r.", markersize=10, label="prediction")
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
        #sequence = sales[-n_steps:].tolist()
        sequence=sales[0,x_axis].reshape(n_outputs,-1)
        print("ss=",sequence.shape)

        #nexts=sequence[-1]+1
        #print("sq=",sequence,"nexts",nexts,"n_steps=",n_steps)

        for iteration in range(n_predict_steps):
            X_batch = np.array(sequence[0,-n_steps:]).reshape(n_steps,1,n_inputs)
            y_pred = sess.run(outputs, feed_dict={X: X_batch})
          #  print("x_batch=\n",X_batch,"y_pred=\n",y_pred)
         #   print("i=",iteration,"y_pred=",y_pred)
            sequence=np.concatenate((sequence,y_pred[-1, 0, n_outputs-1].reshape(n_outputs,-1)),axis=1)
         #   print("sequence.shape=",sequence,sequence.shape)

          #  monthno=np.append(monthno,monthno.shape[0]+1)
           # print("monthno=",monthno)
            
    #sequence=sequence.ravel() 
 #   print("sequence=\n",sequence)
    #lensequence=len(sequence)
  #  print("sequence.shape=",sequence.shape)
  #  print("y_pred=\n",y_pred[0,:,0])    


######

    plt.figure(figsize=(8,4))
    plt.plot(np.arange(0,sequence[0].shape[0]), sequence[0], "b-", linewidth=1)
    plt.plot(x_axis[:lensales], sales[0,x_axis[:lensales]], linewidth=3,label=r"$unit sales$")
    plt.plot(x_axis[:n_steps], sales[0,x_axis[:n_steps]], linewidth=2, label=r"$unit sales$")

  #  plt.plot(monthno_instance, y_pred[0,:,0], "r.", markersize=10, label="prediction")

   # plt.plot(monthno[-lensequence:],sequence,"b-", linewidth=3)
    plt.xlabel("Week")
    plt.ylabel("Qty")
    plt.show()
        
    

################################################################

 #   f.close()
    return

    

if __name__ == '__main__':
    main()

   
