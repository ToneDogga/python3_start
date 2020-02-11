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

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)
 
    
        
        
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
   
 
# t_min, t_max = 0, 30
# resolution = 0.1

# def time_series(t):
#     return t * np.sin(t) / 3 + 2 * np.sin(t*5)

# def next_batch(batch_size, n_steps):
#     t0 = np.random.rand(batch_size, 1) * (t_max - t_min - n_steps * resolution)
#     Ts = t0 + np.arange(0., n_steps + 1) * resolution
#     ys = time_series(Ts)
#     return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1)

  
    



#def time_series(t):
#    return t * np.sin(t) / 3 + 2 * np.sin(t*5)

# def next_batch(sales,batch_size, n_steps,n_inputs):
# # with the sales array the shape is 1,105
# # there is 1 row, and 105 time series unit sales numbers
# # this is the same as 1 instance, 105 time steps, 
# # shape [30,105,1]   instances, time steps, mini-batches-length =>  [batch_size,n_steps,n_inputs]
# #
    
    
#     print("batch_size=",batch_size)
#     print("n_steps=",n_steps)
#     print("n_inputs=",n_inputs)
#     t0 = np.random.randint(0,n_inputs,size=(batch_size, 1)) * (sales.shape[1] - n_steps)
#     print("t0= shape",t0.shape)
#     Ts = t0 + np.arange(0, n_steps + 1,dtype=int)
#     print("ts=\n",Ts,"ts shape=",Ts.shape)
#     Tb = np.random.randint(0,sales.shape[1] - n_inputs,size=(batch_size,1))
#     print("Tb=\n",Tb,"Tb shape",Tb.shape)
#    # print("Tb[3,0]=",Tb[3,0])
#     tbshape=Tb.shape[0]
#     Tr=Tb.reshape(batch_size,-1)
#     for i in range(tbshape):
#         Tr=np.stack((Tb+np.arange(Tb[i,:],Tb[i,:]+n_inputs,dtype=int)),axis=-1)
#     print("Tr=\n",Tr,"Tr shape",Tr.shape)

#  #   time_series=Ts[Tb:Tb+n_inputs]
#  #   print("time series=\n",time_series,"time_series.shape",time_series.shape)
#     ys = sales[:,Tr]
#     print("ys=\n",ys,"ys.shape=",ys.shape)
#  #   return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1)
#     return ys[:, :-1], ys[:, 1:]



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
                
   #    print("batch_series=\n",batch_series)   
  #     time_series=batch_series.T.reshape(n_steps)
  #     time_index=np.vstack((time_index,time_series))
  #     print("time series=\n",time_series,time_series.shape)
       new_series=batch_series[:,:,i].reshape(batch_size*n_steps)
  #     print("new series=\n",new_series,"new_series shape",new_series.shape)
       indexing_series=np.vstack((indexing_series,new_series))
    #   print("indexing series=\n",i,indexing_series,indexing_series.shape)
 #      print(i,sales[i,indexing_series])

    indexing_series=np.delete(indexing_series,0,axis=0)  # first row is all zeros
  #  time_index=np.delete(time_index,0,axis=0)  # first row is all zeros
 #   time_series=batch_series.T.reshape(n_steps,n_inputs,batch_size)
    #time_index=np.vstack((time_index,time_series))
   # print("time series=\n",time_series,time_series.shape)
    indexing_series=indexing_series.reshape(n_inputs,batch_size,n_steps)  

  #  time_series2=batch_series.reshape(n_steps,n_inputs,batch_size)
    #time_index=np.vstack((time_index,time_series))
  #  print("time series2=\n",time_series2,time_series2.shape)


 #   print("final indexing series=\n",indexing_series,indexing_series.shape)
  #  print("final time index series=\n",time_index,time_index.shape)
  

  #  for z in range(0,n_inputs):  
   #     print("sales",z,sales[z,indexing_series[z,:]])     
    #print("ys=\n",ys,"ys.shape=",ys.shape)
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


  #  return s[:, :-1].reshape(-1, n_steps, n_inputs), s[:, 1:].reshape(-1, n_steps, n_inputs)

  #  print("batch series 2=\n",batch_series,batch_series.shape)
  #  batch_series[:,:,0] = batch_series[:,0,0] + np.arange(0, n_steps-1,dtype=int)
  #  print("batch series 2=\n",batch_series,batch_series.shape)
  #  indexing_series=batch_series[:,:,0].reshape(-1,batch_size*n_steps)
  #  print("indexomg series=\n",indexing_series)
  #  ys = sales[0,indexing_series[0]]
  #  print("ys=\n",ys,"ys.shape=",ys.shape)
    #print("ys reshape=",ys.reshape(-1, batch_size*n_steps, n_inputs))
 
    
#     print("ts=\n",Ts,"ts shape=",Ts.shape)
#     Tb = np.random.randint(0,sales.shape[1] - n_inputs,size=(batch_size,1))
#     print("Tb=\n",Tb,"Tb shape",Tb.shape)
#    # print("Tb[3,0]=",Tb[3,0])
#     tbshape=Tb.shape[0]
#    
    
    # Tr=batch_series
   # for i in range(batch_size):
   #     Tr[i]=np.arange(batch_series[i,0],batch_series[i,0]+batch_size,dtype=int)
    #     print("Tr=\n",Tr,"Tr shape",Tr.shape)
    #     ys = sales[:,Tr]
    #     ys3d=ys
    #     for j in range(n_steps):
    #         ys3d[i,j,:]=np.arange(Tr[i,j],Tr[i,j]+n_steps,dtype=int)
    # print("Tr=\n",Tr,"Tr shape",Tr.shape)

 #   time_series=Ts[Tb:Tb+n_inputs]
 #   print("time series=\n",time_series,"time_series.shape",time_series.shape)
 #   ys = sales[:,Tr]
 #   print("ys=\n",ys,"ys.shape=",ys.shape)
    
   # ys3d=ys[...,:,:)
 #   print("ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1)",ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1))
 #   print("ys[:,:, :-1].shape, ys[;,:, 1:].shape",ys[:,:, :-1].shape, ys[:,:, 1:].shape)

  #  return ys[:, :, :-1].reshape(-1, n_steps, 1), ys[:, :, 1:].reshape(-1, n_steps, 1)
    #return ys[:, :-1], ys[:, 1:]


 

# def next_batch2(n_inputs,sales,batch_size, n_steps):
#     sales_size=sales.shape[0]
#   #  print("sales shape[0]",sales_size,"bs=",batch_size)
#     start_array=np.random.randint(sales_size-batch_size-2,size=(n_inputs,1)) #* (sales.shape[0]-n_steps)
# #    print("start array",start_array)
# #    print("start_array.shape=",start_array.shape)

    
#     sales_series = start_array + np.arange(0, n_steps+1,dtype=int)
#   #  print("sales series",sales_series)
# #    print("sales_series.shape=",sales_series.shape)

#     ys = sales[sales_series]

# #    print("ys=",ys[:,:-1],ys.shape)
#     return ys[:,:-1].reshape(-1,n_steps,n_inputs), ys[:,1:].reshape(-1,n_steps,n_inputs)







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



# df=pd.read_excel("NAT-raw310120all.xlsx",-1)  # -1 means all rows

#     # if df.empty:
#     #     #print(sys.argv[1],"Not found.")
#     #     print(cfg.infilename,"Not found.")

#     #     sys.exit()

#  #   print(df)
    
# #############################################################################3
#     #  create a pivot table of code, product, day delta and predicted qty and export back to excel

#  #   df["month"]=df.date.dt.to_period('M')
# df["week"]=df.date.dt.to_period('W')


# # #    print(df)

# mask=((df["product"]=="SJ300") | (df["product"]=="AJ300"))
# #mask=((df["product"]=="SJ300"))

# table = pd.pivot_table(df[mask], values='qty', index=['product'],columns=['week'], aggfunc=np.sum, margins=False, fill_value=0)   #, observed=True)
# print("\ntable=\n",table.head(5))
# #  #   f.write("\n\n"+table.to_string())
# #print("table created.")






 #   df=read_excel(sys.argv[1],-1)  # -1 means all rows
 #   df=pd.read_excel("NAT-raw310120all.xlsx",-1)  # -1 means all rows

    # if df.empty:
    #     #print(sys.argv[1],"Not found.")
    #     print(cfg.infilename,"Not found.")

    #     sys.exit()

 #   print(df)
    
#############################################################################3
    #  create a pivot table of code, product, day delta and predicted qty and export back to excel

 #   df["month"]=df.date.dt.to_period('M')
#    df["week"]=df.date.dt.to_period('W')


# #    print(df)
    
    
    #mask=((df["product"]=="SJ300") | (df["product"]=="AJ300"))

  #  mask=(df["product"]=="SJ300")
  #  table = pd.pivot_table(df[mask], values='qty', index=['product'],columns=['week'], aggfunc=np.sum, margins=False, fill_value=0)   #, observed=True)
  #  print("\ntable=\n",table.head(5))
#  #   f.write("\n\n"+table.to_string())
  #  print("table created.")

  #  tsales=table.to_numpy()

#  #   scaler=StandardScaler()
#  #   scaled_sales=scaler.fit_transform(tsales)
# #    print(scaled_sales)
# #    print(scaled_sales.shape)
# #    sales=scaled_sales.ravel()


  #  sales=tsales   #.ravel()
    
 #   sales=np.array([10000,11000,12000,11650,10987,11450,11002,9000,8790,8867,7356,7654,8007,7654,6543,9567,9677,6789,7890,8000,8500,9500,10500,11800,11000,10600,10000,11000,12000,11650,11450,11002,9000,8000,8500,9500,10500,11800,11000,10600,9800,9700,9900,8700,8500,8300,8790,9800,9700,9900,8700,8500,8300,8790,7678,9878,]).ravel()
  #  print(sales)
  #  print(sales.shape)
   # print("sales[:1]",sales[:1])
   # print("sales[1:]",sales[1:])
   # print("sales[-1:]",sales[-1:])
   # print("sales[:-1]",sales[:-1])
   # print("sales[-1]",sales[-1])
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
    n_inputs = 1
    n_neurons = 101
   # n_layers=1
    n_outputs = 1
    learning_rate = 0.001   #0.001
    n_iterations =1500
    batch_size = 50
  #  m_start=10
    n_steps=100
    n_predict_steps=20

###############################
    sales=np.arange(0,n_steps+1)   #,size=(n_inputs,n_steps))

    sales=sales[-n_steps-1:].reshape(n_inputs,-1)
    lensales=sales.shape[1]
    
    print("sales.shape=",sales.shape)    
  #  X_batch, y_batch = next_batch(sales,2, n_steps,n_inputs)  # one batch to start

  #  print("first X batch=\n",X_batch,X_batch.shape)
  #  print("first y batch=\n",y_batch,y_batch.shape)

 #   t = np.linspace(t_min, t_max, int((t_max - t_min) / resolution))

 #   weekno=np.arange(0,sales.shape[0]).ravel()
 #   print("monthno")
 #   print(monthno)
 #   print(monthno.shape)

  #  m_start=3
  #  n_steps = 90
  #  weekno_instance = np.arange(m_start, m_start+n_steps+1).ravel()

  #  print("monthno_instance")
  #  print(monthno_instance)
  #  print(monthno_instance.shape)
  #  print("\n\n")
    
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

    #n_steps = 40

# batch_size=5  #52
# n_steps=10  #104
# n_inputs=2
# n_neurons = 100
# n_outputs=1
# iterations=2

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
            print("sequence.shape=",sequence,sequence.shape)

          #  monthno=np.append(monthno,monthno.shape[0]+1)
           # print("monthno=",monthno)
            
    #sequence=sequence.ravel() 
    print("sequence=\n",sequence)
    #lensequence=len(sequence)
    print("sequence.shape=",sequence.shape)
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

   
    
 
# try:
#   import tensorflow.compat.v1 as tf
# except Exception:
#   pass

# # #tf.enable_eager_execution()
# # tf.compat.v1.enable_eager_execution(
# #     config=None,
# #     device_policy=None,
# #     execution_mode=None
# # )

# # tf.enable_v2_behavior()



# #import tensorflow as tf

# #import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# print("tensorflow:",tf.__version__)
   



# #reset_graph()


# reset_graph()

# #t_min, t_max = 0, 104
# resolution = 1

# batch_size=5  #52
# n_steps=10  #104
# n_inputs=2
# n_neurons = 100
# n_outputs=1
# iterations=2

# df=pd.read_excel("NAT-raw310120all.xlsx",-1)  # -1 means all rows

#     # if df.empty:
#     #     #print(sys.argv[1],"Not found.")
#     #     print(cfg.infilename,"Not found.")

#     #     sys.exit()

#  #   print(df)
    
# #############################################################################3
#     #  create a pivot table of code, product, day delta and predicted qty and export back to excel

#  #   df["month"]=df.date.dt.to_period('M')
# df["week"]=df.date.dt.to_period('W')


# # #    print(df)

# mask=((df["product"]=="SJ300") | (df["product"]=="AJ300"))
# #mask=((df["product"]=="SJ300"))

# table = pd.pivot_table(df[mask], values='qty', index=['product'],columns=['week'], aggfunc=np.sum, margins=False, fill_value=0)   #, observed=True)
# print("\ntable=\n",table.head(5))
# #  #   f.write("\n\n"+table.to_string())
# #print("table created.")



# # X_batch = np.array([
# #        time_step0,   time_step1
# #         [[0, 1, 2], [9, 8, 7]], # instance 0
# #         [[3, 4, 5], [0, 0, 0]], # instance 1
# #         [[6, 7, 8], [6, 5, 4]], # instance 2
# #         [[9, 0, 1], [3, 2, 1]], # instance 3
# #     ])

# # print("X_batch shape=",X_batch.shape)

# # shape [4,2,3]   instances, time steps, mini-batches-length =>  [batch_size,n_steps,n_inputs]

# # sequence_lwngth_batch= np.array[2,1,2,1]

# # with the sales array the shape is 1,105
# # there is 1 row, and 105 time series unit sales numbers
# # this is the same as 1 instance, 105 time steps, 
# # shape [1,105,30]   instances, time steps, mini-batches-length =>  [batch_size,n_steps,n_inputs]
# #
# #  the instances are 

# #print("x_batch=",X_batch.shape)
# #table=pd.DataFrame([1,2],dtype=int)
# sales=table.to_numpy()
# sales=sales[-n_steps:]
# print(" sales batch=",sales.shape)

# # X_batch,y_batch=next_batch(sales,batch_size,n_steps,n_inputs)  # returns a batch_size number (same as instances) of random batches of sales data un the shape [batch_size, n_steps, n_neurons]

# # print("sales next batch x_batch=",X_batch.shape)
# # print("sales next batch y_batch=",y_batch.shape)


# # sales is in the shape product_code, unit sales size  [2, 105]
# # this is actually [n_steps,batch_size]



# reset_graph()

# X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])

# basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
# outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)


# init = tf.global_variables_initializer()

# for i in range(iterations):
#     X_batch,y_batch=next_batch(sales,batch_size,n_steps,n_inputs)  # returns a batch_size number (same as instances) of random batches of sales data un the shape [batch_size, n_steps, n_neurons]

#     print("sales next batch x_batch=\n",X_batch.shape)
#     print("sales next batch y_batch=\n",y_batch.shape)
 
    
#     with tf.Session() as sess:
#         init.run()
#         outputs_val = outputs.eval(feed_dict={X: X_batch})
        
#     print("Iterations=",i)    
#   #  print("outputs_val=\n",outputs_val)
#   #  print("y_batch=\n",y_batch)
#     print("\n")

    
    
    

