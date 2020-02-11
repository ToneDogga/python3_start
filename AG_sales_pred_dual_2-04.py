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
    
    
    print("batch_size=",batch_size)
    print("n_steps=",n_steps)
    print("n_inputs=",n_inputs)
    print("sales=",sales.shape)
      
    batch_series = np.zeros(shape=(batch_size, n_steps,n_inputs),dtype=int)
  #  print("batch series=\n",batch_series,batch_series.shape)
    indexing_series=np.zeros(dtype=int,shape=(batch_size*n_steps))
    print("indexing series.shape",indexing_series.shape)
    for i in range(0,n_inputs):
       batch_series[0,:,i] = np.random.randint(0,n_steps-1-batch_size,size=(n_steps))
       #print("batch series 1=\n",batch_series,batch_series.shape)
       for j in range(0,n_steps):
            for k in range(1,batch_size):
                #  print("b=",batch_series[i,0,0])  #np.arange(0,n_steps-1)
                batch_series[k,j,i] = batch_series[0,j,i] + k  #np.arange(0,n_steps-1)
                
       new_series=batch_series[:,:,i].reshape(batch_size*n_steps)
       print("new_series shape",new_series.shape)
       indexing_series=np.vstack((indexing_series,new_series))
       print("indexing series=\n",i,indexing_series,indexing_series.shape)
 #      print(i,sales[i,indexing_series])

    indexing_series=np.delete(indexing_series,0,axis=0)  # first row is all zeros
    print("final indexing series=\n",indexing_series,indexing_series.shape)
  

    for z in range(0,n_inputs):  
        print("sales",z,sales[z,indexing_series[z,:]])     
    #print("ys=\n",ys,"ys.shape=",ys.shape)
 
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
    print("ys[:,:, :-1].shape, ys[;,:, 1:].shape",ys[:,:, :-1].shape, ys[:,:, 1:].shape)

    return ys[:, :, :-1].reshape(-1, n_steps, 1), ys[:, :, 1:].reshape(-1, n_steps, 1)
    #return ys[:, :-1], ys[:, 1:]


    
    
 
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
   



#reset_graph()


reset_graph()

#t_min, t_max = 0, 104
resolution = 1

batch_size=52
n_steps=104
n_inputs=2
n_neurons = 100
n_outputs=1
iterations=4

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

mask=((df["product"]=="SJ300") | (df["product"]=="AJ300"))
#mask=((df["product"]=="SJ300"))

table = pd.pivot_table(df[mask], values='qty', index=['product'],columns=['week'], aggfunc=np.sum, margins=False, fill_value=0)   #, observed=True)
print("\ntable=\n",table.head(5))
#  #   f.write("\n\n"+table.to_string())
#print("table created.")



# X_batch = np.array([
#         [[0, 1, 2], [9, 8, 7]], # instance 1
#         [[3, 4, 5], [0, 0, 0]], # instance 2
#         [[6, 7, 8], [6, 5, 4]], # instance 3
#         [[9, 0, 1], [3, 2, 1]], # instance 4
#     ])

# print("X_batch shape=",X_batch.shape)

# shape [4,2,3]   instances, time steps, mini-batches-length =>  [batch_size,n_steps,n_inputs]

# sequence_lwngth_batch= np.array[2,1,2,1]

# with the sales array the shape is 1,105
# there is 1 row, and 105 time series unit sales numbers
# this is the same as 1 instance, 105 time steps, 
# shape [1,105,30]   instances, time steps, mini-batches-length =>  [batch_size,n_steps,n_inputs]
#
#  the instances are 

#print("x_batch=",X_batch.shape)
#table=pd.DataFrame([1,2],dtype=int)
sales=table.to_numpy()
sales=sales[-n_steps:]
print(" sales batch=",sales.shape)

# X_batch,y_batch=next_batch(sales,batch_size,n_steps,n_inputs)  # returns a batch_size number (same as instances) of random batches of sales data un the shape [batch_size, n_steps, n_neurons]

# print("sales next batch x_batch=",X_batch.shape)
# print("sales next batch y_batch=",y_batch.shape)


# sales is in the shape product_code, unit sales size  [2, 105]
# this is actually [n_steps,batch_size]



reset_graph()

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])

basic_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)


init = tf.global_variables_initializer()

for i in range(iterations):
    X_batch,y_batch=next_batch(sales,batch_size,n_steps,n_inputs)  # returns a batch_size number (same as instances) of random batches of sales data un the shape [batch_size, n_steps, n_neurons]

    print("sales next batch x_batch=\n",X_batch,X_batch.shape)
    print("sales next batch y_batch=\n",y_batch,y_batch.shape)
 
    
    with tf.Session() as sess:
        init.run()
        outputs_val = outputs.eval(feed_dict={X: X_batch})
        
    print("Iterations=",i)    
    print("outputs_val=\n",outputs_val)
    print("y_batch=\n",y_batch)
    print("\n")

    
    
    

