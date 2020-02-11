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
   



# t_min, t_max = 0, 30
# resolution = 0.1

# def time_series(t):
#     return t * np.sin(t) / 3 + 2 * np.sin(t*5)

# def next_batch(batch_size, n_steps):
#     t0 = np.random.rand(batch_size, 1) * (t_max - t_min - n_steps * resolution)
#     Ts = t0 + np.arange(0., n_steps + 1) * resolution
#     ys = time_series(Ts)
#     return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1)


def next_batch(sales,batch_size, n_steps,n_inputs):
# with the sales array the shape is 1,105
# there is 1 row, and 105 time series unit sales numbers
# this is the same as 1 instance, 105 time steps, 
# shape [52,104,2]   instances, time steps, mini-batches-length =>  [batch_size,n_steps,n_inputs]
#
    
    
 #   print("batch_size=",batch_size)
 #   print("n_steps=",n_steps)
 #   print("n_inputs=",n_inputs)
    batch_series = np.random.randint(0,n_steps,size=(batch_size+1, n_inputs))
    Tr=batch_series
    for i in range(batch_size):
        Tr[i,:]=np.arange(batch_series[i,0],batch_series[i,0]+n_inputs,dtype=int)
    print("Tr=\n",Tr,"Tr shape",Tr.shape)

 #   time_series=Ts[Tb:Tb+n_inputs]
 #   print("time series=\n",time_series,"time_series.shape",time_series.shape)
    ys = sales[Tr]
    print("ys=\n",ys,"ys.shape=",ys.shape)
 #   print("ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1)",ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1))
    print("ys[:, :-1].shape, ys[:, 1:].shape",ys[:, :-1].shape, ys[:, 1:].shape)

    return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1)
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

batch_size=4
n_steps=102
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

#mask=((df["product"]=="SJ300") | (df["product"]=="AJ300"))
mask=((df["product"]=="SJ300"))

table = pd.pivot_table(df[mask], values='qty', index=['product'],columns=['week'], aggfunc=np.sum, margins=False, fill_value=0)   #, observed=True)
print("\ntable=\n",table.head(5))
#  #   f.write("\n\n"+table.to_string())
print("table created.")



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

sales=table.to_numpy()
sales=sales[-n_steps:]
print(" sales batch=",sales.shape)

# X_batch,y_batch=next_batch(sales,batch_size,n_steps,n_inputs)  # returns a batch_size number (same as instances) of random batches of sales data un the shape [batch_size, n_steps, n_neurons]

# print("sales next batch x_batch=",X_batch.shape)
# print("sales next batch y_batch=",y_batch.shape)


# sales is in the shape product_code, unit sales size  [2, 105]
# this is actually [n_steps,batch_size]






# t = np.linspace(t_min, t_max, int((t_max - t_min) / resolution))

# n_steps = 20
# t_instance = np.linspace(12.2, 12.2 + resolution * (n_steps + 1), n_steps + 1)

# plt.figure(figsize=(11,4))
# plt.subplot(121)
# plt.title("A time series (generated)", fontsize=14)
# plt.plot(t, time_series(t), label=r"$t . \sin(t) / 3 + 2 . \sin(5t)$")
# plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "b-", linewidth=3, label="A training instance")
# plt.legend(loc="lower left", fontsize=14)
# plt.axis([0, 30, -17, 13])
# plt.xlabel("Time")
# plt.ylabel("Value")

# plt.subplot(122)
# plt.title("A training instance", fontsize=14)
# plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="instance")
# plt.plot(t_instance[1:], time_series(t_instance[1:]), "w*", markersize=10, label="target")
# plt.legend(loc="upper left")
# plt.xlabel("Time")


# #save_fig("time_series_plot")
# plt.show()



X_batch, y_batch = next_batch(1, n_steps)



#print(np.c_[X_batch[0], y_batch[0]])




reset_graph()

# n_steps = 20
# n_inputs = 1
# n_neurons = 100
# n_outputs = 1
# learning_rate = 0.001


X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])


cell = tf.nn.rnn_cell.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)
rnn_outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)





stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])


loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()


n_iterations = 1500
batch_size = 50

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



