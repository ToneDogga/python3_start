#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  2 09:48:08 2020

@author: tonedogga
"""


# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

# try:
#     # %tensorflow_version only exists in Colab.
#  #   %tensorflow_version 2.x
#     IS_COLAB = True
# except Exception:
#     IS_COLAB = False

# TensorFlow ≥2.0 is required
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)




from tensorflow import keras
assert tf.__version__ >= "2.0"

#if not tf.config.list_physical_devices('GPU'):
#    print("No GPU was detected. LSTMs and CNNs can be very slow without a GPU.")
#    if IS_COLAB:
#        print("Go to Runtime > Change runtime and select a GPU hardware accelerator.")


# Disable all GPUS 
#tf.config.set_visible_devices([], 'GPU') 



 #visible_devices = tf.config.get_visible_devices() 
# for device in visible_devices: 
#     print(device)
#     assert device.device_type != 'GPU' 

#tf.config.set_visible_devices([], 'GPU') 
#tf.config.set_visible_devices(visible_devices, 'GPU') 


# Common imports
import numpy as np
import os
from pathlib import Path
import pandas as pd
import pickle
#import random

# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

# To plot pretty figures
#%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)


visible_devices = tf.config.get_visible_devices('GPU') 
print("tf.config.get_visible_devices('GPU'):",visible_devices)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)   # turn off traceback errors



# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rnn"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
 
    
 
def plot_learning_curves(loss, val_loss):
    plt.plot(np.arange(len(loss)) + 0.5, loss, "b.-", label="Training loss")
    plt.plot(np.arange(len(val_loss)) + 1, val_loss, "r.-", label="Validation loss")
    plt.gca().xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    plt.axis([1, 20, 0, np.max(loss)])
    plt.legend(fontsize=14)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    
    
#  series = generate_time_series(1, 50 + 10)
#  print("series.shape=",series.shape)
#     1, 60 ,1 
    
# def generate_time_series2(batch_size, n_steps):
#     freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
#     time = np.linspace(0, 1, n_steps)
#     series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  #   wave 1
#     series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # + wave 2
#     series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)   # + noise
#     return series[..., np.newaxis].astype(np.float32)    



def build_mini_batches(series,no_of_batches,n_steps):
#    print("build",no_of_batches,"mini batches")
  #  batch_length+=1  # add an extra step which is the target (y)
#    np.random.seed(45)
    total_steps=series.shape[1]
    
 #   print("total stesp=",total_steps)
 #   print("no_of_batches to build=",no_of_batches)
 #   print("no of steps in each batch=",n_steps)
    if no_of_batches<(total_steps-n_steps):
        print("\nonly one \n")
        repeats_needed=1
        gridtest=np.meshgrid(np.arange(0,n_steps),np.random.randint(0,total_steps-n_steps+1))
        print("raandom",gridtest)
    else:    
        repeats_needed=round(no_of_batches/(total_steps-n_steps),0) 
        gridtest=np.meshgrid(np.arange(0,n_steps),np.arange(0,total_steps-n_steps+1))

    #gridtest=np.meshgrid(np.arange(np.random.random_integers(0,total_steps,n_steps))), np.arange(0,n_steps))
   # print(gridtest.shape)  #.shape)
    start_index=np.repeat(gridtest[0]+gridtest[1],repeats_needed,axis=0)   #[:,:,np.newaxis]
    #print("start index",start_index,start_index.shape)
    np.random.shuffle(start_index)
    print("start index",start_index,start_index.shape)
 
    new_batches=series[0,start_index]
    np.random.shuffle(new_batches)
   # print(new_batches)
    if repeats_needed==1:
        print(" one only - batches complete. batches shape:",new_batches.shape)
    
    return new_batches   #,new_batches[:,1:batch_length+1,:]



# def build_mini_batch_input(series,no_of_batches,batch_length):
#     print("build",no_of_batches,"mini batches")
#   #  batch_length+=1  # add an extra step which is the target (y)
#     np.random.seed(45)
#     no_of_steps=series.shape[1]
    
#     print("batch_length=",batch_length)
#     print("no of steps=",no_of_steps)
 
#     repeats_needed=round(no_of_batches/(no_of_steps-batch_length),0)
    
#     gridtest=np.meshgrid(np.arange(0,batch_length+1),np.arange(0,no_of_steps-(batch_length+1)))
#     start_index=np.repeat(gridtest[0]+gridtest[1],repeats_needed,axis=0)   #[:,:,np.newaxis]
#     np.random.shuffle(start_index)
#     new_batches=series[0,start_index,:]
#     np.random.shuffle(new_batches)
#     print("batches complete. batches shape:",new_batches.shape)
#     return new_batches[:,:batch_length,:],new_batches[:,1:batch_length+1,:]




  
    
def generate_time_series(no_of_batches, n_steps):    
    with open("batch_dict.pkl", "rb") as f:
         batches = pickle.load(f)
    mat_sales_x =batches[0][7]
    #series_table=batches[0][9]    
    series_batch=build_mini_batches(mat_sales_x,no_of_batches,n_steps) 
    print("series_batch.shape",series_batch.shape)
    return series_batch   #[..., np.newaxis].astype(np.float32)    


def plot_series(series, y=None, y_pred=None, x_label="$date$", y_label="$units/day$"):
    plt.plot(series, ".-")
    if y is not None:
        plt.plot(n_steps, y, "bx", markersize=10)
    if y_pred is not None:
        plt.plot(n_steps, y_pred, "ro")
    plt.grid(True)
    if x_label:
        plt.xlabel(x_label, fontsize=16)
    if y_label:
        plt.ylabel(y_label, fontsize=16, rotation=90)
    plt.hlines(0, 0, np.max(series)+1, linewidth=1)
  #  plt.axis([0, n_steps + 1, -1, 1])
    plt.axis([0, n_steps + 11, 0 , 1000])


def plot_multiple_forecasts(X, Y, Y_pred,title_name):
    n_steps = X.shape[1]
    ahead = Y.shape[1]
    plot_series(X[0, :, 0])
    if title_name:
        plt.title(title_name)
    plt.plot(np.arange(n_steps, n_steps + ahead), Y[0, :, 0], "ro-", label="Actual")
    plt.plot(np.arange(n_steps, n_steps + ahead), Y_pred[0, :, 0], "bx-", label="Forecast", markersize=10)
#    plt.axis([0, n_steps + ahead, -1, 1])
    plt.axis([0, n_steps + ahead, 0 , np.max([np.max(Y),np.max(Y_pred)])])

    plt.legend(fontsize=14)

def last_time_step_mse(Y_true, Y_pred):
    return keras.metrics.mean_squared_error(Y_true[:, -1], Y_pred[:, -1])




# fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 4))
# for col in range(3):
#     plt.sca(axes[col])
#     plot_series(X_valid[col, :, 0], y_valid[col, 0],
#                 y_label=("$x(t)$" if col==0 else None))
# save_fig("time_series_plot")
# plt.show()




###############3

#print("use this model to predict the nexzt 10 values")

np.random.seed(42)

n_steps = 50
series = generate_time_series(10000, n_steps + 10)
X_train, Y_train = series[:7000, :n_steps], series[:7000, -10:, 0]
X_valid, Y_valid = series[7000:9000, :n_steps], series[7000:9000, -10:, 0]
X_test, Y_test = series[9000:, :n_steps], series[9000:, -10:, 0]



print(X_test.shape,Y_test.shape)
# X = X_valid
# for step_ahead in range(10):
#     y_pred_one = model.predict(X)[:, np.newaxis, :]
#     X = np.concatenate([X, y_pred_one], axis=1)

# Y_pred = X[:, n_steps:, 0]


# print("Y_pred.shape",Y_pred.shape)


# print("mse",np.mean(keras.metrics.mean_squared_error(Y_valid, Y_pred)))

# Y_naive_pred = Y_valid[:, -1:]
# print("Y_naive pred",np.mean(keras.metrics.mean_squared_error(Y_valid, Y_naive_pred)))


##################################

print("Now lets create an RNN that predicts all 10 next values at once:")


np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20),
    keras.layers.Dense(10)
])

model.compile(loss="mse", optimizer="adam")
history = model.fit(X_train, Y_train, epochs=20,
                    validation_data=(X_valid, Y_valid))


plot_learning_curves(history.history["loss"], history.history["val_loss"])
plt.show()


np.random.seed(43)

series = generate_time_series(1, 50 + 10)
print("series.shape=",series.shape)


X_new, Y_new = series[:, :50, :], series[:, -10:, :]
Y_pred = model.predict(X_new)[..., np.newaxis]

print("X_new",X_new.shape)
print("Y_new",Y_new.shape)
print("Y_pred",Y_pred.shape)



plot_multiple_forecasts(X_new, Y_new, Y_pred,"RNN")
plt.show()


###########################################3


#  Now let's create an RNN that predicts the next 10 steps at each time step. 
# That is, instead of just forecasting time steps 50 to 59 based on time steps 0 to 49,
#  it will forecast time steps 1 to 10 at time step 0, then time steps 2 to 11 at time 
# step 1, and so on, and finally it will forecast time steps 50 to 59 at the last time step.
#  Notice that the model is causal: when it makes predictions at any time step, 
# it can only see past time steps.

print("cretae an RNN that predicts the next 10 steps at each time step")

np.random.seed(42)

n_steps = 50
series = generate_time_series(10000, n_steps + 10)
print("series.shape",series.shape)
X_train = series[:7000, :n_steps]
X_valid = series[7000:9000, :n_steps]
X_test = series[9000:, :n_steps]
Y = np.empty((series.shape[0], n_steps, 10))
for step_ahead in range(1, 10 + 1):
    Y[..., step_ahead - 1] = series[..., step_ahead:step_ahead + n_steps, 0]
Y_train = Y[:7000]
Y_valid = Y[7000:9000]
Y_test = Y[9000:]


print("Y shape",Y.shape)

print("X_train",X_train.shape)
print("Y_train",Y_train.shape)




np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.SimpleRNN(20, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(10))
])


model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=0.01), metrics=[last_time_step_mse])
history = model.fit(X_train, Y_train, epochs=100,
                    validation_data=(X_valid, Y_valid))


plot_learning_curves(history.history["loss"], history.history["val_loss"])
plt.show()

np.random.seed(43)

series = generate_time_series(1, 50 + 10)
X_new, Y_new = series[:, :50, :], series[:, 50:, :]
Y_pred = model.predict(X_new)[:, -1][..., np.newaxis]

print("series.shape",series.shape)

print("Y pred shape",Y_pred.shape)

print("X_new",X_new.shape)
print("Y_new",Y_new.shape)

print("Y pred shape",Y_pred.shape)

plot_multiple_forecasts(X_new, Y_new, Y_pred,"RNN +10")
plt.show()


#######################################3


print("Deep RNN with batch norm")


np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.BatchNormalization(),
    keras.layers.SimpleRNN(20, return_sequences=True),
    keras.layers.BatchNormalization(),
    keras.layers.TimeDistributed(keras.layers.Dense(10))
])

model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
history = model.fit(X_train, Y_train, epochs=100,
                    validation_data=(X_valid, Y_valid))


plot_learning_curves(history.history["loss"], history.history["val_loss"])
plt.show()

np.random.seed(43)

series = generate_time_series(1, 50 + 10)
X_new, Y_new = series[:, :50, :], series[:, 50:, :]
Y_pred = model.predict(X_new)[:, -1][..., np.newaxis]

print("DRNN series.shape",series.shape)

print("DRNN Y pred shape",Y_pred.shape)

print("DRNN X_new",X_new.shape)
print("DRNN Y_new",Y_new.shape)

print("DRNN Y pred shape",Y_pred.shape)

plot_multiple_forecasts(X_new, Y_new, Y_pred,"Deep RNN with batch norm")
plt.show()

###########################################################33

print("layer norm with a simple rNN")

from tensorflow.keras.layers import LayerNormalization


class LNSimpleRNNCell(keras.layers.Layer):
    def __init__(self, units, activation="tanh", **kwargs):
        super().__init__(**kwargs)
        self.state_size = units
        self.output_size = units
        self.simple_rnn_cell = keras.layers.SimpleRNNCell(units,
                                                          activation=None)
        self.layer_norm = LayerNormalization()
        self.activation = keras.activations.get(activation)
    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        if inputs is not None:
            batch_size = tf.shape(inputs)[0]
            dtype = inputs.dtype
        return [tf.zeros([batch_size, self.state_size], dtype=dtype)]
    def call(self, inputs, states):
        outputs, new_states = self.simple_rnn_cell(inputs, states)
        norm_outputs = self.activation(self.layer_norm(outputs))
        return norm_outputs, [norm_outputs]
    
    
    
    
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.RNN(LNSimpleRNNCell(20), return_sequences=True,
                      input_shape=[None, 1]),
    keras.layers.RNN(LNSimpleRNNCell(20), return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(10))
])

model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
history = model.fit(X_train, Y_train, epochs=100,
                    validation_data=(X_valid, Y_valid))    


plot_learning_curves(history.history["loss"], history.history["val_loss"])
plt.show()


np.random.seed(43)

series = generate_time_series(1, 50 + 10)
X_new, Y_new = series[:, :50, :], series[:, 50:, :]
Y_pred = model.predict(X_new)[:, -1][..., np.newaxis]

print("LN series.shape",series.shape)

print("LN Y pred shape",Y_pred.shape)

print("LN X_new",X_new.shape)
print("LN Y_new",Y_new.shape)

print("LN Y pred shape",Y_pred.shape)

plot_multiple_forecasts(X_new, Y_new, Y_pred,"layer norm with simple RNN")
plt.show()

######################################################

print("\ncreating a custom RNN class")

class MyRNN(keras.layers.Layer):
    def __init__(self, cell, return_sequences=False, **kwargs):
        super().__init__(**kwargs)
        self.cell = cell
        self.return_sequences = return_sequences
        self.get_initial_state = getattr(
            self.cell, "get_initial_state", self.fallback_initial_state)
    def fallback_initial_state(self, inputs):
        return [tf.zeros([self.cell.state_size], dtype=inputs.dtype)]
    @tf.function
    def call(self, inputs):
        states = self.get_initial_state(inputs)
        n_steps = tf.shape(inputs)[1]
        if self.return_sequences:
            sequences = tf.TensorArray(inputs.dtype, size=n_steps)
        outputs = tf.zeros(shape=[n_steps, self.cell.output_size], dtype=inputs.dtype)
        for step in tf.range(n_steps):
            outputs, states = self.cell(inputs[:, step], states)
            if self.return_sequences:
                sequences = sequences.write(step, outputs)
        if self.return_sequences:
            return sequences.stack()
        else:
            return outputs

np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    MyRNN(LNSimpleRNNCell(20), return_sequences=True,
          input_shape=[None, 1]),
    MyRNN(LNSimpleRNNCell(20), return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(10))
])

model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
history = model.fit(X_train, Y_train, epochs=100,
                    validation_data=(X_valid, Y_valid))


plot_learning_curves(history.history["loss"], history.history["val_loss"])
plt.show()


np.random.seed(43)

series = generate_time_series(1, 50 + 10)
X_new, Y_new = series[:, :50, :], series[:, 50:, :]
Y_pred = model.predict(X_new)[:, -1][..., np.newaxis]

print("Custom RNN series.shape",series.shape)

print("Custom RNN Y pred shape",Y_pred.shape)

print("Custom RNN X_new",X_new.shape)
print("Custom RNN Y_new",Y_new.shape)

print("Custom RNN Y pred shape",Y_pred.shape)

plot_multiple_forecasts(X_new, Y_new, Y_pred,"custom RNN class")
plt.show()


######################################################3

print("\nLSTM")


np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.LSTM(20, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(10))
])

model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
history = model.fit(X_train, Y_train, epochs=100,
                    validation_data=(X_valid, Y_valid))

model.evaluate(X_valid, Y_valid)


plot_learning_curves(history.history["loss"], history.history["val_loss"])
plt.show()


np.random.seed(43)

series = generate_time_series(1, 50 + 10)
X_new, Y_new = series[:, :50, :], series[:, 50:, :]
Y_pred = model.predict(X_new)[:, -1][..., np.newaxis]

print("LSTM series.shape",series.shape)

print("LSTM Y pred shape",Y_pred.shape)

print("LSTM X_new",X_new.shape)
print("LSTM Y_new",Y_new.shape)

print("LSTM Y pred shape",Y_pred.shape)

plot_multiple_forecasts(X_new, Y_new, Y_pred,"LSTM")
plt.show()

#################################################

print("grus")


np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.GRU(20, return_sequences=True, input_shape=[None, 1]),
    keras.layers.GRU(20, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(10))
])

model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
history = model.fit(X_train, Y_train, epochs=100,
                    validation_data=(X_valid, Y_valid))

model.evaluate(X_valid, Y_valid)


plot_learning_curves(history.history["loss"], history.history["val_loss"])
plt.show()

# np.random.seed(43)

# series = generate_time_series(1, 50 + 10)
# X_new, Y_new = series[:, :50, :], series[:, 50:, :]
# Y_pred = model.predict(X_new)[:, -1][..., np.newaxis]

np.random.seed(43)

series = generate_time_series(1, 50 + 10)
X_new, Y_new = series[:, :50, :], series[:, 50:, :]
Y_pred = model.predict(X_new)[:, -1][..., np.newaxis]

print("gru series.shape",series.shape)

print("gru Y pred shape",Y_pred.shape)

print("gru X_new",X_new.shape)
print("gru Y_new",Y_new.shape)

print("gru Y pred shape",Y_pred.shape)

plot_multiple_forecasts(X_new, Y_new, Y_pred,"gru")
plt.show()

############################################

print("using 1D conv to process sequ")

np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Conv1D(filters=20, kernel_size=4, strides=2, padding="valid",
                        input_shape=[None, 1]),
    keras.layers.GRU(20, return_sequences=True),
    keras.layers.GRU(20, return_sequences=True),
    keras.layers.TimeDistributed(keras.layers.Dense(10))
])

model.compile(loss="mse", optimizer="adam", metrics=[last_time_step_mse])
history = model.fit(X_train, Y_train[:, 3::2], epochs=100,
                    validation_data=(X_valid, Y_valid[:, 3::2]))


#model.evaluate(X_valid, Y_valid)


plot_learning_curves(history.history["loss"], history.history["val_loss"])
plt.show()

# np.random.seed(43)

# series = generate_time_series(1, 50 + 10)
# X_new, Y_new = series[:, :50, :], series[:, 50:, :]
# Y_pred = model.predict(X_new)[:, -1][..., np.newaxis]

np.random.seed(43)

series = generate_time_series(1, 50 + 10)
X_new, Y_new = series[:, :50, :], series[:, 50:, :]
Y_pred = model.predict(X_new)[:, -1][..., np.newaxis]

print("1D conv series.shape",series.shape)

print("1D conv Y pred shape",Y_pred.shape)

print("1D conv X_new",X_new.shape)
print("1D conv Y_new",Y_new.shape)

print("1d conv Y pred shape",Y_pred.shape)

plot_multiple_forecasts(X_new, Y_new, Y_pred,"1D conv")
plt.show()


