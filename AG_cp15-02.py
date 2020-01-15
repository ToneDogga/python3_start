# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 11:50:37 2020

@author: Anthony2013
"""

# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import os
import sys

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
CHAPTER_ID = "autoencoders"

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

def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axis("off")
    
    
def plot_multiple_images(images, n_rows, n_cols, pad=2):
    images = images - images.min()  # make the minimum == 0, so the padding looks white
    w,h = images.shape[1:]
    image = np.zeros(((w+pad)*n_rows+pad, (h+pad)*n_cols+pad))
    for y in range(n_rows):
        for x in range(n_cols):
            image[(y*(h+pad)+pad):(y*(h+pad)+pad+h),(x*(w+pad)+pad):(x*(w+pad)+pad+w)] = images[y*n_cols+x]
    plt.imshow(image, cmap="Greys", interpolation="nearest")
    plt.axis("off")    

def show_reconstructed_digits(X, outputs, model_path = None, n_test_digits = 2):
    with tf.Session() as sess:
        if model_path:
            saver.restore(sess, model_path)
        X_test = mnist.test.images[:n_test_digits]
        outputs_val = outputs.eval(feed_dict={X: X_test})

    fig = plt.figure(figsize=(8, 3 * n_test_digits))
    for digit_index in range(n_test_digits):
        plt.subplot(n_test_digits, 2, digit_index * 2 + 1)
        plot_image(X_test[digit_index])
        plt.subplot(n_test_digits, 2, digit_index * 2 + 2)
        plot_image(outputs_val[digit_index])



# def shuffle_batchX(X, batch_size):
#     rnd_idx = np.random.permutation(len(X))
#     n_batches = len(X) // batch_size
#     for batch_idx in np.array_split(rnd_idx, n_batches):
#         X_batch = X[batch_idx]
#         yield X_batch


def next_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch  = X[batch_idx]
        yield X_batch


# def next_batch(batch_size, n_steps):
#     t0 = np.random.rand(batch_size, 1) * (t_max - t_min - n_steps * resolution)
#     Ts = t0 + np.arange(0., n_steps + 1) * resolution
#     ys = time_series(Ts)
#     return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1)    

#import input_data

#import tensorflow.keras.datasets.mnist as input_data
#mnist = input_data.load_data("MNIST_data")   #, one_hot=True)
#mnist = tfds.load(name = "mnist")
#from tensorflow.examples.tutorials.mnist import input_data
#from tensorflow.keras.datasets.mnist import input_data

import tensorflow.keras.datasets.mnist as tfds





#import tensorflow.contrib.keras.python.keras.datasets.mnist as tfds


#from tensorflow.examples.tutorials.mnist import input_data
#mnist = tfds.read_data_sets("/tmp/data/")
mnist = tfds.load_data("mnist")   #tmp\\data\\")

#  returns a Tuple of Numpy arrays: (x_train, y_train), (x_test, y_test).

print("X_train=",mnist[0][0].shape)
print("y_train=",mnist[0][1].shape)
print("X_test=",mnist[1][0].shape)
print("y_test-",mnist[1][1].shape)

X_train=mnist[0][0]
y_train=mnist[0][1]


#mnist = input_data.read_data_sets("/tmp/data/")

# (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
# X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
# X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
# y_train = y_train.astype(np.int32)
# y_test = y_test.astype(np.int32)
# X_valid, X_train = X_train[:5000], X_train[5000:]
# y_valid, y_train = y_train[:5000], y_train[5000:]

# X_test = X_test.reshape((-1, n_steps, n_inputs))
# # one hot encode for 10 MNIST classes
# def my_one_hot(feature, label):
#     return feature, tf.one_hot(label, depth=10)

# # load your data from tfds
# mnist_train, train_info = tfds.load(name="MNIST_data", with_info=True, as_supervised=True, split=tfds.Split.TRAIN)

# # convert your labels in one-hot
# mnist_train = mnist_train.map(my_one_hot)

# # you can batch your data here
# mnist_train = mnist_train.batch(8)





reset_graph()

from functools import partial

reset_graph()

n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 150  # codings
n_hidden3 = n_hidden1
n_outputs = n_inputs

learning_rate = 0.01
l2_reg = 0.0005


activation = tf.nn.elu
regularizer = tf.keras.regularizers.l2(l=l2_reg)
initializer = tf.variance_scaling_initializer()

X = tf.placeholder(tf.float32, shape=[None, n_inputs])

weights1_init = initializer([n_inputs, n_hidden1])
weights2_init = initializer([n_hidden1, n_hidden2])

weights1 = tf.Variable(weights1_init, dtype=tf.float32, name="weights1")
weights2 = tf.Variable(weights2_init, dtype=tf.float32, name="weights2")
weights3 = tf.transpose(weights2, name="weights3")  # tied weights
weights4 = tf.transpose(weights1, name="weights4")  # tied weights

biases1 = tf.Variable(tf.zeros(n_hidden1), name="biases1")
biases2 = tf.Variable(tf.zeros(n_hidden2), name="biases2")
biases3 = tf.Variable(tf.zeros(n_hidden3), name="biases3")
biases4 = tf.Variable(tf.zeros(n_outputs), name="biases4")

hidden1 = activation(tf.matmul(X, weights1) + biases1)
hidden2 = activation(tf.matmul(hidden1, weights2) + biases2)
hidden3 = activation(tf.matmul(hidden2, weights3) + biases3)
outputs = tf.matmul(hidden3, weights4) + biases4

reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
reg_loss = regularizer(weights1) + regularizer(weights2)
loss = reconstruction_loss + reg_loss

optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()


saver = tf.train.Saver()

n_epochs = 5
batch_size = 150
n_batches=10



with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
     #   n_batches=shuffle_batchX(X_train,batch_size) // batch_size
    #    n_batches = tfds.train.num_examples // batch_size
        for iteration in range(n_batches):
            print("\r{}%".format(100 * iteration // n_batches), end="")
            sys.stdout.flush()
            X_train = X_train.reshape((-1, n_batches, n_inputs))
                   
            X_batch  = next_batch(X_train,y_train,batch_size)
            sess.run(training_op, feed_dict={X: X_batch})
        loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})
        print("\r{}".format(epoch), "Train MSE:", loss_train)
        saver.save(sess, "./my_model_tying_weights.ckpt")

show_reconstructed_digits(X, outputs, "./my_model_tying_weights.ckpt")