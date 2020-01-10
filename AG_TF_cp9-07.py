import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler




##
##
##
##housing=fetch_california_housing()
##
##m,n = housing.data.shape
###print("m,n",m,n)
###housing_data_plus_bias=np.c_[np.ones((m,1)), housing.data]
##
##
###X = housing_data_plus_bias
###y = housing.target.reshape(-1, 1)
##
##scaler = StandardScaler()
##scaled_housing_data = scaler.fit_transform(housing.data)
##scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]



tf.reset_default_graph()

from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)
#logdir = "."   #/tf_logs/"    #"/run-{}/".format(now)


##
##def fetch_batch(epoch, batch_index, batch_size):
##    np.random.seed(epoch * n_batches + batch_index)  # not shown in the book
##    indices = np.random.randint(m, size=batch_size)  # not shown
##    X_batch = scaled_housing_data_plus_bias[indices] # not shown
##    y_batch = housing.target.reshape(-1, 1)[indices] # not shown
##    return X_batch, y_batch
##



def relu(X):
    with tf.name_scope("relu"):
        w_shape = (int(X.get_shape()[1]), 1)                          # not shown in the book
        w = tf.Variable(tf.random_normal(w_shape), name="weights")    # not shown
        b = tf.Variable(0.0, name="bias")                             # not shown
        z = tf.add(tf.matmul(X, w), b, name="z")                      # not shown
        return tf.maximum(z, 0., name="max")

n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name="output")

file_writer = tf.summary.FileWriter("tf_logs/relu2", tf.get_default_graph())
file_writer.close()

##    
##n_epochs = 10
##batch_size = 100
##n_batches = int(np.ceil(m / batch_size))
##learning_rate = 0.01
##
##X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
##y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
##theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
##y_pred = tf.matmul(X, theta, name="predictions")
##
##with tf.name_scope("loss") as scope:
##    error = y_pred - y
##    mse = tf.reduce_mean(tf.square(error), name="mse")
##
##
##
###error = y_pred - y
###mse = tf.reduce_mean(tf.square(error), name="mse")
##optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
##training_op = optimizer.minimize(mse)
##
##init = tf.global_variables_initializer()
##
##mse_summary = tf.summary.scalar('MSE', mse)
##file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
##
##
##
##
##with tf.Session() as sess:                                                        # not shown in the book
##    sess.run(init)                                                                # not shown
##
##
##
##    for epoch in range(n_epochs):                                                 # not shown
##        for batch_index in range(n_batches):
##            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
##            if batch_index % 10 == 0:
##                summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
##                step = epoch * n_batches + batch_index
##                file_writer.add_summary(summary_str, step)
##            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
##
##    best_theta = theta.eval()      
##
##
##print("best theta",best_theta)     
##file_writer.close()

##
##
##with tf.Session() as sess:
##    saver.restore(sess, "my_model_final.ckpt")
##    best_theta_restored = theta.eval() # not shown in the book
##
##
##print("correct restore?",np.allclose(best_theta, best_theta_restored))
##
##    
