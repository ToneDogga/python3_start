# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 11:16:51 2020

@author: Anthony2013
"""
from __future__ import absolute_import, division, print_function
#from __future__ import absolute_import, division, print_function, unicode_literal

try:
  import tensorflow.compat.v2 as tf
except Exception:
  pass

#tf.enable_eager_execution()
tf.compat.v1.enable_eager_execution(
    config=None,
    device_policy=None,
    execution_mode=None
)

tf.enable_v2_behavior()


print("tensorflow:",tf.__version__)

#import tensorflow as tf

#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()

def variables_on_cpu(op):
    if op.type=="Variable":
        return "/cpu:0"
    else:
        return "/gpu:0"


@tf.function
def test(x):
  return (x+1)**4


#


with tf.device("/gpu:0"):
    W = tf.Variable(tf.random.normal(shape=(10,20,30,10),seed=42), name="W")
    b = tf.Variable(tf.random.uniform(shape=(10,20,30,10),seed=42), name="b")
    c = tf.Variable(tf.random.normal(shape=(10,20,30,10),seed=42), name="c")
    
    @tf.function
    def forward(x):
      return W * x + test(b)
    
    out_a = forward(c)
#print("one",out_a)

    

    
    out_c = forward(out_a)
    out_d=tf.multiply(b,out_c)*1000
    print("two",out_d)
    test=tf.math.sqrt(out_d)
    print("three",out_d)

#initializer = tf.initializers.random_normal(-1, 1)

    
# with tf.device("/gpu:0"):   #variables_on_cpu):       
#     rand_t = tf.random.uniform([5], 0, 10, dtype=tf.int32, seed=42)
#     a = tf.Variable(rand_t,name="a")
#     b = tf.Variable(rand_t,name="b")
#     c=tf.add(a,b)
# #    init=tf.global_variables_initialiser()
# #    init=tf.global_variables_initializer()
#     print("c=",c.eval())
    
    
#     # with tf.Session() as sess:
#     #     sess.run(init)
    #     print("two",c.eval())
