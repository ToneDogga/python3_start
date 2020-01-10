import tensorflow as tf
x=tf.Variable(3, name="x")
y=tf.Variable(4, name='y')
f=x*x*y+y+2

sess=tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result1=sess.run(f)
print(result1)
sess.close()


print("next2\n\n")
with tf.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result2=f.eval()

print(result2)

print("next3\n\n")

init=tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    result3=f.eval()

print(result3)

print("next4\n\n")
sess=tf.InteractiveSession()
init.run()
result4=f.eval()
print(result4)
sess.close()


x1=tf.Variable(1)
print(x1.graph is tf.get_default_graph())

graph=tf.Graph()
with graph.as_default():
    x2=tf.Variable(2)


tf.reset_default_graph()

w=tf.constant(3)
x=w+2
y=x+5
z=x*3

with tf.Session() as sess:
    print(y.eval())
    print(z.eval())


with tf.Session() as sess:
    y_val, z_val =sess.run([y,z])
    print(y_val)
    print(z_val)

    

    
