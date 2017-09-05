import tensorflow as tf


# Documentation:
# You might think of TensorFlow Core programs as consisting of two discrete sections:
# 1. Building the computational graph.
# 2. Running the computational graph.


# Define tensors:
# [tensor]: an [n]-dimensional array
# Note: These tensors are not initiated because tf.Session is not called

# Build computational graphs:
# Each node takes zero or more tensors as inputs and produces a tensor as an 
# output. One type of node is a constant. Like all TensorFlow constants, 
# it takes no inputs, and it outputs a value it stores internally
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # This becomes tf.float implicitly

print(node1, node2)
# Output:
#$ Tensor("Const:0",   shape=(), dtype=float32)
#$ Tensor("Const_1:0", shape=(), dtype=float32)

# The following code creates a Session object and then invokes its run method to run enough of the computational graph to evaluate node1 and node2. By running the computational graph in a session as follows:

sess = tf.Session()
print(sess.run([node1, node2]))


### Operations
# -- Add existing nodes to a new node: they are basically summed in the new node
node3 = tf.add(node1, node2)

print("node3:", node3)
print("sess.run(node3):", sess.run(node3))


### TensorBoard: Display picture of the computational graph
# [placeholder]: A promise to provide a value later.

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

node_summer = a + b # shortcut for: tf.add(a, b)
node_square = node_summer * node_summer 

parameters1 = sess.run(node_summer, {a: 1, b: 4.5})
parameters2 = sess.run(node_summer, {a: [3,1], b: [7,2]})
parameters3 = sess.run(node_square, {a: 5, b: 5})
        
print("session:", parameters3)


### Variables
# [variable]: To make the model trainable, we need to be able to modify the graph to get new outputs with the same input. Variables allow us to add trainable parameters to a graph. They are constructed with a type and initial value

W = tf.Variable([0.3],  dtype=tf.float32)
b = tf.Variable([-0.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)

y = "training data"
# [loss_function] - A loss function measures how far apart the current model is from the provided data

linear_model = W * x + b

### Constants
# [constants]: They are initialized after calling:
# tf.constant
# [variables] initilization:
init = tf.global_variables_initializer()
sess.run(init) # Until we call sess.run, the variables are uninitialized


### Model: run
run = sess.run(linear_model, {x: [1, 2, 3, 4]})

print(run)

### Loss value function
print ("Loss function:")
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))


### Re-assigning values:
fixW = tf.assign(W, [-1.0])
fixb = tf.assign(b, [1.0])
sess.run([fixW, fixb])

print(sess.run (loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

### Optimizer:
# A complete discussion of machine learning is out of the scope of this tutorial. 
# However, TensorFlow provides optimizers that slowly change each variable in order 
# to minimize the loss function. The simplest optimizer is gradient descent.
# It modifies each variable according to the magnitude of the derivative of loss with 
# respect to that variable. In general, computing symbolic derivatives manually is tedious and error-prone. 
# Consequently, TensorFlow can automatically produce derivatives given only a description 
# of the model using the function tf.gradients. For simplicity, optimizers typically do this for you.

gradient_value = 0.01

optimizer = tf.train.GradientDescentOptimizer( gradient_value )

train = optimizer.minimize(loss)

# reset values:
sess.run(init)

# Training data:
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

for i in range(1000):
    sess.run(train, {x: x_train, y: y_train} )

print ('Optimizer: Gradient optimizer with value:', gradient_value)
print( sess.run([W, b]) )


# Print end result
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

