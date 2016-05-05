import numpy as np
import tensorflow as tf
import math
import os
import sys
#import matplotlib.pyplot as plt
from data_utils import load_CIFAR100 as load

path = sys.argv[1]
#file = open('loss.txt','w')
batch_size = 100
num_train = 49000
num_val = 1000
num_test = 1000
dropout = 0.5
display_step = 10
reg = 0.1
drop = False
learning_rate = 0.01
first_layer = 20
second_layer = 50
verbose = True
test = True
fsize = 5

Xtr, Ytr, Xte, Yte = load(path)

i_placeholder = tf.placeholder(tf.float32, shape=[batch_size,32,32,3], name='images')
l_placeholder = tf.placeholder(tf.float32, shape=(batch_size,20,),name='labels')
drop_placeholder = tf.placeholder(tf.float32)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_nxn(x, n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='SAME')

def one_hot(labels, num_classes=20):
    num_labels = labels.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels.ravel()] = 1
    return labels_one_hot

def l2(var):
    return tf.nn.l2_loss(var)

# Convolution with 4 feature maps
#W1 = tf.constant(0.1, shape=[5,5,3,first_layer])
W1 = tf.Variable(tf.truncated_normal(shape=[fsize, fsize, 3, first_layer],stddev=1e-3))
b1 = tf.constant(0.0, shape=[first_layer])
h1 = tf.nn.relu(conv2d(i_placeholder, W1) + b1)

#h1_pool = h1
h1_pool = max_pool_nxn(h1, 2)

if drop:
    h1_pool = tf.nn.dropout(h1_pool, dropout)

# Convolution with 6 feature maps
#W2 = tf.constant(0.1, shape=[5,5,first_layer,second_layer])
W2 = tf.Variable(tf.truncated_normal(shape=[fsize, fsize, first_layer, second_layer],stddev=1e-3))
b2 = tf.constant(0.0, shape=[second_layer])
h2 = tf.nn.relu(conv2d(h1_pool, W2) + b2)

#h2_pool = h2
h2_pool = max_pool_nxn(h2, 2)

if drop:
    h2_pool = tf.nn.dropout(h2_pool, dropout)

# Fully connected layer
input_size = W2.get_shape().as_list()[-1]*8*8
Wconn = tf.Variable(tf.truncated_normal([input_size, 1024],stddev=1.0/math.sqrt(float(input_size))))
bconn = tf.Variable(tf.zeros([1024]))
h3 = tf.reshape(h2_pool, [-1, Wconn.get_shape().as_list()[0]]) # Reshape conv2 output to fit dense layer input
h3 = tf.nn.relu(tf.matmul(h3, Wconn) + bconn)

if drop:
    h3 = tf.nn.dropout(h3, dropout)

# Output, class prediction
Wout = tf.Variable(tf.truncated_normal([1024,20],stddev=1e-3))
bout = tf.Variable(tf.zeros([20]))
out = tf.matmul(h3, Wout) + bout

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out, l_placeholder, name='xentropy'))
    regularizers = l2(W2)+l2(b2)+l2(W1)+l2(b1)+l2(Wconn)+l2(bconn)+l2(Wout)+l2(bout)
    loss += reg*regularizers

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


correct_pred = tf.equal(tf.argmax(out,1), tf.argmax(l_placeholder,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    step = 1
    while step < 5000:
        index = np.random.random_integers(0,num_train-1,batch_size)
        x_batch = Xtr[index]
        y_batch = Ytr[index]
        y_batch = one_hot(y_batch)
        # Fit training using batch data
        sess.run(optimizer, feed_dict={i_placeholder: x_batch, l_placeholder: y_batch, drop_placeholder: dropout})
        if step % display_step == 0 and verbose:
            acc = sess.run(accuracy, feed_dict={i_placeholder: x_batch, l_placeholder: y_batch, drop_placeholder: 1.})
            b_loss = sess.run(loss, feed_dict={i_placeholder: x_batch, l_placeholder: y_batch, drop_placeholder: 1.})
            print ("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(b_loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
            iteration = step*batch_size
            # file.write(str(iteration))
            # file.write(": ")
            # file.write(str(b_loss))
            # file.write("\n")
        step += 1
    print "Optimization Finished!"
    #file.close()
    #Calculate accuracy for test images
    if test:
        step = 0
        test_acc = 0.0
        while step < num_test/batch_size-1:
            start = step*batch_size
            x_batch = Xte[start:(start+batch_size)]
            y_batch = Yte[start:(start+batch_size)]
            y_batch = one_hot(y_batch)
            test_acc += sess.run(accuracy, feed_dict={i_placeholder: x_batch, l_placeholder: y_batch, drop_placeholder: 1.})
            step += 1
        test_acc /= num_test/batch_size

        print "Testing Accuracy:", test_acc
