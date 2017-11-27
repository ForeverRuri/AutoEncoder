import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

input_dim = 784
hidden1_num = 200
hidden2_num = 100
learning_rate = 0.01
train_steps = 10000
batch_size = 128

weights = {
    'encoder_hidden1' : tf.Variable(tf.random_normal([input_dim, hidden1_num])),
    'encoder_hidden2' : tf.Variable(tf.random_normal([hidden1_num, hidden2_num])),
    'decoder_hidden1' : tf.Variable(tf.random_normal([hidden2_num, hidden1_num])),
    'decoder_hidden2' : tf.Variable(tf.random_normal([hidden1_num, input_dim]))
}

bias = {
    'encoder_hidden1' : tf.Variable(tf.random_normal([hidden1_num])),
    'encoder_hidden2' : tf.Variable(tf.random_normal([hidden2_num])),
    'decoder_hidden1' : tf.Variable(tf.random_normal([hidden1_num])),
    'decoder_hidden2' : tf.Variable(tf.random_normal([input_dim]))
}



def Encoder(x):
    out = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_hidden1']), bias['encoder_hidden1']))
    out = tf.nn.sigmoid(tf.add(tf.matmul(out, weights['encoder_hidden2']), bias['encoder_hidden2']))
    return out

def Decoder(x):
    out = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_hidden1']), bias['decoder_hidden1']))
    out = tf.nn.sigmoid(tf.add(tf.matmul(out, weights['decoder_hidden2']), bias['decoder_hidden2']))
    return out

x = tf.placeholder(tf.float32, [None, 784])

x_ = Encoder(x)
re_x = Decoder(x_)

cost = tf.reduce_mean(tf.pow((x - re_x), 2))

optimize_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1, 1 + train_steps):
    batch_x, _ = mnist.train.next_batch(batch_size)
    loss,_ = sess.run([cost, optimize_op],
                        feed_dict= {x : batch_x})
    if i % 100 == 0:
        print('The loss of %s is %f'%(i, loss))

n = 6
canvas_rebuild = np.empty([n * 28, n * 28])
canvas_origin = np.empty([n * 28, n * 28])
for i in range(n):
    batch_x, _ = mnist.train.next_batch(n)
    recons = sess.run(re_x, feed_dict = {x : batch_x})
    for j in range(n):
        canvas_rebuild[i * 28: i * 28 + 28, j * 28:j * 28 + 28] = recons[j].reshape([28, 28])
        canvas_origin[i * 28: i * 28 + 28, j * 28:j * 28 + 28] = batch_x[j].reshape([28, 28])

plt.figure(1)
plt.imshow(canvas_origin, cmap = 'gray')
plt.figure(2)
plt.imshow(canvas_rebuild, cmap = 'gray')
plt.show()