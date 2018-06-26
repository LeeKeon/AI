import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./dataset/mnist/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 28, 28])
Y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.random_normal([128, 10]))
b = tf.Variable(tf.random_normal([10]))

# BasicRNNCell,BasicLSTMCell,GRUCell
cell = tf.nn.rnn_cell.BasicLSTMCell(128, forget_bias=1.0)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

outputs = tf.transpose(outputs, [1, 0, 2])
outputs = outputs[-1]
model = tf.nn.softmax(tf.matmul(outputs, W) + b)

cost = -tf.reduce_mean(Y*tf.log(tf.clip_by_value(model,1e-10,1.0)))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

batch_size = 128
total_batch = int(mnist.train.num_examples/batch_size)

total_epoch = 15

for epoch in range(total_epoch):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape((batch_size, 28, 28))

        _, cost_val = sess.run([optimizer, cost],
                               feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1),
          'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

test_batch_size = len(mnist.test.images)
test_xs = mnist.test.images.reshape(test_batch_size, 28, 28)
test_ys = mnist.test.labels

print('Accuracy:', sess.run(accuracy,
                       feed_dict={X: test_xs, Y: test_ys}))

sess.close()