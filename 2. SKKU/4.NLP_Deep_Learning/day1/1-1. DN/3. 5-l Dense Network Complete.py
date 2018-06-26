import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./dataset/mnist/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)
epsilon = 1e-3

W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
B1 = tf.Variable(tf.random_normal(shape=[256], stddev=0.01))
L1 = tf.matmul(X, W1) + B1
L1 = tf.contrib.layers.batch_norm(L1, is_training=is_training, center=True, scale = True, updates_collections=None)
L1 = tf.nn.relu(L1)
L1 = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
B2 = tf.Variable(tf.random_normal(shape=[256], stddev=0.01))
L2 = tf.matmul(L1, W2) + B2
L2 = tf.contrib.layers.batch_norm(L2, is_training=is_training, center=True, scale = True, updates_collections=None)
L2 = tf.nn.relu(L2)
L2 = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
B3 = tf.Variable(tf.random_normal(shape=[256], stddev=0.01))
L3 = tf.matmul(L2, W3) + B3
L3 = tf.contrib.layers.batch_norm(L3, is_training=is_training, center=True, scale = True, updates_collections=None)
L3 = tf.nn.relu(L3)
L3 = tf.nn.dropout(L3, keep_prob)

W4 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
B4 = tf.Variable(tf.random_normal(shape=[256], stddev=0.01))
L4 = tf.matmul(L3, W4) + B4
L4 = tf.contrib.layers.batch_norm(L4, is_training=is_training, center=True, scale = True, updates_collections=None)
L4 = tf.nn.relu(L4)
L4 = tf.nn.dropout(L4, keep_prob)


W5 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
B5 = tf.Variable(tf.random_normal(shape=[10], stddev=0.01))
model = tf.nn.softmax(tf.matmul(L4, W5) + B5)

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(tf.clip_by_value(model, 1e-10, 1.0)), [1]))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session(config=config)
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

for epoch in range(15):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.7, is_training: True})
        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1), 'Avg. cost =', '{:.3f}'.format(total_cost / total_batch), 'Train Acc. =', sess.run(accuracy, feed_dict={X: mnist.train.images, Y: mnist.train.labels, keep_prob: 1.0, is_training: False}))

print('Training Done!')
print('Test Acc. = ', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0, is_training: False   }))
sess.close()