import tensorflow as tf
from tensorflow.python.layers.core import Dense

# Write your code here
class Model(object):
    def __init__(self, use_clip=True, class_size=26, learning_rate=0.01, keep_prob=1.0):
        self.initializer = tf.random_uniform_initializer(-0.1, 0.1)
        self.use_clip = use_clip
        self.class_size = class_size
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob

        self.x_image = tf.placeholder(dtype=tf.float32, shape=(None, 134, 91, 3))
        self.y_label = tf.placeholder(dtype=tf.int32, shape=(None, ))

        self.build_model()
        self.build_loss()
        self.build_opt()

    def build_model(self):
        conv_f1 = self.get_var(name='conv_f1', shape=[3, 3, 3, 256])
        conv_b1 = self.get_var(name='conv_b1', shape=[256])
        conv1 = tf.nn.relu(tf.nn.conv2d(self.x_image, filter=conv_f1, strides=[1, 1, 1, 1], padding='SAME') + conv_b1)
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        conv_f2 = self.get_var(name='conv_f2', shape=[3, 3, 256, 128])
        conv_b2 = self.get_var(name='conv_b2', shape=[128])
        conv2 = tf.nn.relu(tf.nn.conv2d(conv1, filter=conv_f2, strides=[1, 1, 1, 1], padding='SAME') + conv_b2)
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
#         conv2 = tf.nn.dropout(conv2, self.keep_prob)

        conv_f3 = self.get_var(name='conv_f3', shape=[3, 3, 128, 128])
        conv_b3 = self.get_var(name='conv_b3', shape=[128])
        conv3 = tf.nn.relu(tf.nn.conv2d(conv2, filter=conv_f3, strides=[1, 1, 1, 1], padding='SAME') + conv_b3)
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
#         conv3 = tf.nn.dropout(conv3, self.keep_prob)

        conv_f4 = self.get_var(name='conv_f4', shape=[3, 3, 128, 128])
        conv_b4 = self.get_var(name='conv_b4', shape=[128])
        conv4 = tf.nn.relu(tf.nn.conv2d(conv3, filter=conv_f4, strides=[1, 1, 1, 1], padding='SAME') + conv_b4)
        conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
#         conv4 = tf.nn.dropout(conv4, self.keep_prob)

        conv_f5 = self.get_var(name='conv_f5', shape=[3, 3, 128, 64])
        conv_b5 = self.get_var(name='conv_b5', shape=[64])
        conv5 = tf.nn.relu(tf.nn.conv2d(conv4, filter=conv_f5, strides=[1, 1, 1, 1], padding='SAME') + conv_b5)
        conv5 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
#         conv5 = tf.nn.dropout(conv5, self.keep_prob)

        w1 = self.get_var(name='w1', shape=[4 * 2 * 64, 256])
        b1 = self.get_var(name='b1', shape=[256])
        l1 = tf.reshape(conv5, [-1, 4 * 2 * 64])
        l2 = tf.nn.relu(tf.matmul(l1, w1) + b1)

        w2 = self.get_var(name='w2', shape=[256, 128])
        b2 = self.get_var(name='b2', shape=[128])
        l3 = tf.nn.relu(tf.matmul(l2, w2) + b2)

        w3 = self.get_var(name='w3', shape=[128, 23])
        b3 = self.get_var(name='b3', shape=[23])
        self.genre_prob = tf.nn.softmax(tf.matmul(l3, w3) + b3)

        #최종 결과
        self.y_pred = tf.argmax(self.genre_prob, 1)

    def build_loss(self):
        self.o1 = tf.reshape(self.y_label, [-1])
        self.o2 = tf.to_int32(self.o1)
        self.chk = tf.one_hot(self.o2, self.class_size, 1.0, 0.0)

        self.cross_entropy = -tf.reduce_sum(self.chk
            * tf.log(tf.clip_by_value(tf.reshape(self.genre_prob, [-1, self.class_size]), 1e-20, 1.0)), 1)
        self.loss = tf.reduce_mean(self.cross_entropy)

        # self.cross_entropy = -tf.reduce_sum(
        #     tf.one_hot(tf.to_int32(tf.reshape(self.y_label, [-1])), self.class_size, 1.0, 0.0)
        #     * tf.log(tf.clip_by_value(tf.reshape(self.genre_prob, [-1, self.class_size]), 1e-20, 1.0)), 1)
        # self.loss = tf.reduce_mean(self.cross_entropy)

    def build_opt(self):
        # define optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        grad, var = zip(*optimizer.compute_gradients(self.loss))

        # gradient clipping
        def clipped_grad(grad):
            return [None if g is None else tf.clip_by_norm(g, 2.5) for g in grad]

        if self.use_clip:
            grad = clipped_grad(grad)

        self.update = optimizer.apply_gradients(zip(grad, var))

    def leaky_relu(self, x):
        return tf.maximum((x), 0.1*(x))

    def get_var(self, name='', shape=None, dtype=tf.float32):
        return tf.get_variable(name, shape, dtype=dtype, initializer=self.initializer)


    def save(self, sess, global_step=None):
        var_list = [var for var in tf.all_variables()]
        saver = tf.train.Saver(var_list)
        save_path = saver.save(sess, save_path="models/cnn", global_step=global_step)
        print(' * model saved at \'{}\''.format(save_path))

    # Load whole weights
    def restore(self, sess):
        print(' - Restoring variables...')
        var_list = [var for var in tf.all_variables()]
        saver = tf.train.Saver(var_list)
        saver.restore(sess, "models/cnn")
        print(' * model restored ')