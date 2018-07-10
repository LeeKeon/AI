import tensorflow as tf
from tensorflow.python.layers.core import Dense

# Write your code here
class Model(object):
    def __init__(self, use_clip=True, class_size=24, learning_rate=0.01):
        self.initializer = tf.random_uniform_initializer(-0.1, 0.1)
        self.use_clip = use_clip
        self.class_size = class_size
        self.learning_rate = learning_rate

        self.x_image = tf.placeholder(dtype=tf.float32, shape=(None, 134, 91, 3))
        self.y_label = tf.placeholder(dtype=tf.int32, shape=(None, ))
        self.keep_prob = tf.placeholder_with_default(1.0, shape=None)

        self.y_float = tf.cast(self.y_label, tf.float32)

        self.build_model()
        self.build_loss()
        self.build_opt()
        self.build_acc()

    def build_model(self):
        conv1 = self.conv(name = 'c1', inputs = self.x_image, shape=[3,3,3,256], s = 1, padding = 'SAME')
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        conv2 = self.conv(name = 'c2', inputs = conv1, shape=[3,3,256,128], s = 1, padding = 'SAME')
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        conv2 = tf.nn.dropout(conv2, self.keep_prob)

        conv3 = self.conv(name = 'c3', inputs = conv2, shape=[3,3,128,128], s = 1, padding = 'SAME')
        conv3 = tf.nn.relu(conv3)
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        conv3 = tf.nn.dropout(conv3, self.keep_prob)

        conv4 = self.conv(name = 'c4', inputs = conv3, shape=[3,3,128,64], s = 1, padding = 'SAME')
        conv4 = tf.nn.relu(conv4)
        conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
        conv4 = tf.nn.dropout(conv4, self.keep_prob)

        _, w, h, d = conv4.get_shape()
        length = int(w) * int(h) * int(d)
        w1 = self.get_var(name='w1', shape=[length, 256])
        b1 = self.get_var(name='b1', shape=[256])
        l1 = tf.reshape(conv4, [-1, length])
        l2 = tf.nn.relu(tf.matmul(l1, w1) + b1)
        l2 = tf.nn.dropout(l2, self.keep_prob)

        w2 = self.get_var(name='w2', shape=[256, 128])
        b2 = self.get_var(name='b2', shape=[128])
        l3 = tf.nn.relu(tf.matmul(l2, w2) + b2)
        l3 = tf.nn.dropout(l3, self.keep_prob)

        w3 = self.get_var(name='w3', shape=[128, 4])
        b3 = self.get_var(name='b3', shape=[4])
        self.genre_prob = tf.nn.softmax(tf.matmul(l3, w3) + b3)

        #최종 결과
        self.y_pred = tf.argmax(self.genre_prob, 1)

    def build_loss(self):
        self.o1 = tf.reshape(self.y_label, [-1])
        self.o2 = tf.to_int32(self.o1)
        self.chk = tf.one_hot(self.o2, self.class_size, 1.0, 0.0)
        
        self.cross_entropy = -tf.reduce_sum(self.chk
             * tf.log(tf.clip_by_value(tf.reshape(self.genre_prob, [-1, self.class_size]), 1e-20, 1.0)), 1)
                                            
#         self.cross_entropy = -tf.reduce_sum(
#             self.y_float
#             * tf.log(tf.clip_by_value(tf.reshape(self.genre_prob, [-1, self.class_size]), 1e-20, 1.0)), 1)
        self.loss = tf.reduce_mean(self.cross_entropy)

        # self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y_float)
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
    
    def build_acc(self):
        self.prob = tf.to_int32(tf.argmax(self.genre_prob, 1))
        self.is_correct = tf.equal(self.prob, self.y_label)
        self.accuracy = tf.reduce_mean(tf.cast(self.is_correct, tf.float32))

    def get_var(self, name='', shape=None, dtype=tf.float32):
        return tf.get_variable(name, shape, dtype=dtype, initializer=self.initializer)

    def conv(self, name='', inputs=None, shape=[], s=None, padding='SAME'):
        w = self.get_var(name='w'+name, shape=shape)
        b = self.get_var(name='b'+name, shape=shape[-1])
        return tf.nn.conv2d(inputs, filter=w, strides=[1, s, s, 1], padding=padding) + b

    def save(self, sess, global_step=None):
        var_list = [var for var in tf.all_variables()]
        saver = tf.train.Saver(var_list)
        save_path = saver.save(sess, save_path="models/model", global_step=global_step)
        print(' * model saved at \'{}\''.format(save_path))

    # Load whole weights
    def restore(self, sess):
        print(' - Restoring variables...')
        var_list = [var for var in tf.all_variables()]
        saver = tf.train.Saver(var_list)
        saver.restore(sess, "models/model")
        print(' * model restored ')
