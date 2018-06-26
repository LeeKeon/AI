import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
import numpy as np

init = tf.global_variables_initializer()
sess = tf.Session(config=config)
sess.run(init)

image = np.array([[[[1], [2], [3]],
                   [[4], [5], [6]],
                   [[7], [8], [9]]]], dtype=np.float32)
print('image.shape', image.shape)
print(image.reshape(3,3))

pool = tf.nn.max_pool(image, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='VALID')
pool_img = pool.eval(session=sess)
print('pool_img.shape', pool.shape)
print(pool_img.reshape(2,2))
sess.close()