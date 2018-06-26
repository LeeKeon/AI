import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
import numpy as np

init = tf.global_variables_initializer()
sess = tf.Session(config=config)
sess.run(init)

image = np.array([[[[9], [1], [0]],
                   [[9], [9], [9]],
                   [[9], [0], [1]]]], dtype=np.float32)
print('image.shape', image.shape)
print(image.reshape(3,3))

weight = tf.constant([[[[1.]], [[0.]]],
                      [[[1.]], [[0.]]]])
print('weight.shape', weight.shape)
print(weight.eval(session=sess).reshape(2,2))

conv2d = tf.nn.conv2d(image, weight, strides=[1, 1, 1, 1], padding='VALID')
conv2d_img = conv2d.eval(session=sess)
print('conv2d_img.shape', conv2d_img.shape)
print(conv2d_img.reshape(2,2))
sess.close()