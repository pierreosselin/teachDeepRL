import tensorflow as tf
import numpy as np

with tf.device('/device:XLA_GPU:0'):
    x_ph = tf.placeholder(dtype=tf.float32, shape=(None,2))
    a_ph = tf.placeholder(dtype=tf.float32, shape=(2,1))

result = tf.matmul(x_ph, a_ph)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.global_variables_initializer())


print(sess.run(result, feed_dict={x_ph: np.array([[1.0, 2.0], [3.0, 4.0]]), a_ph:np.array([[1.0], [4.0]])}))
