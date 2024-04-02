
import tensorflow as tf
import time
import numpy as np

with tf.device('/device:gpu:1'):
    def make_variables(k, initializer):
      return (tf.Variable(initializer(shape=[k,k,k,k], dtype=tf.float32)),
              tf.Variable(initializer(shape=[k,k,k,k], dtype=tf.float32)))

    tensor1, tensor2 = make_variables(100, tf.random_uniform_initializer(minval=-1., maxval=1.))

    start = time.time()
    tf.multiply(tensor1, tensor2)
    end = time.time()
    print(end-start)

array1 = np.random.rand(100, 100, 100, 100)
array2 = np.random.rand(100, 100, 100, 100)
start = time.time()
np.matmul(array1, array2)
end = time.time()
print(end-start)

from tensorflow.python.keras import backend as K
K._get_available_gpus()