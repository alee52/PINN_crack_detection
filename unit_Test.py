import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import matplotlib.pyplot as plt

def sample_uniform_points_in_rectangle_tf_simple(n, domain_bottom_left0, domain_top_right0, dtype=tf.float32):
    """
    Sample n points uniformly inside a rectangle using TensorFlow,
    without manually rescaling from [0,1].
    """
    x_min, y_min = domain_bottom_left0
    x_max, y_max = domain_top_right0

    xs = tf.random.uniform(shape=(n,), minval=x_min, maxval=x_max, dtype=dtype)
    ys = tf.random.uniform(shape=(n,), minval=y_min, maxval=y_max, dtype=dtype)

    return tf.stack([xs, ys], axis=1)

# Example
sam = sample_uniform_points_in_rectangle_tf_simple(1000, [-1.0, -1.0], [1.0, 1.0])

plt.scatter(sam[:, 0].numpy(), sam[:, 1].numpy(), s=6)
plt.gca().set_aspect('equal', adjustable='box')
plt.show()