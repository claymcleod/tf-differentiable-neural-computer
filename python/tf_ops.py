import tensorflow as tf
import numpy as np

def sigmoid(x):
    return 1 / (1 + tf.exp(-x))

def one_plus(x):
    return 1 + tf.log(1 + tf.exp(x))
    
def softmax(x):
    e_x = tf.exp(x - tf.reduce_max(x, reduction_indices=[0]))
    return e_x / tf.reduce_sum(e_x)

def cosine_similarity(a, b):
    r = tf.reduce_sum(tf.mul(a, b)) / (tf.sqrt(tf.reduce_sum(tf.square(a))) * tf.sqrt(tf.reduce_sum(tf.square(b))) + 1e-6)
    return r

def content_lookup(M, k, B):
    """Content lookup for a single read/write head.
    
    Args:
        M (tf.Tensor): Memory matrix with dimensions (N, W).
        k (tf.Tensor): Read/write key emitted by the controller
            with dimensions (W,).
        B/beta (tf.Tensor): Read/write strength emitted by the 
            controller which is an int. This represents how 
            strongly you want your head to attend to the closest 
            matching memory location (B=1 shows almost no 
            preference, B=100 will almost always single out
            the closest matching location).
            
    Returns:
        np.array: normalized probability distribution over 
            the locations in memory with dimensions (N,).
            You can think of this as the attention that the
            read head pays to each location based on the 
            content similarity.
    """

    locations = tf.map_fn(lambda x: tf.exp(cosine_similarity(x, k)*B), M)
    r = locations / tf.reduce_sum(locations) + 1e-6
    return tf.squeeze(r)

def tf_argsort(v):
    return tf.py_func(np.argsort, [v], [tf.int64])

class VariableFactory(object):
    
    def __init__(self, dtype):
        self.dtype = dtype

    def zeros(self, name, shape, **kwargs):
        return tf.get_variable(name, shape=shape, dtype=self.dtype, initializer=tf.zeros_initializer, **kwargs)

    def ones(self, name, shape, **kwargs):
        return tf.get_variable(name, shape=shape, dtype=self.dtype, initializer=tf.ones_initializer, **kwards)

    def random(self, name, shape, **kwargs):
        return tf.get_variable(name, shape=shape, dtype=self.dtype, initializer=tf.random_normal_initializer(), **kwargs)
