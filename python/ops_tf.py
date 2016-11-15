import numpy as np
import tensorflow as tf

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


class GradientToolkit(object):

    def __init__(self, optimizer_fn, loss_fn):

        self.optimizer_fn = optimizer_fn
        self.loss_fn = loss_fn
        grads_and_vars = optimizer_fn.compute_gradients(loss_fn)
        self.filtered_grads_and_vars = []
        self.filtered_vars = []

        warn = False
        for (grad, var) in grads_and_vars:
            if var is None or grad is None:
                warn = True
                print(var.name, "=", grad)
            else:
                self.filtered_grads_and_vars.append((tf.clip_by_value(grad, -1.0, 1.0), var))
                self.filtered_vars.append(var.name)

        if warn:
            import warnings
            warnings.warn("All of the above variables probably are causing problems " +
                "in your graph. You should probably quit now and check these out. However, " +
                "I will continue for now.")

        self.apply_grads = self.optimizer_fn.apply_gradients(self.filtered_grads_and_vars)

    def diagnose_grads(self, session, feed_dict):
        grads = session.run([self.filtered_grads_and_vars,
                             self.apply_grads], feed_dict=feed_dict)
        for (grad, var) in zip(grads, self.filtered_vars):
            if grad and np.any(np.isnan(grad[0])):
                print("Looks like", var, "has a NaN gradient!")
