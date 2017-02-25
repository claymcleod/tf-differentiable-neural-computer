import numpy as np
import tensorflow as tf

def sigmoid(x):
    """Sigmoid function to gate functions between [-1.0, 1.0].

    Args:
        x (tf.Tensor): input to gate.

    Returns:
       tf.Tensor: sigmoid transformed tensor.
    """
    return 1 / (1 + tf.exp(-x))

def one_plus(x):
    """One plus function to gate between [1, infinity).

    Args:
        x (tf.Tensor): input to gate.

    Returns:
       tf.Tensor: one_plus transformed tensor.
    """

    return 1 + tf.log(1 + tf.exp(x))

def softmax(x):
    """Softmax that normalizes the output to sum to 1.

    Args:
        x (tf.Tensor): input to gate.

    Returns:
       tf.Tensor: softmax transformed tensor.
    """

    e_x = tf.exp(x - tf.reduce_max(x, reduction_indices=[0]))
    return e_x / tf.reduce_sum(e_x)

def cosine_similarity(a, b):
    """Similarity function which returns 1 when
    a == b and 0 when a is completely oppose b.

    Args:
        a (tf.Tensor): input vector.
        b (tf.Tensor): input vector.

    Returns:
       tf.Tensor: cosine similarity transformed tensor.
    """

    return tf.reduce_sum(tf.multiply(a, b)) / (tf.sqrt(tf.reduce_sum(tf.square(a))) * tf.sqrt(tf.reduce_sum(tf.square(b))) + 1e-6)

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
    """Function to wrap numpy's argsort function.

    Args:
        v (tf.Tensor): tensor to sort.

    Returns:
        tf.Tensor: sorted indices of v.
    """

    return tf.py_func(np.argsort, [v], [tf.int64])

def xavier_fill(shape, name=None):
    [fan_in, fan_out] = shape
    low = -1*np.sqrt(6.0/(fan_in + fan_out))
    high = 1*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform(shape, minval=low, maxval=high, name=name)

def summarize_var(var, name):
    pass
    #tf.histogram_summary(name, var)
