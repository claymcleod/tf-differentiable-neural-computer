from tf_ops import *
import tensorflow as tf

if __name__ == "__main__":

	U = tf.constant([1.0, 0.0, 0.0], dtype=tf.float64)
	V = tf.constant([1.0, 1.0, 1.0], dtype=tf.float64)
	W = tf.constant([0.0, 0.0, 0.0], dtype=tf.float64)
	sim = cosine_similarity(U, V)

	v1 = tf.constant([1000.0, -1000.0], dtype=tf.float64)
	v2 = tf.constant([2.0, 1.0], dtype=tf.float64)

	M = tf.constant([[1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [0.8, 0.4, 0.2]], dtype=tf.float64)
	k = tf.constant([1.0, 1.0, 1.0], dtype=tf.float64)
	beta = tf.constant(20.0, dtype=tf.float64)

	with tf.Session() as sess:
		print("== Sigmoid ==")
		print("Sigmoid of {} is {}".format(v1.eval(), sigmoid(v1).eval()))
		print()
		print("== One plus ==")
		print("One_plus of {} is {}".format(v1.eval(), one_plus(v1).eval()))
		print()
		print("== Softmax ==")
		print("Softmax of {} is {}".format(v2.eval(), softmax(v2).eval()))
		print()
		print("== Cosine similarity ==")
		print("Cosine similarity of {} and {} is {:0.3f}.".format(W.eval(), V.eval(), cosine_similarity(W, V).eval()))
		print("Cosine similarity of {} and {} is {:0.3f}.".format(U.eval(), V.eval(), cosine_similarity(U, V).eval()))
		print("Cosine similarity of {} and {} is {:0.3f}.".format(V.eval(), V.eval(), cosine_similarity(V, V).eval()))
		print()
		print("== Content Lookup ==")
		print()
		print("  ## Memory ##\n\n{}\n".format(M.eval()))
		print("  ## Key ##  {}\n  ## Beta ## {}\n".format(k.eval(), beta.eval()))
		print("  ## Content ##\n\n{}".format(content_lookup(M, k, beta).eval()))
