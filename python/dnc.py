import os
import tensorflow as tf

from tensorflow.contrib import rnn
from tensorflow.contrib.layers.python.layers.optimizers import optimize_loss, OPTIMIZER_SUMMARIES

from ops_tf import *
from ops_np import *
from util_data import *

class DNC(object):
    """Differential Neural Computer implementation
    in tensorflow. You can find this paper here: http://go.nature.com/2dIULo5
    """

    def __init__(self, X_train, y_train, X_test, y_test, N=256, W=64, R=2,
                    n_hidden=512, batch_size=1, disable_memory=False,
                    summary_dir=None, checkpoint_file=None,
                    optimizer="Adagrad", learning_rate=0.001,
                    clip_gradients=10.0, data_dir="./data"):

        ################
        # DNC settings #
        ################

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.N = N
        self.W = W
        self.R = R
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.disable_memory = disable_memory
        self.summary_dir = summary_dir
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.clip_gradients = clip_gradients
        self.data_dir = data_dir

        #######################
        # Controller settings #
        #######################

        (self.n_train_instances, self.n_timesteps, self.n_env_inputs) = self.X_train.shape
        (self.n_test_instances, _, _) = self.X_test.shape
        (_, self.seq_output_len, self.n_classes) = self.y_train.shape
        self.n_read_inputs = self.W*self.R
        self.n_interface_outputs = (self.W*self.R) + 3*self.W + 5*self.R + 3

        #######################
        # Tensorflow settings #
        #######################

        self.session = tf.Session()
        self.compile()
        if checkpoint_file:
            self.checkpoint_file_path = os.path.join("checkpoints", checkpoint_file)
            self.saver = tf.train.Saver()
            if os.path.exists(self.checkpoint_file_path):
                print("Restoring from checkpoint!")
                print()
                self.saver.restore(self.session, self.checkpoint_file_path)
            else:
                print("No checkpoint found! Starting from scratch...")
                print()

    def compile(self):

        self.read_keys_list = []
        self.write_keys_list = []
        self.allocation_gate_list = []
        self.free_gates_list = []
        self.write_gate_list = []
        self.preds = []
        self.losses = []
        self.accuracies = []

        # For convenience
        N = self.N
        W = self.W
        R = self.R

        with tf.variable_scope("input"):
            self.input_x = tf.placeholder(tf.float32, [self.batch_size, self.n_timesteps, self.n_env_inputs])
            self.input_y = tf.placeholder(tf.float32, [self.batch_size, self.seq_output_len, self.n_classes])

        self.memory = tf.fill([N, W], 1e-6, name="memory")

        #########################
        ## Read head variables ##
        #########################

        self.read_keys = tf.fill([R, W], 1e-6, name="read_keys_0")
        self.read_strengths = tf.fill([R], 1e-6, name="read_strengths_0")
        self.free_gates = tf.fill([R], 1e-6, name="free_gates_0")
        self.read_modes = tf.fill([R, 3], 1e-6, name="read_modes_0", )
        self.read_weightings = tf.fill([R, N], 1e-6, "read_weightings_0")

        ##########################
        ## Write head variables ##
        ##########################

        self.write_key = tf.fill([W], 1e-6, name="write_key_0")
        self.write_strength = tf.fill([], 1e-6, name="write_strength_0")
        self.write_gate = tf.fill([], 1e-6, name="write_gate_0")
        self.write_weighting = tf.fill([N], 1e-6, name="write_weighting_0")

        #####################
        ## Other variables ##
        #####################

        self.usage_vector = tf.fill([N], 1e-6, name="usage_vector_0")
        self.write_vector = tf.fill([W], 1e-6, name="write_vector_0")
        self.erase_vector = tf.fill([W], 1e-6, name="erase_vector_0")
        self.allocation_gate = tf.fill([], 1e-6, name="allocation_gate_0")
        self.linkage_matrix = tf.fill([N, N], 1e-6, name="linkage_matrix_0")
        self.precedense = tf.fill([N], 1e-6, name="precedense_0")

        lstm_cell = rnn.BasicLSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True)
        state = lstm_cell.zero_state(self.batch_size, tf.float32)

        reads_in = tf.fill([self.batch_size, self.n_read_inputs], 1e-6, name="reads_in_initial")
        self.reads = [reads_in]
        for i in range(self.n_timesteps):
            print("\rCompiling timestep {}/{} ({:.2f} %)".format(i+1, self.n_timesteps,  ((i+1)/self.n_timesteps) * 100), end="")
            with tf.variable_scope("timestep_{}".format(i)):
                reads_in_flat = tf.reshape(reads_in, [-1])
                reads_in_transformed = tf.expand_dims(reads_in_flat, 0)
                input_ = tf.concat([self.input_x[:,i,:], reads_in_transformed], 1)
                output, state = lstm_cell(input_, state)
            if self.disable_memory:
                reads_in = tf.fill([self.batch_size, self.n_read_inputs], 1e-6, name="reads_in_{}".format(i))
            else:
                if i != self.n_timesteps-1:
                    weights = tf.Variable(xavier_fill([self.n_hidden, self.n_interface_outputs]), name="interface_w_{}".format(i))
                    biases = tf.Variable(tf.fill([self.n_interface_outputs], 1e-6), name="interface_b_{}".format(i))
                    interface_out_t = tf.matmul(output, weights) + biases
                    reads_in = self.get_reads(interface_out_t, i)

            summarize_var(reads_in, name="reads_{}".format(i))
            self.reads.append(reads_in)

        print("\rCompiling loss, optimizer, predictions, etc...", end="")
        with tf.variable_scope("final"):
            for i in range(self.seq_output_len):
                weights = tf.Variable(xavier_fill([self.n_hidden, self.n_classes]), name="lstm_w_{}".format(i))
                biases = tf.Variable(tf.fill([self.n_classes], 1e-6), name="lstm_b_{}".format(i))
                pred_t = tf.matmul(output, weights) + biases
                self.preds.append(pred_t)
                loss_t = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_t, labels=self.input_y[:, i]))
                self.losses.append(loss_t)
                accuracy_t = tf.cast(tf.equal(tf.argmax(pred_t, 1), tf.argmax(self.input_y[:, i], 1)), tf.float32)
                self.accuracies.append(accuracy_t)

            self.loss_fn = tf.reduce_sum(self.losses)
            self.opt_fn = optimize_loss(self.loss_fn, None,
                                self.learning_rate, self.optimizer,
                                clip_gradients=self.clip_gradients,
                                summaries=OPTIMIZER_SUMMARIES)

            # Evaluate model
            self.accuracy_fn = tf.reduce_mean(self.accuracies)
            if self.summary_dir:
                if os.path.exists(self.summary_dir):
                    os.system("rm -rf {0} && mkdir -p {0}".format(self.summary_dir))
                self.summaries = tf.summary.merge_all()
                self.summary_writer = tf.summary.FileWriter(self.summary_dir, self.session.graph)
            else:
                self.summaries = None

        print("\rFinished compiling.                                 ")
        print()

    def parse_interface(self, interface_out, i):

        offset = 0

        # For convenience
        R = self.R
        W = self.W
        N = self.N

        result = {}

        read_keys = tf.reshape(interface_out[:,offset:R*W+offset], (R, W))
        summarize_var(read_keys, "read_keys_{}".format(i+1))
        offset += R*W
        self.read_keys_list.append(read_keys)

        read_strengths = tf.reshape(one_plus(interface_out)[:,offset:R+offset], (R,))
        summarize_var(read_strengths, "read_strengths_{}".format(i+1))
        offset += R

        write_key = tf.reshape(interface_out[:,offset:W+offset], (W,))
        summarize_var(write_key, "write_key_{}".format(i+1))
        offset += W
        self.write_keys_list.append(write_key)

        write_strength = one_plus(interface_out[:, offset:offset+1])
        summarize_var(write_strength, "write_strength_{}".format(i+1))
        offset += 1

        erase_vector = tf.reshape(sigmoid(interface_out)[:,offset:W+offset], (W,))
        summarize_var(erase_vector, "erase_vector_{}".format(i+1))
        offset += W

        write_vector = tf.reshape(interface_out[:,offset:W+offset], (W,))
        summarize_var(write_vector, "write_vector_{}".format(i+1))
        offset += W

        free_gates = tf.reshape(sigmoid(interface_out)[:,offset:R+offset], (R,))
        summarize_var(free_gates, "free_gates_{}".format(i+1))
        offset += R
        self.free_gates_list.append(free_gates)

        allocation_gate = sigmoid(interface_out[:, offset:offset+1])
        summarize_var(allocation_gate, "allocation_gate_{}".format(i+1))
        offset += 1
        self.allocation_gate_list.append(allocation_gate)

        write_gate = sigmoid(interface_out[:, offset:offset+1])
        summarize_var(write_gate, "write_gate_{}".format(i+1))
        offset += 1
        self.write_gate_list.append(write_gate)

        read_modes = tf.reshape(softmax(interface_out)[:,offset:R*3+offset], (R, 3))
        summarize_var(read_modes, "read_modes_{}".format(i+1))
        offset += R*3

        return read_keys, read_strengths, write_key, write_strength, \
               erase_vector, write_vector, free_gates, allocation_gate, \
               write_gate, read_modes

    def get_reads(self, interface_out, i):

        with tf.variable_scope("interface_ops"):

            ##################
            # Parse reads in #
            ##################

            with tf.variable_scope("parse_reads"):
                read_keys, read_strengths, write_key, write_strength, \
                erase_vector, write_vector, free_gates, allocation_gate, \
                write_gate, read_modes = self.parse_interface(interface_out, i)


            ###################
            # Content lookups #
            ###################

            with tf.variable_scope("content_lookup"):
                read_content_lookups = tf.stack([content_lookup(self.memory, K, b) for (K, b) in zip(tf.unstack(read_keys),
                                                                                                    tf.unstack(read_strengths))])

                write_content_lookup = content_lookup(self.memory, write_key, write_strength)

            ##########################
            # Update read weightings #
            ##########################

            with tf.variable_scope("read_weighting"):
                results = []
                for (r_w, c_t, pi) in zip(tf.unstack(self.read_weightings),
                                          tf.unstack(read_content_lookups),
                                          tf.unstack(read_modes)):
                    f_t = tf.reduce_sum(self.linkage_matrix * r_w, reduction_indices=[0])
                    b_t = tf.reduce_sum(tf.transpose(self.linkage_matrix) * r_w, reduction_indices=[0])
                    results.append(pi[0]*b_t + pi[1]*c_t + pi[2]*f_t)

                self.read_weightings = tf.stack(results)

            ##########################
            # Update write weighting #
            ##########################

            with tf.variable_scope("write_weighting"):

                # (1) get retention vector
                with tf.variable_scope("retention_vector"):
                    retention_vector = tf.reduce_prod(tf.stack([1-f*wr for (f, wr) in zip(tf.unstack(self.free_gates),
                                                                                          tf.unstack(self.read_weightings))]),
                                                                                          reduction_indices=[0])
                # (2) update usage vector
                with tf.variable_scope("usage_vector"):
                    self.usage_vector = tf.multiply(self.usage_vector + self.write_weighting
                                               - tf.multiply(self.usage_vector, self.write_weighting), retention_vector)

                # (3) get allocation weightings
                with tf.variable_scope("allocation_weightings"):
                    phi = tf.cast(tf_argsort(self.usage_vector)[0], tf.int32)
                    allocation_vector_list = []
                    for j in range(self.N):
                        part1 = (1 - tf.slice(self.usage_vector, [phi[j]], [1]))
                        others = []
                        for i in range(j-1):
                            var = tf.slice(self.usage_vector, [phi[i]], [1])
                            others.append(var)

                        if len(others) == 0:
                            part2 = tf.constant(0, dtype=tf.float32)
                        else:
                            part2 = tf.stack(tf.squeeze(others))
                        part3 = tf.reduce_prod(part2)
                        val = part1 + part3
                        allocation_vector_list.append(val)

                    allocation_vector = tf.squeeze(tf.stack(allocation_vector_list))

                # (4) update write weightings
                wg = self.write_gate
                ag = self.allocation_gate
                aw = allocation_vector
                c = write_content_lookup
                self.write_weighting = wg*(ag*aw + (1.0 - ag)*c)

            ##################
            # Linkage update #
            ##################

            with tf.variable_scope("linkage"):
                new_linkage_matrix = []

                # Update linkages
                for i in range(self.N):
                    this_row = []
                    for j in range(self.N):
                        if i == j:
                            this_row.append(0.0) # diagonals == 0
                        else:
                            this_row.append((1 - self.write_weighting[i] - self.write_weighting[j])*self.linkage_matrix[i][j] + self.write_weighting[i]*self.precedense[j])

                    new_linkage_matrix.append(this_row)

                self.linkage_matrix = tf.squeeze(tf.stack(new_linkage_matrix))

            #################
            # Update memory #
            #################

            with tf.variable_scope("update_memory"):
                w_t = tf.expand_dims(self.write_weighting, 1)
                et_T = tf.expand_dims(self.erase_vector, 0)
                vt_T = tf.expand_dims(self.write_vector, 0)
                self.memory = tf.multiply(self.memory, tf.ones_like(self.memory) -  tf.matmul(w_t, et_T)) + tf.matmul(w_t, vt_T)

            ################
            # Read vectors #
            ################

            with tf.variable_scope("compute_read_vector"):
                results = []
                for read_weighting in tf.unstack(self.read_weightings):
                    results.append(tf.transpose(self.memory) * read_weighting)
                read_vectors = tf.squeeze(tf.stack(results))

        return read_vectors

    def assess(self):
        accs = []
        losses = []

        for (batch_x, batch_y) in make_batches(self.X_train, self.y_train, batch_size=self.batch_size):
            acc, loss = self.session.run([self.accuracy_fn, self.loss_fn], feed_dict={self.input_x: batch_x, self.input_y: batch_y})
            accs.append(acc)
            losses.append(loss)

        return np.array(accs).mean(), np.array(losses).mean()

    def train(self, iterations=100, save_every_n_batches=750):
        self.session.run(tf.initialize_all_variables())
        print("== Trainable vars ==")
        for var in tf.trainable_variables():
            print(var.name)

        n_iter = 0

        while n_iter < iterations:
            i = 0

            ####################
            # Train on batches #
            ####################

            for (batch_x, batch_y) in make_batches(self.X_train, self.y_train, batch_size=self.batch_size):

                #############
                # Run graph #
                #############

                #self.gradient_toolkit.diagnose_grads(self.session, feed_dict={self.input_x: batch_x, self.input_y: batch_y})
                [_, loss, rks, wks, ags, fgs, wgs, summaries, reads] = self.session.run([self.opt_fn,
                                                  self.loss_fn,
                                                  self.read_keys_list,
                                                  self.write_keys_list,
                                                  self.allocation_gate_list,
                                                  self.free_gates_list,
                                                  self.write_gate_list,
                                                  self.summaries,
                                                  self.reads], feed_dict={self.input_x: batch_x, self.input_y: batch_y})
                rks = [e.tolist() for e in rks]
                wks = [e.tolist() for e in wks]
                ags = [e.tolist() for e in ags]
                fgs = [e.tolist() for e in fgs]
                wgs = [e.tolist() for e in wgs]
                write_dnc_json(self.data_dir, rks, wks, ags, fgs, wgs)

                ############
                # Printing #
                ############

                if self.summary_dir:
                    self.summary_writer.add_summary(summaries, i)
                    self.summary_writer.flush()
                percent = (i/self.n_train_instances)

                #progress_bar = "["+'='*np.floor(percent*40.0)+'>'+' '*(40.0-np.floor(percent*40.0))+"]"
                progress_bar = ""
                print("\rIteration {}: {} {}/{} ({:.2f}%) Loss: {:.5f}".format(n_iter+1, progress_bar,
                    i, self.n_train_instances, percent*100.0, loss), end="")
                i += self.batch_size

                #self.reset_memory_state(reuse=True)
                if i % save_every_n_batches == 0 and hasattr(self, "checkpoint_file_path"):
                    self.saver.save(self.session, self.checkpoint_file_path)


            ##################################
            # Assess at the end of iteration #
            ##################################

            acc, loss = self.assess()
            print("\rIteration {}: Test Loss={:.6f}, Test Accuracy={:.5f}".format(n_iter+1, loss, acc))
            n_iter += 1
