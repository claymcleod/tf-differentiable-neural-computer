import os
import tensorflow as tf

from tensorflow.python.ops import rnn, rnn_cell

from ops_tf import *
from ops_np import *
from sklearn.model_selection import train_test_split

class DNC(object):
    """Differential Neural Computer implementation
    in tensorflow. You can find this paper here: http://go.nature.com/2dIULo5
    """
    
    def __init__(self, X, y, validation_split=0.25, N=256, W=64, R=2, n_hidden=512, batch_size=1,
                disable_memory=False, dtype=tf.float32, summary_dir=None, checkpoint_file=None):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_split)
        
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
        self.dtype = dtype
        self.summary_dir = summary_dir
        self.global_step = tf.Variable(0, name='global_step', trainable=False).initialized_value()
                
        #######################
        # Controller settings #
        #######################

        (self.n_train_instances, self.n_timesteps, self.n_env_inputs) = self.X_train.shape
        (self.n_test_instances, _, _) = self.X_test.shape
        (_, self.n_classes) = self.y_train.shape
        self.n_read_inputs = self.W*self.R
        self.n_interface_outputs = (self.W*self.R) + 3*self.W + 5*self.R + 3
        
        #######################
        # Tensorflow settings #
        #######################

        self.var_factory = VariableFactory(dtype)
        self.session = tf.Session()
        self.reset_memory_state()
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
            
        
    def reset_memory_state(self, reuse=False):
        """Reset the memory state after each training iteration"""
        
        # Just for convenience
        N = self.N
        W = self.W
        R = self.R
        
        with tf.variable_scope("memory_state", reuse=reuse):

            self.memory = self.var_factory.zeros("memory", [N, W])
            
            #########################
            ## Read head variables ##
            #########################
            
            self.read_keys = self.var_factory.zeros("read_keys_0", [R, W])
            self.read_strengths = self.var_factory.zeros("read_strengths_0", [R])
            self.free_gates = self.var_factory.zeros("free_gates_0", [R])
            self.read_modes = self.var_factory.zeros("read_modes_0", [R, 3])
            self.read_weightings = self.var_factory.zeros("read_weightings_0", [R, N])
            
            ##########################
            ## Write head variables ##
            ##########################
            
            self.write_key = self.var_factory.zeros("write_key_0", [W])
            self.write_strength = self.var_factory.zeros("write_strength_0", [])
            self.write_gate = self.var_factory.zeros("write_gate_0", [])
            self.write_weighting = self.var_factory.zeros("write_weighting_0", [N])
            
            #####################
            ## Other variables ##
            #####################
            
            self.usage_vector = self.var_factory.zeros("usage_vector_0", [N])
            self.write_vector = self.var_factory.zeros("write_vector_0", [W])
            self.erase_vector = self.var_factory.zeros("erase_vector_0", [W])
            self.allocation_gate = self.var_factory.zeros("allocation_gate_0", [])
            self.linkage_matrix = self.var_factory.zeros("linkage_matrix_0", [N, N])
            self.precedense = self.var_factory.zeros("precedense_0", [N])

    def compile(self):

        with tf.variable_scope("input"):
            self.input_x = tf.placeholder(self.dtype, [None, self.n_timesteps, self.n_env_inputs])
            self.input_y = tf.placeholder(self.dtype, [None, self.n_classes])

        with tf.variable_scope("lstm"):
            weights = self.var_factory.random("lstm_weights", [self.n_hidden, self.n_classes])
            biases = self.var_factory.zeros("lstm_biases", [self.n_classes])
        
            lstm_cell = rnn_cell.BasicLSTMCell(self.n_hidden, forget_bias=1.0, state_is_tuple=True)
            state = lstm_cell.zero_state(self.batch_size, self.dtype)

        reads_in = self.var_factory.zeros("reads_in_initial", [self.batch_size, self.n_read_inputs], trainable=False)
        self.reads = [reads_in]
        for i in range(self.n_timesteps):
            print("\rCompiling timestep {}/{} ({:.2f} %)".format(i+1, self.n_timesteps,  ((i+1)/self.n_timesteps) * 100), end="")
            with tf.variable_scope("LSTM_{}".format(i)):
                input_ = tf.concat(1, [self.input_x[:,i,:], tf.expand_dims(tf.reshape(reads_in, [-1]), 0)])
                output, state = lstm_cell(input_, state)
            if self.disable_memory:
                reads_in = self.var_factory.zeros("reads_in_{}".format(i), [self.batch_size, self.n_read_inputs], trainable=False)
            else:
                reads_in = self.get_reads(output, i)

            self.reads.append(reads_in)

        print("\rCompiling loss, optimizer, predictions, etc...", end="")
        with tf.variable_scope("final"):
            self.pred_fn = tf.matmul(output, weights) + biases
            self.loss_fn = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.pred_fn, self.input_y))
        
            #self.gradient_toolkit = GradientToolkit(tf.train.AdagradOptimizer(0.01), self.loss_fn)
            self.opt_fn = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(self.loss_fn)#self.gradient_toolkit.apply_grads
                    
            # Evaluate model
            self.correct_pred_fn = tf.equal(tf.argmax(self.pred_fn, 1), tf.argmax(self.input_y, 1))
            self.accuracy_fn = tf.reduce_mean(tf.cast(self.correct_pred_fn, self.dtype))
            if self.summary_dir:
                self.summaries = tf.merge_all_summaries()
                self.summary_writer = tf.train.SummaryWriter(self.summary_dir, self.session.graph)
            else:
                self.summaries = None

        print("\rFinished compiling.                                 ")
        print()

    def parse_interface(self, interface_out):

        offset = 0 
        
        # For convenience
        R = self.R
        W = self.W
        N = self.N
        
        result = {}
        
        read_keys = tf.reshape(interface_out[:,offset:R*W+offset], (R, W))
        offset += R*W
        
        read_strengths = tf.reshape(one_plus(interface_out)[:,offset:R+offset], (R,))
        offset += R
        
        write_key = tf.reshape(interface_out[:,offset:W+offset], (W,))
        offset += W
        
        write_strength = one_plus(interface_out[:, offset:offset+1])
        offset += 1
        
        erase_vector = tf.reshape(sigmoid(interface_out)[:,offset:W+offset], (W,))
        offset += W
        
        write_vector = tf.reshape(interface_out[:,offset:W+offset], (W,))
        offset += W
        
        free_gates = tf.reshape(sigmoid(interface_out)[:,offset:R+offset], (R,))
        offset += R
        
        allocation_gate = sigmoid(interface_out[:, offset:offset+1])
        offset += 1
        
        write_gate = sigmoid(interface_out[:, offset:offset+1])
        offset += 1
        
        read_modes = tf.reshape(softmax(interface_out)[:,offset:R*3+offset], (R, 3))
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
                write_gate, read_modes = self.parse_interface(interface_out)


            ###################
            # Content lookups #
            ###################
            
            with tf.variable_scope("content_lookup"):
                read_content_lookups = tf.pack([content_lookup(self.memory, K, b) for (K, b) in zip(tf.unpack(read_keys), 
                                                                                                    tf.unpack(read_strengths))])
                
                write_content_lookup = content_lookup(self.memory, write_key, write_strength) 

            ##########################
            # Update read weightings #
            ##########################
            
            with tf.variable_scope("read_weighting"):
                results = []
                for (r_w, c_t, pi) in zip(tf.unpack(self.read_weightings), 
                                          tf.unpack(read_content_lookups), 
                                          tf.unpack(read_modes)):
                    f_t = tf.reduce_sum(self.linkage_matrix * r_w, reduction_indices=[0])
                    b_t = tf.reduce_sum(tf.transpose(self.linkage_matrix) * r_w, reduction_indices=[0])
                    results.append(pi[0]*b_t + pi[1]*c_t + pi[2]*f_t)
                        
                self.read_weightings = tf.pack(results)
            
            ##########################
            # Update write weighting #
            ##########################
            
            with tf.variable_scope("write_weighting"):

                # (1) get retention vector
                with tf.variable_scope("retention_vector"):
                    retention_vector = tf.reduce_prod(tf.pack([1-f*wr for (f, wr) in zip(tf.unpack(self.free_gates), 
                                                                                          tf.unpack(self.read_weightings))]),
                                                                                          reduction_indices=[0])
                # (2) update usage vector
                with tf.variable_scope("usage_vector"):
                    self.usage_vector = tf.mul(self.usage_vector + self.write_weighting 
                                               - tf.mul(self.usage_vector, self.write_weighting), retention_vector)
                
                # (3) get allocation weightings
                with tf.variable_scope("allocation_weightings"):
                    if self.dtype == tf.float32:
                        phi = tf.cast(tf_argsort(self.usage_vector)[0], tf.int32)
                    else:
                        phi = tf.cast(tf_argsort(self.usage_vector)[0], tf.int64)
                        
                    allocation_vector_list = []
                    for j in range(self.N):
                        part1 = (1 - tf.slice(self.usage_vector, [phi[j]], [1]))
                        others = []
                        for i in range(j-1):
                            var = tf.slice(self.usage_vector, [phi[i]], [1])
                            others.append(var)

                        if len(others) == 0:
                            part2 = tf.constant(0, dtype=self.dtype)
                        else:
                            part2 = tf.pack(tf.squeeze(others))
                        part3 = tf.reduce_prod(part2)
                        val = part1 + part3
                        allocation_vector_list.append(val)
                    
                    allocation_vector = tf.squeeze(tf.pack(allocation_vector_list))

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
                
                self.linkage_matrix = tf.squeeze(tf.pack(new_linkage_matrix))
            
            #################
            # Update memory #
            #################
            
            with tf.variable_scope("update_memory"):
                w_t = tf.expand_dims(self.write_weighting, 1)
                et_T = tf.expand_dims(self.erase_vector, 0)
                vt_T = tf.expand_dims(self.write_vector, 0)
                self.memory = tf.mul(self.memory, tf.ones_like(self.memory) -  tf.matmul(w_t, et_T)) + tf.matmul(w_t, vt_T)

            ################
            # Read vectors #
            ################
            
            with tf.variable_scope("compute_read_vector"):
                results = []
                for read_weighting in tf.unpack(self.read_weightings):
                    results.append(tf.transpose(self.memory) * read_weighting)
                read_vectors = tf.squeeze(tf.pack(results))
        
        return read_vectors

    def assess(self):
        accs = []
        losses = []
        
        for (batch_x, batch_y) in make_batches(self.X_train, self.y_train, batch_size=self.batch_size):
            acc, loss = self.session.run([self.accuracy_fn, self.loss_fn], feed_dict={self.input_x: batch_x, self.input_y: batch_y})
            accs.append(acc)
            losses.append(loss)
            
        return np.array(accs).mean(), np.array(losses).mean()

    def train(self, iterations=100, save_every_n_batches=200):
        self.session.run(tf.initialize_all_variables())

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
                [_, loss, r_] = self.session.run([self.opt_fn, self.loss_fn, self.reads], feed_dict={self.input_x: batch_x, self.input_y: batch_y})

                ############
                # Printing #
                ############

                #if self.summary_dir:
                #    self.summary_writer.add_summary(summaries, i)
                percent = (i/self.n_train_instances)

                #progress_bar = "["+'='*np.floor(percent*40.0)+'>'+' '*(40.0-np.floor(percent*40.0))+"]"
                progress_bar = ""
                print("\rIteration {}: {} {}/{} ({:.2f}%) Loss: {:.5f}".format(n_iter+1, progress_bar, 
                    i, self.n_train_instances, percent*100.0, loss), end="")
                i += self.batch_size

                self.reset_memory_state(reuse=True)
                if i % save_every_n_batches == 0 and hasattr(self, "checkpoint_file_path"):
                    self.saver.save(self.session, self.checkpoint_file_path)
                    print("Saving checkpoint")
                

            ##############################
            # Assess at the end of batch #
            ##############################

            acc, loss = self.assess()
            print("\rIteration {}: Test Loss={:.6f}, Test Accuracy={:.5f}".format(n_iter+1, loss, acc))
            n_iter += 1
