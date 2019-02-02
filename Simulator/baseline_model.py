import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import fully_connected


class BaseLineModel(object):
    def __init__(self,
                 FLAGS,
                 training=True):
        """ Construct baseline model, including stacked LSTM, GRU and DNN """
        self.num_units = FLAGS.num_units
        self.num_layers = FLAGS.num_layers
        self.input_size = FLAGS.input_size
        self.output_size = FLAGS.output_size

        self.batch_size = FLAGS.batch_size
        self.n_steps = FLAGS.num_steps
        self.l2_weight = FLAGS.l2_weight
        self.grad_clip = FLAGS.grad_clip

        # inputs.shape = (number of examples, number of input, dimension of each input).
        if FLAGS.model == 'dnn':
            self.inputs = tf.placeholder(tf.float32, [None, self.input_size], name="inputs")
        else:
            self.inputs = tf.placeholder(tf.float32, [None, self.n_steps, self.input_size], name="inputs")
        self.targets = tf.placeholder(tf.float32, [None, self.output_size], name="targets")
        self.learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")
        self.keep_prob = tf.placeholder(tf.float32, None, name="keep_prob")

        if training and FLAGS.keep_prob:
            self.inputs = tf.nn.dropout(self.inputs, FLAGS.keep_prob)

        if FLAGS.model == 'dnn':
            hidden = fully_connected(self.inputs, self.num_units)
            for _ in range(self.num_layers - 1):
                hidden = fully_connected(hidden, self.num_units)
                if training and FLAGS.keep_prob < 1.0:
                    hidden = rnn.DropoutWrapper(hidden,
                                                input_keep_prob=FLAGS.keep_prob,
                                                output_keep_prob=FLAGS.keep_prob)
            self.cell_outputs = hidden
        else:  # choose different rnn cell
            if FLAGS.model == 'rnn':
                cell_fn = rnn.RNNCell
            elif FLAGS.model == 'gru':
                cell_fn = rnn.GRUCell
            elif FLAGS.model == 'lstm':
                cell_fn = rnn.LSTMCell
            elif FLAGS.model == 'nas':
                cell_fn = rnn.NASCell
            else:
                raise Exception("model type not supported: {}".format(FLAGS.model))

            # warp multi layered rnn cell into one cell with dropout
            cells = []
            for _ in range(self.num_layers):
                cell = cell_fn(self.num_units)
                if training and FLAGS.keep_prob < 1.0:
                    cell = rnn.DropoutWrapper(cell,
                                              input_keep_prob=FLAGS.keep_prob,
                                              output_keep_prob=FLAGS.keep_prob)
                cells.append(cell)
            self.cell = rnn.MultiRNNCell(cells, state_is_tuple=True)

            self.cell_init_state = self.cell.zero_state(self.batch_size, dtype=tf.float32)
            cell_outputs, cell_final_state = tf.nn.dynamic_rnn(
                self.cell, self.inputs, initial_state=self.cell_init_state, time_major=False, scope="dynamic_rnn")

            # outputs.get_shape() = (batch_size, num_steps, cell_size)
            self.cell_outputs = cell_outputs[:, -1, :]

        # pred = W * out + b
        ws_out = tf.Variable(
            tf.truncated_normal([self.num_units, self.output_size]), name="W_out")
        bs_out = tf.Variable(
            tf.constant(0.1, shape=[self.output_size]), name="bias_out")
        self.pred = tf.matmul(self.cell_outputs, ws_out) + bs_out


        # train loss
        self.tv = tf.trainable_variables()
        self.l2_loss = self.l2_weight * tf.reduce_sum(
            [tf.nn.l2_loss(v) for v in self.tv if not ("noreg" in v.name or "bias" in v.name)], name="l2_loss")
        self.mse = tf.reduce_mean(tf.square(self.pred - self.targets), name="loss_mse_train")
        self.loss = self.mse + self.l2_loss

        # gradients clip
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.tv), self.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_opt = optimizer.apply_gradients(zip(grads, self.tv))

        # summary
        self.loss_summ = tf.summary.scalar("loss_mse_train", self.loss)
        self.learning_rate_summ = tf.summary.scalar("learning_rate", self.learning_rate)
        # for var in tf.trainable_variables():
        #     tf.summary.histogram(var.name, var)
        self.merged_summ = tf.summary.merge_all()

