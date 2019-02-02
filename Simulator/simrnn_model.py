import tensorflow as tf
import numpy as np
import os
import random
import time
from Simulator.simrnn_cell import SimulatorRNNCell


class RNNSimulatorModel(object):
    def __init__(self,
                 cell_config,
                 FLAGS):
        """ Construct simulator model using self_designed cell """
        self.coaler_cell_size, self.burner_cell_size, self.steamer_cell_size = cell_config.num_units
        self.input_size = FLAGS.input_size
        self.output_size = FLAGS.output_size
        self.coaler_output_size = cell_config.coaler_state_size
        self.burner_output_size = cell_config.burner_state_size
        self.steamer_output_size = cell_config.steamer_state_size

        self.batch_size = FLAGS.batch_size
        self.n_steps = FLAGS.num_steps
        self.l2_weight = FLAGS.l2_weight
        self.grad_clip = FLAGS.grad_clip

        # inputs.shape = (number of examples, number of input, dimension of each input).
        self.inputs = tf.placeholder(tf.float32, [None, self.n_steps, self.input_size], name="inputs")
        self.targets = tf.placeholder(tf.float32, [None, self.output_size], name="targets")
        self.learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")
        self.keep_prob = tf.placeholder(tf.float32, None, name="keep_prob")

        self.cell = SimulatorRNNCell(cell_config, self.keep_prob)
        # Run dynamic RNN
        self.cell_init_state = self.cell.zero_state(self.batch_size, dtype=tf.float32)
        cell_outputs, cell_final_state = tf.nn.dynamic_rnn(
            self.cell, self.inputs, initial_state=self.cell_init_state, time_major=False, scope="dynamic_rnn")

        # outputs.get_shape() = (batch_size, num_steps, cell_size)
        coaler_output, burner_output, steamer_output = cell_outputs
        self.coaler_output = coaler_output[:, -1, :]
        self.burner_output = burner_output[:, -1, :]
        self.steamer_output = steamer_output[:, -1, :]

        # pred = W * out + b
        ws_out_coaler = tf.Variable(
            tf.truncated_normal([self.coaler_cell_size, self.coaler_output_size]), name="W_coaler")
        bs_out_coaler = tf.Variable(
            tf.constant(0.1, shape=[self.coaler_output_size]), name="bias_coaler")
        ws_out_burner = tf.Variable(
            tf.truncated_normal([self.burner_cell_size, self.burner_output_size]), name="W_burner")
        bs_out_burner = tf.Variable(
            tf.constant(0.1, shape=[self.burner_output_size]), name="bias_burner")
        ws_out_steamer = tf.Variable(
            tf.truncated_normal([self.steamer_cell_size, self.steamer_output_size]), name="W_steamer")
        bs_out_steamer = tf.Variable(
            tf.constant(0.1, shape=[self.steamer_output_size]), name="bias_steamer")

        self.coaler_pred = tf.matmul(self.coaler_output, ws_out_coaler) + bs_out_coaler
        self.burner_pred = tf.matmul(self.burner_output, ws_out_burner) + bs_out_burner
        self.steamer_pred = tf.matmul(self.steamer_output, ws_out_steamer) + bs_out_steamer
        self.pred = tf.concat([self.coaler_pred, self.burner_pred, self.steamer_pred], axis=1)
        self.pred = tf.sigmoid(self.pred)
        # self.pred_summ = tf.summary.histogram("pred", self.pred)


        # train loss
        self.tv = tf.trainable_variables()
        self.l2_loss = self.l2_weight * tf.reduce_sum(
            [tf.nn.l2_loss(v) for v in self.tv if not ("noreg" in v.name or "bias" in v.name)], name="l2_loss")
        self.mse = tf.reduce_mean(tf.square(self.pred - self.targets), name="loss_mse_train")
        self.loss = self.mse + self.l2_loss

        # gradients clip
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, self.tv), self.grad_clip)
        # optimizer = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
        # optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_opt = optimizer.apply_gradients(zip(grads, self.tv))

        # summary
        self.loss_summ = tf.summary.scalar("loss_mse_train", self.loss)
        self.learning_rate_summ = tf.summary.scalar("learning_rate", self.learning_rate)
        # for var in tf.trainable_variables():
        #     tf.summary.histogram(var.name, var)
        self.merged_summ = tf.summary.merge_all()

