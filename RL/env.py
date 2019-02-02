import tensorflow as tf
import numpy as np
from collections import deque
import random

from Simulator.simrnn_model import RNNSimulatorModel
from Simulator.simrnn_main import cell_config, FLAGS
from RL.util import *

OUTER_START_POS = 0
OUTER_SIZE = 11
STATE_SIZE = 47
ACTION_SIZE = 51
STATE_START_POS = OUTER_START_POS + OUTER_SIZE
ACTION_START_POS = STATE_START_POS + STATE_SIZE
NEW_STATE_START_POS = ACTION_START_POS + ACTION_SIZE


class SimulatorEnvironment(object):
    def __init__(self, sess):
        self.sess = sess
        self.replay_buffer = np.load('../Simulator/data/replay_buffer.npy')
        self.state_buffer = deque()

        # model construction
        self.rnn_model = RNNSimulatorModel(cell_config(), FLAGS)

        self.sess.run(tf.global_variables_initializer())

        # path
        model_name = "sim_rnn"
        model_path = '../Simulator/logs/{}-{}-{}-{}-{}-{:.2f}-{:.4f}-{:.2f}-{:.5f}/'.format(
            model_name, cell_config.num_units[0], cell_config.num_units[1], cell_config.num_units[2],
            FLAGS.num_steps, FLAGS.keep_prob, FLAGS.learning_rate, FLAGS.learning_rate_decay, FLAGS.l2_weight)
        model_path += 'saved_models/final_model.ckpt'

        saver = tf.train.Saver()
        saver.restore(self.sess, model_path)
        print("Model successfully restored from file: %s" % model_path)

    def reset(self):
        """ Resets the state of the environment and returns an initial observation. """
        self.state_buffer = deque()
        nums = len(self.replay_buffer)
        init_state_indice = random.randint(10, nums)
        for i in range(10):
            self.state_buffer.append(self.replay_buffer[init_state_indice-(9-i), :NEW_STATE_START_POS])
        self.new_state = init_state = self.replay_buffer[init_state_indice, :ACTION_START_POS]
        # self.new_state = init_state.reshape(1, -1)
        self.outer_state = self.replay_buffer[init_state_indice, OUTER_START_POS:STATE_START_POS]

        return self.new_state

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, cost, done, info).
        """
        self.state_buffer.append(np.concatenate([self.new_state, action]))
        self.state_buffer.popleft()

        # transpose from 2D to 3D
        model_inputs_2D = np.array(self.state_buffer)
        num_step, dim = model_inputs_2D.shape
        model_inputs_3D = model_inputs_2D.reshape(1, num_step, dim)

        test_data_feed = {
            self.rnn_model.keep_prob: 1.0,
            self.rnn_model.inputs: model_inputs_3D,
        }
        new_state = self.sess.run(self.rnn_model.pred, test_data_feed)
        self.new_state = np.concatenate([self.outer_state, new_state[0]])  # (1, 47) -> (47, )

        reward = compute_reward(self.new_state)
        cost = compute_cost(self.new_state)
        done = compute_done(self.new_state)

        return self.new_state, reward, cost, done




