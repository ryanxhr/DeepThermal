import tensorflow as tf
import numpy as np
import math


LAYER1_SIZE = 256
LAYER2_SIZE = 256
LEARNING_RATE = 0.0001
TAU = 0.001
L2 = 0.0001


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.03, shape=shape)
    return tf.Variable(initial)


class CostCriticNetwork(object):
    def __init__(self, sess, input_config, summ_writer):
        self.time_step = 0
        self.sess = sess
        self.state_dim = input_config.state_dim
        self.action_dim = input_config.action_dim
        self.clip_norm = input_config.clip_norm
        self.step = 0
        self.log_iter = input_config.log_iter  # logging interval in training phase
        self.log_path = input_config.log_path  # logging interval in training phase

        self.train_writer_cost = summ_writer


        # create cost network
        self.state_input, \
        self.action_input, \
        self.cost_value_output, \
        self.cost_net = self.create_cost_network(self.state_dim, self.action_dim)

        # create target cost network (the same structure with cost network)
        self.target_state_input, \
        self.target_action_input, \
        self.target_cost_value_output, \
        self.cost_target_update = self.create_target_cost_network(self.state_dim, self.action_dim, self.cost_net)

        self.create_training_method()

        self.sess.run(tf.global_variables_initializer())

        self.update_target()


    def create_training_method(self):
        # Define training optimizer
        self.z_input = tf.placeholder("float", [None, 1])
        weight_decay = tf.add_n([L2 * tf.nn.l2_loss(var) for var in self.cost_net])
        self.cost_cost = tf.reduce_mean(tf.square(self.z_input - self.cost_value_output)) + weight_decay
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost_cost)
        self.action_gradients_cost = tf.gradients(self.cost_value_output, self.action_input)



    # def create_cost_network(self, state_dim, action_dim):
    #     # the layer size could be changed
    #     layer1_size = LAYER1_SIZE
    #     layer2_size = LAYER2_SIZE
    #
    #     state_input = tf.placeholder("float", [None, state_dim])
    #     action_input = tf.placeholder("float", [None, action_dim])
    #
    #     W1 = self.variable([state_dim, layer1_size], state_dim)
    #     b1 = self.variable([layer1_size], state_dim)
    #     W2 = self.variable([layer1_size, layer2_size], layer1_size + action_dim)
    #     W2_action = self.variable([action_dim, layer2_size], layer1_size + action_dim)
    #     b2 = self.variable([layer2_size], layer1_size + action_dim)
    #     W3 = tf.Variable(tf.random_uniform([layer2_size, 1], -3e-3, 3e-3))
    #     b3 = tf.Variable(tf.random_uniform([1], -3e-3, 3e-3))
    #
    #     layer1 = tf.nn.relu(tf.matmul(state_input, W1) + b1)
    #     layer2 = tf.nn.relu(tf.matmul(layer1, W2) + tf.matmul(action_input, W2_action) + b2)
    #     cost_value_output = tf.identity(tf.matmul(layer2, W3) + b3)
    #
    #     return state_input, action_input, cost_value_output, [W1, b1, W2, W2_action, b2, W3, b3]

    def create_cost_network(self, state_dim, action_dim):
        # the layer size could be changed
        layer1_size = LAYER1_SIZE
        layer2_size = LAYER2_SIZE

        state_input = tf.placeholder("float", [None, state_dim])
        action_input = tf.placeholder("float", [None, action_dim])

        # Input -> Hidden Layer
        w1 = weight_variable([state_dim, layer1_size])
        b1 = bias_variable([layer1_size])
        # Hidden Layer -> Hidden Layer + Action
        w2 = weight_variable([layer1_size, layer2_size])
        w2a = weight_variable([action_dim, layer2_size])
        b2 = bias_variable([layer2_size])
        # Hidden Layer -> Output (Q)
        w3 = weight_variable([layer2_size, 1])
        b3 = bias_variable([1])

        # 1st Hidden layer, OPTION: Softmax, relu, tanh or sigmoid
        h1 = tf.nn.relu(tf.matmul(state_input, w1) + b1)
        # 2nd Hidden layer, OPTION: Softmax, relu, tanh or sigmoid
        # Action inserted here
        h2 = tf.nn.relu(tf.matmul(h1, w2) + tf.matmul(action_input, w2a) + b2)

        cost_value_output = tf.matmul(h2, w3) + b3

        return state_input, action_input, cost_value_output, [w1, b1, w2, w2a, b2, w3, b3]

    def create_target_cost_network(self, state_dim, action_dim, net):
        state_input = tf.placeholder("float", [None, state_dim])
        action_input = tf.placeholder("float", [None, action_dim])

        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]

        layer1 = tf.nn.relu(tf.matmul(state_input, target_net[0]) + target_net[1])
        layer2 = tf.nn.relu(tf.matmul(layer1, target_net[2]) + tf.matmul(action_input, target_net[3]) + target_net[4])
        cost_value_output = tf.identity(tf.matmul(layer2, target_net[5]) + target_net[6])

        return state_input, action_input, cost_value_output, target_update

    def update_target(self):
        self.sess.run(self.cost_target_update)

    def train(self, z_batch, state_batch, action_batch):
        # c_loss_summ = tf.summary.scalar('cost_critic_loss', self.cost_cost)
        # self.merged_cost = tf.summary.merge([c_loss_summ])

        train_feed_dict = {
            self.z_input: z_batch,
            self.state_input: state_batch,
            self.action_input: action_batch
        }
        _, cost_critic_loss, cost_action_grad_norm = \
            self.sess.run([self.optimizer, self.cost_cost, self.action_gradients_cost], train_feed_dict)

        # if self.step % self.log_iter == 0:
        #     self.train_writer_cost.add_summary(merged_summ_cost, global_step=self.step)

        self.step += 1

        return cost_critic_loss, cost_action_grad_norm

    def pretrain(self, z_batch, state_batch, action_batch):
        train_feed_dict = {
            self.z_input: z_batch,
            self.state_input: state_batch,
            self.action_input: action_batch
        }
        _, cost_critic_loss = self.sess.run([self.optimizer, self.cost_cost], train_feed_dict)
        return cost_critic_loss

    def gradients(self, state_batch, action_batch):
        return self.sess.run(self.action_gradients_cost, feed_dict={
            self.state_input: state_batch,
            self.action_input: action_batch
        })[0]

    def target_cost(self, state_batch, action_batch):
        return self.sess.run(self.target_cost_value_output, feed_dict={
            self.target_state_input: state_batch,
            self.target_action_input: action_batch
        })

    def cost_value(self, state_batch, action_batch):
        return self.sess.run(self.cost_value_output, feed_dict={
            self.state_input: state_batch,
            self.action_input: action_batch})

        # f fan-in size
    def variable(self, shape, f):
        return tf.Variable(tf.random_uniform(shape, -1 / math.sqrt(f), 1 / math.sqrt(f)))

'''
    def load_network(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_cost_critic_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print "Successfully loaded:", checkpoint.model_checkpoint_path
        else:
            print "Could not find old network weights"
    def save_network(self,time_step):
        print 'save cost-critic-network...',time_step
        self.saver.save(self.sess, 'saved_cost_critic_networks/' + 'cost-critic-network', global_step = time_step)
'''






