import tensorflow as tf
import math


# Hyper Parameters
LAYER1_SIZE = 256
LAYER2_SIZE = 256
LAYER3_SIZE = 128
LEARNING_RATE = 0.0001
TAU = 0.001


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name)


def bias_variable(shape, name):
    initial = tf.constant(0.03, shape=shape)
    return tf.Variable(initial, name)


class ActorNetwork(object):
    """ Map: state + limit_load -> action """

    def __init__(self, sess, input_config, load_model, summ_writer):
        self.sess = sess
        self.state_dim = input_config.state_dim
        self.action_dim = input_config.action_dim
        self.save_iter = input_config.save_iter  # interval of saving log
        self.save_path = input_config.model_save_path + "/actor"  # interval of saving model
        self.log_iter = input_config.log_iter  # logging interval in training phase
        self.log_path = input_config.log_path  # log path
        self.clip_norm = input_config.clip_norm
        self.step = 0

        self.train_writer = summ_writer

        # create actor network
        self.state_input, self.action_output, self.net = self.create_network(self.state_dim, self.action_dim)
        # create target actor network
        self.target_state_input, self.target_action_output, self.target_update, self.target_net = self.create_target_network(
            self.state_dim, self.action_dim, self.net)
        self.create_training_method()

        self.saver = tf.train.Saver()
        # self.saver = tf.train.Saver(tf.global_variables(scope=scope))
        if load_model:
            # restore actor network
            print('actor network restore weights')
            self.saver.restore(sess=self.sess, save_path=tf.train.latest_checkpoint(input_config.load_path))
        else:
            self.sess.run(tf.global_variables_initializer())

        self.update_target()


    def create_training_method(self):
        self.q_gradient_input = tf.placeholder("float", [None, self.action_dim])
        self.unnormalized_actor_gradients = tf.gradients(self.action_output, self.net, -self.q_gradient_input)
        # self.actor_gradients = list(map(lambda x: tf.div(x, BATCH_SIZE), self.unnormalized_actor_gradients))
        # gradients clip
        # self.actor_gradients, _ = tf.clip_by_global_norm(self.actor_gradients, clip_norm=self.clip_norm)

        # extra_ops = tf.get_collection('actor_parameters_extra_option')
        # apply_op = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.unnormalized_actor_gradients, self.net))
        apply_op = tf.train.RMSPropOptimizer(LEARNING_RATE).apply_gradients(zip(self.unnormalized_actor_gradients, self.net))

        # train_ops = [apply_op] + extra_ops
        # self.optimizer = tf.group(*apply_op)
        self.optimizer = apply_op

        diff = self.action_output - self.target_action_output
        self.mse = tf.reduce_mean(tf.square(diff))
        pretrain_grad = tf.gradients(self.mse, self.net)
        self.pretrain_update = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(
            zip(pretrain_grad, self.net))



    def create_network(self, state_dim, action_dim):
        layer1_size = LAYER1_SIZE
        layer2_size = LAYER2_SIZE

        state_input = tf.placeholder("float", [None, state_dim])

        w1 = self.variable([state_dim, layer1_size], state_dim)
        b1 = self.variable([layer1_size], state_dim)
        w2 = self.variable([layer1_size, layer2_size], layer1_size)
        b2 = self.variable([layer2_size], layer1_size)
        w3 = tf.Variable(tf.random_uniform([layer2_size, action_dim], -3e-3, 3e-3))
        b3 = tf.Variable(tf.random_uniform([action_dim], -3e-3, 3e-3))

        layer1 = tf.nn.relu(tf.matmul(state_input, w1) + b1)
        layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)
        action_output = tf.sigmoid(tf.matmul(layer2, w3) + b3)
        out_summ = tf.summary.histogram('action_output', action_output)

        w1_summ = tf.summary.histogram('W1', values=w1)
        b1_summ = tf.summary.histogram('b1', values=b1)

        w2_summ = tf.summary.histogram('W2', values=w2)
        b2_summ = tf.summary.histogram('b2', values=b2)

        w3_summ = tf.summary.histogram('W3', values=w3)
        b3_summ = tf.summary.histogram('b3', values=b3)

        self.merged_summ = tf.summary.merge([out_summ, w1_summ, b1_summ, w2_summ, b2_summ, w3_summ, b3_summ])
        # self.merged_summ = tf.summary.merge([out_summ])

        return state_input, action_output, [w1, b1, w2, b2, w3, b3]

    # def create_network(self, state_dim, action_dim):
    #     layer1_size = LAYER1_SIZE
    #     layer2_size = LAYER2_SIZE
    #
    #     state_input = tf.placeholder("float", [None, state_dim])
    #
    #     # Input -> Hidden Layer
    #     w1 = weight_variable([self.state_dim, layer1_size], 'W1')
    #     b1 = bias_variable([layer1_size], 'b1')
    #     # Hidden Layer -> Hidden Layer
    #     w2 = weight_variable([layer1_size, layer2_size], 'W2')
    #     b2 = bias_variable([layer2_size], 'b2')
    #     # Hidden Layer -> Output
    #     w3 = weight_variable([layer2_size, self.action_dim], 'W3')
    #     b3 = bias_variable([self.action_dim], 'b3')
    #
    #     # 1st Hidden layer, OPTION: Softmax, relu, tanh or sigmoid
    #     h1 = tf.nn.relu(tf.matmul(state_input, w1) + b1)
    #     # 2nd Hidden layer, OPTION: Softmax, relu, tanh or sigmoid
    #     h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
    #
    #     # Run sigmoid on output to get 0 to 1
    #     action_output = tf.nn.sigmoid(tf.matmul(h2, w3) + b3)
    #     out_summ = tf.summary.histogram('action_output', action_output)
    #
    #     w1_summ = tf.summary.histogram('W1', values=w1)
    #     b1_summ = tf.summary.histogram('b1', values=b1)
    #
    #     w2_summ = tf.summary.histogram('W2', values=w2)
    #     b2_summ = tf.summary.histogram('b2', values=b2)
    #
    #     w3_summ = tf.summary.histogram('W3', values=w3)
    #     b3_summ = tf.summary.histogram('b3', values=b3)
    #
    #     self.merged_summ = tf.summary.merge([out_summ, w1_summ, b1_summ, w2_summ, b2_summ, w3_summ, b3_summ])
    #     # self.merged_summ = tf.summary.merge([out_summ])
    #
    #     # scaled_out = tf.multiply(out, self.action_bound)  # Scale output to -action_bound to action_bound
    #
    #     return state_input, action_output, [w1, b1, w2, b2, w3, b3]


    def create_target_network(self, state_dim, action_dim, net):
        state_input = tf.placeholder("float", [None, state_dim])
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)
        target_update = ema.apply(net)
        target_net = [ema.average(x) for x in net]

        layer1 = tf.nn.relu(tf.matmul(state_input, target_net[0]) + target_net[1])
        layer2 = tf.nn.relu(tf.matmul(layer1, target_net[2]) + target_net[3])

        action_output = tf.tanh(tf.matmul(layer2, target_net[4]) + target_net[5])

        return state_input, action_output, target_update, target_net

    def update_target(self):
        self.sess.run(self.target_update)

    def train(self, q_gradient_batch, state_batch):
        train_feed_dict = {
            self.q_gradient_input: q_gradient_batch,
            self.state_input: state_batch
        }
        summ, _ = self.sess.run([self.merged_summ, self.optimizer], feed_dict=train_feed_dict)
        # _ = self.sess.run([self.optimizer], feed_dict=train_feed_dict)

        # save actor network
        if self.step % self.save_iter == 0:
            self.saver.save(self.sess, save_path=self.save_path, global_step=self.step)

        if self.step % self.log_iter == 0:
            self.train_writer.add_summary(summ, global_step=self.step)

        self.step += 1

    def pretrain(self, state, label):
        # cost
        train_feed_dict = {self.state_input: state, self.target_action_output: label}
        _, net, mse = self.sess.run([self.pretrain_update, self.net, self.mse], feed_dict=train_feed_dict)
        # save actor network
        if self.step % self.save_iter == 0:
            self.saver.save(self.sess, save_path=self.save_path, global_step=self.step)

        self.step += 1
        return net, mse

    def actions(self, state_batch):
        return self.sess.run(self.action_output, feed_dict={
            self.state_input: state_batch
        })

    def action(self, state):
        return self.sess.run(self.action_output, feed_dict={
            self.state_input: [state]
        })[0]

    def target_actions(self, state_batch):
        return self.sess.run(self.target_action_output, feed_dict={
            self.target_state_input: state_batch
        })

        # f fan-in size
    def variable(self, shape, f):
        return tf.Variable(tf.random_uniform(shape, -1 / math.sqrt(f), 1 / math.sqrt(f)))

    def save_network(self, episode):
        print('save actor-network...', episode)
        self.saver.save(self.sess, 'saved_actor_networks/' + 'actor-network', global_step=episode)

'''
    def load_network(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_actor_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print "Successfully loaded:", checkpoint.model_checkpoint_path
        else:
            print "Could not find old network weights"

'''

