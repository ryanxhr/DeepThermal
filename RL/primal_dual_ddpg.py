import tensorflow as tf
import numpy as np
import os
from RL.ou_noise import OUNoise
from RL.reward_critic_network import RewardCriticNetwork
from RL.cost_critic_network import CostCriticNetwork

from RL.actor_network import ActorNetwork
from RL.replay_buffer import ReplayBuffer
from RL.util import *

# EPSILON定义一个极小值
EPSILON = 1e-5
# Hyper Parameters:
REPLAY_MEMORY_SIZE = 10000
REPLAY_START_SIZE = 1000
GAMMA = 0.9
COST_EPSILON = 1
DUAL_STEP_SIZE = 0.01
is_grad_inverter = False


class PrimalDualDDPG(object):
    """ Primal Dual Deep Deterministic Policy Gradient Algorithm"""

    def __init__(self, sess, input_config, is_batch_norm, summ_writer=None, load_model=False):
        self.state_dim = input_config.state_dim
        self.action_dim = input_config.action_dim
        self.dual_lambda = input_config.init_dual_lambda
        self.save_path = input_config.model_save_path
        self.train_display_iter = input_config.train_display_iter
        self.batch_size = input_config.batch_size
        self.gamma = GAMMA
        self.summay_writer = summ_writer

        self.sess = sess
        self.step = 0


        if is_batch_norm:
            self.rewward_critic_network = RewardCriticNetwork_bn(self.sess, self.state_dim, self.action_dim)
            self.cost_critic_network = CostCriticNetwork_bn(self.sess, self.state_dim, self.action_dim)
            self.actor_network = ActorNetwork_bn(self.sess, self.state_dim, self.action_dim)

        else:
            self.reward_critic_network = RewardCriticNetwork(self.sess, input_config, self.summay_writer)
            self.cost_critic_network = CostCriticNetwork(self.sess, input_config, self.summay_writer)
            self.actor_network = ActorNetwork(self.sess, input_config, load_model=False, summ_writer=self.summay_writer)

        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_MEMORY_SIZE)

        # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
        self.exploration_noise = OUNoise(self.action_dim)

        # for name in input_config.__dict__:
        #     if isinstance(input_config.__dict__[name], int) or isinstance(input_config.__dict__[name], float):
        #         self.log(f'parameter|input_config_{name}:{input_config.__dict__[name]}')

        # model saver
        self.saver = tf.train.Saver()
        if load_model:
            self.saver.restore(sess=self.sess, save_path=tf.train.latest_checkpoint(self.save_path))


    # def __del__(self):
    #     self.logfile.close()
    #
    # def log(self, *args):
    #     self.logfile.write(*args)
    #     self.logfile.write('\n')

    def train(self):
        # print "train step", self.time_step
        # Sample a random minibatch of N transitions from replay buffer
        minibatch = self.replay_buffer.get_batch(self.batch_size)
        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        cost_batch = np.asarray([data[3] for data in minibatch])
        next_state_batch = np.asarray([data[4] for data in minibatch])
        done_batch = np.asarray([data[5] for data in minibatch])

        # Calculate y_batch
        target_action_batch = self.actor_network.target_actions(next_state_batch)
        target_reward_value = self.reward_critic_network.target_reward(next_state_batch, target_action_batch)
        target_cost_value = self.cost_critic_network.target_cost(next_state_batch, target_action_batch)
        y_batch, z_batch = [], []
        for i in range(len(minibatch)):
            if done_batch[i]:
                y_batch.append(reward_batch[i])
                z_batch.append(cost_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * target_reward_value[i])
                z_batch.append(cost_batch[i] + GAMMA * target_cost_value[i])

        y_batch = np.resize(y_batch, [self.batch_size, 1])
        z_batch = np.resize(z_batch, [self.batch_size, 1])

        # Update reward critic by minimizing the loss L
        reward_critic_loss, reward_action_grad_norm = self.reward_critic_network.train(y_batch, state_batch, action_batch)
        # q_value = self.critic_network.get_q_value(state_limit_batch, action_batch)

        # Update cost critic by minimizing the loss L
        cost_critic_loss, cost_action_grad_norm = self.cost_critic_network.train(z_batch, state_batch, action_batch)

        # Update the actor policy using the sampled gradient
        if is_grad_inverter:
            action_batch_for_gradients = self.actor_network.actions(state_batch)
            action_batch_for_gradients = self.grad_inv.invert(action_batch_for_gradients, )
        else:
            action_batch_for_gradients = self.actor_network.actions(state_batch)
        print('action_batch_for_gradients', action_batch_for_gradients)
        reward_gradient_batch = self.reward_critic_network.gradients(state_batch, action_batch_for_gradients)
        cost_gradient_batch = self.cost_critic_network.gradients(state_batch, action_batch_for_gradients)
        q_gradient_batch = reward_gradient_batch - self.dual_lambda * cost_gradient_batch
        self.actor_network.train(q_gradient_batch, state_batch)

        # Update the dual variable using the sample gradient
        cost_value_batch = self.cost_critic_network.cost_value(state_batch, action_batch_for_gradients)
        cost_limit_batch = np.array([[COST_EPSILON] for _ in range(self.batch_size)])
        self.dual_gradients = np.mean(cost_value_batch - cost_limit_batch)
        self.dual_lambda += DUAL_STEP_SIZE * self.dual_gradients
        self.dual_lambda = np.max([EPSILON, self.dual_lambda])  # ensure dual >= 0

        if self.step % self.train_display_iter == 0:
            print("reward_critic: loss:{:.3f} action_grads_norm:{:.3f} "
                  "| cost_critic: loss:{:.3f} action_grads_norm:{:.3f}"
                  "| q_gradient:{:.3f}".format(
                reward_critic_loss, np.mean(reward_action_grad_norm),
                cost_critic_loss, np.mean(cost_action_grad_norm), np.mean(q_gradient_batch)))
            print("Dual lambda: {}".format(self.dual_lambda))


        # Update the target networks
        self.reward_critic_network.update_target()
        self.cost_critic_network.update_target()
        self.actor_network.update_target()
        self.step += 1

    def noise_action(self, state, episode):
        # Select action a_t according to the current policy and exploration noise
        action = self.actor_network.action(state)
        if episode % 10 == 0:
            self.exploration_noise.update_weight()
        noise_action = action + self.exploration_noise.noise()
        noise_action = np.minimum(np.maximum(noise_action, 0), 1)  # bound action to [0, 1]
        return noise_action

    def action(self, state):
        action = self.actor_network.action(state)
        return action

    def get_dual_lambda(self):
        return self.dual_lambda

    def perceive(self, state, action, reward, cost, next_state, done, mix_ratio):
        # Store transition (s_t,a_t,r_t,c_t,s_{t+1}) in replay buffer
        self.replay_buffer.add(state, action, reward, cost, next_state, done, mix_ratio)

        # Store transitions to replay start size then start training
        if self.replay_buffer.count() > REPLAY_START_SIZE:
            self.train()

        #if self.time_step % 10000 == 0:
            #self.actor_network.save_network(self.time_step)
            #self.critic_network.save_network(self.time_step)

        # Re-iniitialize the random process when an episode ends
        if done:
            self.exploration_noise.reset()

    def save_model(self):
        self.saver.save(sess=self.sess, save_path=self.save_path)  #global_step=10,会自动生成名字-10
















