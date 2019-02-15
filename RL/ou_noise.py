# --------------------------------------
# Ornstein-Uhlenbeck Noise
# Author: Flood Sung
# Date: 2016.5.4
# Reference: https://github.com/rllab/rllab/blob/master/rllab/exploration_strategies/ou_strategy.py
# --------------------------------------

import numpy as np
import numpy.random as nr


class OUNoise:
    """docstring for OUNoise"""
    def __init__(self, action_dimension, mu=0.5, theta=0.4, sigma=0.2, weight_decay=0.9999):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.weight = 1
        self.weight_decay = weight_decay
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def update_weight(self):
        self.weight *= self.weight_decay

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * nr.randn(len(x))
        self.state = x + dx * self.weight
        return self.state



# # Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
# class OrnsteinUhlenbeckActionNoise(ActionNoise):
#     def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None, weight_decay_factor=0.999):
#         self.theta = theta
#         self.mu = mu
#         self.sigma = sigma
#         self.dt = dt
#         self.x0 = x0
#         self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
#
#         self.weight_decay_factor = weight_decay_factor
#         self.weight_decay = 1
#
#         self.reset()
#
#     def get_noise(self):
#         x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(
#             self.dt) * np.random.normal(size=self.mu.shape)
#         self.x_prev = x
#         return x
#
#     @property
#     def shape(self):
#         return self.mu.shape
#
#     def reset(self):
#         self.weight_decay = 1
#
#     def noise_decay(self):
#         self.weight_decay *= self.weight_decay_factor
#
#     def __call__(self, action):
#         r = action + self.get_noise() * self.weight_decay
#         return r
#
#     def __repr__(self):
#         return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={}, weight_decay_factor=)'.format(self.mu, self.sigma,
#                                                                                             self.weight_decay_factor)


if __name__ == '__main__':
    ou = OUNoise(3)
    states = []
    for i in range(10000):
        ou.update_weight()
        states.append(ou.noise())
    import matplotlib.pyplot as plt

    plt.plot(states)
    plt.show()