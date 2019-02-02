from collections import deque
import random
import numpy as np
from RL.util import compute_reward, compute_cost, compute_done


class ReplayBuffer(object):
    """Using explorated data based on simulator"""
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()
        self.real_data = np.load('/Users/xhr/PycharmProjects/Boiler/Simulator/data/replay_buffer.npy')
        nums = len(self.real_data)
        self.num_indices = list(range(nums))
        random.shuffle(self.num_indices)
        self.real_start_indice = 0

    def get_batch(self, batch_size):
        # Randomly sample batch_size examples
        return random.sample(self.buffer, batch_size)

    def get_real_batch(self, batch_size):
        return self.real_data[np.random.choice(self.real_data.shape[0], batch_size, replace=False), :]

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, cost, new_state, done, mix_ratio):
        experience = (state, action, reward, cost, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            for _ in range(mix_ratio):
                s, a, s_, done = self.generate_real()
                r = compute_reward(s)
                c = compute_cost(s)
                d = compute_done(s)
                e = (s, a, r, c, s_, d)
                # print('s-{}-a{}-ns{}'.format(s.shape, a.shape, s_.shape))

                self.buffer.append(e)
            self.num_experiences += 1
        else:
            for _ in range(mix_ratio+1):
                self.buffer.popleft()
            self.buffer.append(experience)
            for _ in range(mix_ratio):
                s, a, s_, done = self.generate_real()
                r = compute_reward(s)
                c = compute_cost(s)
                d = compute_done(s)
                e = (s, a, r, c, s_, d)
                self.buffer.append(e)

    def generate_real(self):
            s = self.real_data[self.real_start_indice, :58]
            a = self.real_data[self.real_start_indice, 58:109]
            s_ = self.real_data[self.real_start_indice, 109:156]
            s_ = np.concatenate([s[:11], s_])
            done = self.real_data[self.real_start_indice, -1]
            self.real_start_indice += 1
            if self.real_start_indice == len(self.real_data):
                self.real_start_indice = 0
            return s, a, s_, done

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0




