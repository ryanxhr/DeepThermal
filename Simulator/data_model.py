import numpy as np
import os
import pandas as pd
import random


class BoilerDataSet(object):
    """
    first run data_preparation.py to generate data.csv
    prepare boiler training and validation dataset
    simple version(small action dimension)

    """
    def __init__(self, num_steps, val_ratio=0.1):
        self.num_steps = num_steps
        self.val_ratio = val_ratio

        # Read csv file
        self.raw_seq = pd.read_csv(os.path.join("data", "sim_train.csv"), index_col='date')
        self.train_X, self.train_y, self.val_X, self.val_y = self._prepare_data(self.raw_seq)

    def _prepare_data(self, seq):
        # split into groups of num_steps
        X = np.array([seq.iloc[i: i + self.num_steps].values
                      for i in range(len(seq) - self.num_steps)])
        y = np.array([seq.ix[i + self.num_steps, 'A磨煤机料位':'1号机组下部水冷壁出口平均壁温'].values
                      for i in range(len(seq) - self.num_steps)])

        train_size = int(len(X) * (1.0 - self.val_ratio))
        train_X, val_X = X[:train_size], X[train_size:]
        train_y, val_y = y[:train_size], y[train_size:]
        return train_X, train_y, val_X, val_y

    def generate_one_epoch(self, data_X, data_y, batch_size):
        num_batches = int(len(data_X)) // batch_size
        # if batch_size * num_batches < len(self.train_X):
        #     num_batches += 1

        batch_indices = list(range(num_batches))
        random.shuffle(batch_indices)
        for j in batch_indices:
            batch_X = data_X[j * batch_size: (j + 1) * batch_size]
            batch_y = data_y[j * batch_size: (j + 1) * batch_size]
            yield batch_X, batch_y

