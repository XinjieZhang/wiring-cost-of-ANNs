# coding utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split


class MnistData:

    def __init__(self):
        # read MNIST data
        (train_x, train_y), (test_x, test_y) = mnist.load_data()

        # normalize data
        train_x = train_x.astype('float32') / 255.
        test_x = test_x.astype('float32') / 255.

        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

        self.train_x = train_x.reshape([-1, 28, 28])
        self.valid_x = valid_x.reshape([-1, 28, 28])
        self.test_x = test_x.reshape([-1, 28, 28])

        self.train_x = np.transpose(self.train_x, (1, 0, 2))
        self.valid_x = np.transpose(self.valid_x, (1, 0, 2))
        self.test_x = np.transpose(self.test_x, (1, 0, 2))
        self.train_y = train_y
        self.valid_y = valid_y
        self.test_y = test_y

    def iterate_train(self, batch_size=128):
        total_seqs = self.train_x.shape[1]
        permutation = np.random.permutation(total_seqs)
        total_batches = total_seqs // batch_size

        for i in range(total_batches):
            start = i * batch_size
            end = start + batch_size
            batch_x = self.train_x[:, permutation[start:end]]
            batch_y = self.train_y[permutation[start:end]]
            yield (batch_x, batch_y)


if __name__ == '__main__':

    mnist_data = MnistData()

    print('The shape of training: ', mnist_data.train_x.shape)
    print('The shape of test: ', mnist_data.test_x.shape)
    print('The shape of valid: ', mnist_data.valid_x.shape)


