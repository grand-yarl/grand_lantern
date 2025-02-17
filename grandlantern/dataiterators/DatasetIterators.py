import numpy as np
from .Datasets import *
from copy import copy


class DatasetIterator:
    dataset: Dataset
    batch_size: int
    n_batches: int
    shuffle: bool

    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_batches = 0
        self.shuffle = shuffle

    def __copy__(self):
        cls = self.__class__
        new_dataset = copy(self.dataset)
        new_iterator = cls(new_dataset, batch_size=self.batch_size)
        return new_iterator

    def fill(self, X, y):
        self.dataset.fill(X, y)
        self.n_batches = int(np.ceil(len(self.dataset) / self.batch_size))
        return self

    def batch(self, iteration):
        if iteration == self.n_batches:
            batch_slice = slice(iteration * self.batch_size, len(self.dataset))
        else:
            batch_slice = slice(iteration * self.batch_size, (iteration + 1) * self.batch_size)
        X_batch, y_batch = self.dataset[batch_slice]
        return X_batch, y_batch

    def __call__(self):
        if self.shuffle:
            self.dataset.shuffle()
        for it in range(self.n_batches):
            X_batch, y_batch = self.batch(it)
            yield X_batch, y_batch
