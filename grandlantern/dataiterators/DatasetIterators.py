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
        self.shuffle = shuffle
        self.n_batches = 0

    def fill(self, X, y):
        self.dataset.fill(X, y)
        self.n_batches = int(np.ceil(len(self.dataset) / self.batch_size))
        return self

    def batch(self, iteration):
        start = iteration * self.batch_size
        end = min((iteration + 1) * self.batch_size, len(self.dataset))
        return self.dataset[start:end]

    def __call__(self):
        if self.shuffle:
            self.dataset.shuffle()
        for it in range(self.n_batches):
            yield self.batch(it)

    def __copy__(self):
        new_dataset = copy(self.dataset)
        new_iterator = self.__class__(new_dataset, self.batch_size, self.shuffle)
        return new_iterator
