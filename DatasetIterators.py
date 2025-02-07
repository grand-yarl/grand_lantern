from Datasets import *
from copy import copy


class DatasetIterator:
    dataset: Dataset
    batch_size: int
    n_batches: int

    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_batches = 0

    def __copy__(self):
        cls = self.__class__
        new_dataset = copy(self.dataset)
        new_iterator = cls(new_dataset, batch_size=self.batch_size)
        return new_iterator

    def fill(self, X, y):
        self.dataset.fill(Matrix(X), Matrix(y))
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
        for it in range(self.n_batches):
            X_batch, y_batch = self.batch(it)
            yield X_batch, y_batch


class SequenceIterator(DatasetIterator):

    def fill(self, X, y):
        self.dataset.fill(Matrix(X), Matrix(y))
        self.n_batches = len(self.dataset) - self.batch_size + 1
        return self

    def batch(self, iteration):
        batch_slice = slice(iteration, iteration + self.batch_size)
        X_batch, y_batch = self.dataset[batch_slice]
        return X_batch, y_batch