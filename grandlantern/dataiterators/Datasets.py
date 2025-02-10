import numpy as np
from grandlantern.matrix.Matrix import Matrix


class Dataset():

    def __init__(self):
        return

    def fill(self, X, y):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class TableDataset(Dataset):
    X: Matrix
    y: Matrix

    def __init__(self):
        super().__init__()
        return

    def fill(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError
        self.X = Matrix(X)
        self.y = Matrix(y)
        return self

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ImageDataset(Dataset):
    X: Matrix
    y: Matrix

    def __init__(self):
        super().__init__()
        return

    def fill(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError
        if X.ndim == 3:
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        self.X = Matrix(X)
        self.y = Matrix(y)
        return self

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]