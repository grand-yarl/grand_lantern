import numpy as np
from grandlantern.matrix.Matrix import Matrix
from copy import copy


class Dataset:

    def __init__(self):
        return

    def fill(self, X, y):
        pass

    def shuffle(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


class TableDataset(Dataset):
    X: Matrix
    y: Matrix
    
    def fill(self, X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        self.X = Matrix(X)
        self.y = Matrix(y)
        return self

    def shuffle(self):
        indices = np.random.permutation(len(self))
        self.X = Matrix(self.X.value[indices])
        self.y = Matrix(self.y.value[indices])

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class ImageDataset(TableDataset):

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


class SequenceDataset(TableDataset):
    def __init__(self, seq_len):
        super().__init__()
        self.seq_len = seq_len

    def fill(self, X, y):
        X_arr = X.value if isinstance(X, Matrix) else np.array(X)
        y_arr = y.value if isinstance(y, Matrix) else np.array(y)
        n_seq = X_arr.shape[0] - self.seq_len + 1
        if n_seq <= 0:
            raise ValueError("Sequence length exceeds number of samples")
        X_seq = np.zeros((n_seq, self.seq_len, X_arr.shape[1]))
        y_seq = np.zeros((n_seq, self.seq_len, y_arr.shape[1]))
        for i in range(n_seq):
            X_seq[i] = X_arr[i:i+self.seq_len]
            y_seq[i] = y_arr[i:i+self.seq_len]
        self.X = Matrix(X_seq)
        self.y = Matrix(y_seq)
        return self
