try:
    import cupy as np
except:
    import numpy as np
from grandlantern.matrix.Matrix import Matrix

EPS = 10e-5

class Loss:
    use_mask: bool

    def __init__(self, use_mask = False):
        self.use_mask = use_mask

    def __call__(self, y_true, y_pred):
        pass

    def __str__(self):
        return f"Base"


class MSELoss(Loss):

    def __call__(self, y_true, y_pred, mask=None):
        squared_error = (y_pred - y_true) ** 2
        if (self.use_mask) and (mask is not None):
            n_real = Matrix.sum(mask)
            return Matrix.sum(mask * squared_error) / n_real
        return Matrix.mean(squared_error)

    def __str__(self):
        return f"MSE"
    

class BinaryCrossEntropy(Loss):
    
    def __call__(self, y_true, y_pred, mask=None):
        log_error = y_true * Matrix.log(y_pred + EPS) + (1 - y_true) * Matrix.log(1 - y_pred + EPS)
        if (self.use_mask) and (mask is not None):
            n_real = Matrix.sum(mask)
            return -1 * Matrix.sum(mask * log_error) / n_real
        return -1 * Matrix.mean(log_error)

    def __str__(self):
        return f"BinaryCrossEntropy"


class CrossEntropy(Loss):
    use_one_hot: bool

    def __init__(self, use_mask = False, use_one_hot = False):
        super().__init__(use_mask)
        self.use_one_hot = use_one_hot

    def __call__(self, y_true, y_pred, mask=None):
        if (self.use_one_hot):
            log_error = Matrix.sum(y_true * Matrix.log(y_pred + EPS), axis=-1)
            if (self.use_mask) and (mask is not None):
                n_real = Matrix.sum(mask)
                return -1 * Matrix.sum(mask * log_error) / n_real
            return -1 * Matrix.mean(log_error)
        else:
            probas = Matrix.take_along_axis(y_pred, y_true, axis=-1)
            log_error = Matrix.log(probas + EPS)
            if (self.use_mask) and (mask is not None):
                n_real = Matrix.sum(mask)
                return -1 * Matrix.sum(mask * log_error) / n_real
            return -1 * Matrix.mean(log_error)

    def __str__(self):
        return f"CrossEntropy"
