try:
    import cupy as np
except:
    import numpy as np
from grandlantern.matrix.Matrix import Matrix

EPS = 10e-5

class Loss:

    def __call__(self, y_true, y_pred):
        pass

    def __str__(self):
        return f"Base"


class MSELoss(Loss):

    def __call__(self, y_true, y_pred):
        return Matrix.mean((y_pred - y_true) ** 2)

    def __str__(self):
        return f"MSE"
    

class BinaryCrossEntropy(Loss):
    
    def __call__(self, y_true, y_pred):
        summ = -1 * Matrix.mean(y_true * Matrix.log(y_pred + EPS) + (1 - y_true) * Matrix.log(1 - y_pred + EPS))
        return summ

    def __str__(self):
        return f"BinaryCrossEntropy"


class CrossEntropy(Loss):

    def __call__(self, y_true, y_pred):
        summ = -1 * Matrix.mean(Matrix.sum(y_true * Matrix.log(y_pred + EPS), axis=1))
        return summ

    def __str__(self):
        return f"CrossEntropy"
