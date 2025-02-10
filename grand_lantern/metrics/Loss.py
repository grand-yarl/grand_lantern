from grand_lantern.matrix.Matrix import *


class Loss:

    def __call__(self, y_true, y_pred):
        pass

    def __str__(self):
        return f"Base"


class MSELoss(Loss):

    def __call__(self, y_true, y_pred):
        return (y_pred - y_true) ** 2

    def __str__(self):
        return f"MSE"


class CrossEntropy(Loss):

    def __call__(self, y_true, y_pred):
        summ = -1 * Matrix.sum(y_true * Matrix.log(y_pred + 10e-5), axis=1)
        return summ

    def __str__(self):
        return f"CrossEntropy"
