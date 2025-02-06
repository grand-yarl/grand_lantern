import numpy as np
from Matrix import *


class Metric:

    def __call__(self, y_true, y_pred):
        pass

    def __str__(self):
        return f"Base"


class Accuracy(Metric):

    def __call__(self, y_true, y_pred):
        yt = np.argmax(y_true.value, axis=1)
        yp = np.argmax(y_pred.value, axis=1)
        return len(np.where(yt == yp)[0]) / len(yt)

    def __str__(self):
        return f"Accuracy"


class MSEMetric(Metric):

    def __call__(self, y_true, y_pred):
        summ = np.sum((y_pred.value - y_true.value) ** 2)
        return 1 / len(y_true.value) * summ

    def __str__(self):
        return f"MSE"
