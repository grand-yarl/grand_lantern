import numpy as np
from grandlantern.matrix.Matrix import Matrix


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
        return np.mean((y_true.value - y_pred.value) ** 2)

    def __str__(self):
        return f"MSE"


class R2(Metric):

    def __call__(self, y_true, y_pred):
        ss_res = np.sum((y_true.value - y_pred.value) ** 2)
        ss_tot = np.sum((y_true.value - np.mean(y_true.value)) ** 2)
        return 1 - ss_res / ss_tot

    def __str__(self):
        return f"R2"
