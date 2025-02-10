from grand_lantern.matrix.Matrix import *


class ActivationFunction:

    def __call__(self, X):
        pass

    def __str__(self):
        return f"Base"


class Linear(ActivationFunction):

    def __call__(self, X):
        return X

    def __str__(self):
        return f"Linear"


class Sigmoid(ActivationFunction):

    def __call__(self, X):
        sigmoid = Matrix.sigmoid(X)
        return sigmoid

    def __str__(self):
        return f"Sigmoid"


class Tanh(ActivationFunction):

    def __call__(self, X):
        tanh = Matrix.tanh(X)
        return tanh

    def __str__(self):
        return f"Tanh"


class ReLU(ActivationFunction):

    def __call__(self, X):
        return X * (Matrix.sign(X) + 1) / 2

    def __str__(self):
        return f"ReLU"


class LReLU(ActivationFunction):
    alpha: float

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def __call__(self, X):
        return X * ((1 + Matrix.sign(X)) / 2 + self.alpha * (1 + Matrix.sign(-X)) / 2)

    def __str__(self):
        return f"LReLU"


class SoftMax(ActivationFunction):

    def __call__(self, X):
        return Matrix.safe_softmax(X)

    def __str__(self):
        return f"SoftMax"
