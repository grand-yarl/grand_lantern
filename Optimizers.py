import numpy as np
from Matrix import *


class Optimizer:

    def optimize(self, parameters, gradients):
        pass


class SGD(Optimizer):
    learning_rate: float

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        return

    def optimize(self, parameters, gradients):
        for param in parameters:
            param.value -= self.learning_rate * gradients[param]
        return self


class NAD(SGD):
    inertia_moment: float
    last_gradients: float

    def __init__(self, learning_rate, inertia_moment):
        super().__init__(learning_rate)
        self.inertia_moment = inertia_moment
        self.last_gradients = {}
        return

    def optimize(self, parameters, gradients):
        for param in parameters:
            if param not in self.last_gradients:
                self.last_gradients[param] = self.learning_rate * gradients[param]
            else:
                self.last_gradients[param] = self.learning_rate * gradients[param] + self.inertia_moment * \
                                             self.last_gradients[param]
            param.value -= self.learning_rate * self.last_gradients[param]
        return self


class Adagrad(SGD):
    sum_square_update: np.array

    def __init__(self, learning_rate):
        super().__init__(learning_rate)
        self.sum_square_update = {}
        return

    def optimize(self, parameters, gradients):
        for param in parameters:
            if param not in self.sum_square_update:
                self.sum_square_update[param] = gradients[param] ** 2
            else:
                self.sum_square_update[param] += gradients[param] ** 2

            param.value -= self.learning_rate / (np.sqrt(self.sum_square_update[param] + 10e-10)) * gradients[param]
        return self


class Adam(NAD, Adagrad):
    beta1: float
    beta2: float
    iteration: int

    def __init__(self, learning_rate, beta1=0.9, beta2=0.99):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.sum_square_update = {}
        self.last_gradients = {}
        self.iteration = 0
        return

    def optimize(self, parameters, gradients):
        self.iteration += 1

        for param in parameters:
            if param not in self.last_gradients:
                self.last_gradients[param] = (1 - self.beta1) * gradients[param]
                self.sum_square_update[param] = (1 - self.beta2) * gradients[param] ** 2
            else:
                self.last_gradients[param] = (1 - self.beta1) * gradients[param] + self.beta1 * self.last_gradients[
                    param]
                self.sum_square_update[param] = (1 - self.beta2) * gradients[param] ** 2 + self.beta2 * \
                                                self.sum_square_update[param]

            new_gradient = self.last_gradients[param] / (1 - self.beta1 ** self.iteration)
            new_sum_square_update = self.sum_square_update[param] / (1 - self.beta2 ** self.iteration)

            param.value -= self.learning_rate / (np.sqrt(new_sum_square_update + 10e-10)) * new_gradient
        return self
