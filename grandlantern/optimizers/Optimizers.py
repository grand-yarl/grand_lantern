from grandlantern.matrix.Matrix import Matrix
import numpy as np


class Optimizer:

    def optimize(self, parameters, gradients):
        pass


class SGD(Optimizer):
    learning_rate: float
    clip: float

    def __init__(self, learning_rate, clip=None):
        self.learning_rate = learning_rate
        self.clip = clip
        return

    def clipping(self, gradient):
        if isinstance(gradient, int):
            if abs(gradient) > self.clip:
                gradient = np.sign(gradient) * self.clip
        else:
            gradient[abs(gradient) > self.clip] = np.sign(gradient[abs(gradient) > self.clip]) * self.clip
        return gradient

    def optimize(self, parameters, gradients):
        for param in parameters:
            if self.clip:
                gradients[param] = self.clipping(gradients[param])
            param.value -= self.learning_rate * gradients[param]
        return self


class NAD(SGD):
    inertia_moment: float
    last_gradients: dict

    def __init__(self, learning_rate, inertia_moment, clip=None):
        SGD.__init__(self, learning_rate, clip=clip)
        self.inertia_moment = inertia_moment
        self.last_gradients = {}
        return

    def update_last_gradient(self, parameter, gradient):
        if parameter not in self.last_gradients:
            self.last_gradients[parameter] = self.learning_rate * gradient
        else:
            self.last_gradients[parameter] = self.learning_rate * gradient + \
                                         self.inertia_moment * self.last_gradients[parameter]
        return

    def optimize(self, parameters, gradients):
        for param in parameters:
            if self.clip:
                gradients[param] = self.clipping(gradients[param])
            self.update_last_gradient(param, gradients[param])
            param.value -= self.last_gradients[param]
        return self


class Adagrad(SGD):
    sum_square_grad: np.array

    def __init__(self, learning_rate, clip=None):
        SGD.__init__(self, learning_rate, clip=clip)
        self.sum_square_grad = {}
        return

    def update_sum_square_grad(self, parameter, gradient):
        if parameter not in self.sum_square_grad:
            self.sum_square_grad[parameter] = gradient ** 2
        else:
            self.sum_square_grad[parameter] += gradient ** 2
        return

    def optimize(self, parameters, gradients):
        for param in parameters:
            if self.clip:
                gradients[param] = self.clipping(gradients[param])
            self.update_sum_square_grad(param, gradients[param])
            param.value -= self.learning_rate / (np.sqrt(self.sum_square_grad[param] + 10e-10)) * gradients[param]
        return self


class RMSProp(Adagrad):
    last_grad_moment: float

    def __init__(self, learning_rate, last_grad_moment, clip=None):
        Adagrad.__init__(self, learning_rate, clip=clip)
        self.last_grad_moment = last_grad_moment
        return

    def update_sum_square_grad(self, parameter, gradient):
        if parameter not in self.sum_square_grad:
            self.sum_square_grad[parameter] = (1 - self.last_grad_moment) * gradient ** 2
        else:
            self.sum_square_grad[parameter] = (1 - self.last_grad_moment) * gradient ** 2 + \
                                          self.last_grad_moment * self.sum_square_grad[parameter]
        return


class Adam(NAD, RMSProp):
    iteration: int

    def __init__(self, learning_rate, beta1=0.9, beta2=0.99, clip=None):
        NAD.__init__(self, learning_rate, beta1, clip=clip)
        RMSProp.__init__(self, learning_rate, beta2, clip=clip)
        self.iteration = 0
        return

    def update_last_gradient(self, parameter, gradient):
        if parameter not in self.last_gradients:
            self.last_gradients[parameter] = (1 - self.inertia_moment) * gradient
        else:
            self.last_gradients[parameter] = (1 - self.inertia_moment) * gradient +\
                                             self.inertia_moment * self.last_gradients[parameter]
        return

    def optimize(self, parameters, gradients):
        self.iteration += 1

        for param in parameters:
            if self.clip:
                gradients[param] = self.clipping(gradients[param])
            self.update_last_gradient(param, gradients[param])
            self.update_sum_square_grad(param, gradients[param])

            new_gradient = self.last_gradients[param] / (1 - self.inertia_moment ** self.iteration)
            new_sum_square_grad = self.sum_square_grad[param] / (1 - self.last_grad_moment ** self.iteration)

            param.value -= self.learning_rate / (np.sqrt(new_sum_square_grad + 10e-10)) * new_gradient
        return self
