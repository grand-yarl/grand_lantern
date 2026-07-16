try:
    import cupy as np
except:
    import numpy as np
from grandlantern.matrix.Matrix import Matrix

EPS = 10e-8

class Optimizer:

    def optimize(self, parameters):
        pass


class SGD(Optimizer):
    learning_rate: float
    clip: float

    def __init__(self, learning_rate: float, clip: float = None):
        self.learning_rate = learning_rate
        self.clip = clip
        return

    def clipping(self, param: Matrix):
        grad = param.grad_
        if grad is None:
            return
        if np.ndim(grad) == 0:
            param.grad_ = np.clip(grad, -self.clip, self.clip)
        else:
            norm = np.linalg.norm(grad)
            if norm > self.clip:
                param.grad_ = grad / norm * self.clip

    def optimize(self, parameters: Matrix):
        for param in parameters:
            if param.grad_ is None:
                continue
            if self.clip:
              self.clipping(param)  
            param.value -= self.learning_rate * param.grad_
        return self


class NAD(SGD):
    inertia_moment: float
    last_gradients: dict[Matrix, np.ndarray]

    def __init__(self, learning_rate: float, inertia_moment: float, clip: float = None):
        SGD.__init__(self, learning_rate, clip=clip)
        self.inertia_moment = inertia_moment
        self.last_gradients = {}
        return

    def update_last_gradient(self, parameter: Matrix):
        if parameter not in self.last_gradients:
            self.last_gradients[parameter] = self.learning_rate * parameter.grad_
        else:
            self.last_gradients[parameter] = self.learning_rate * parameter.grad_ + \
                                         self.inertia_moment * self.last_gradients[parameter]
        return

    def optimize(self, parameters: Matrix):
        for param in parameters:
            if param.grad_ is None:
                continue
            if self.clip:
                self.clipping(param)
            self.update_last_gradient(param)
            param.value -= self.last_gradients[param]
        return self


class Adagrad(SGD):
    sum_square_grad: dict[Matrix, np.ndarray]

    def __init__(self, learning_rate: float, clip: float = None):
        SGD.__init__(self, learning_rate, clip=clip)
        self.sum_square_grad = {}
        return

    def update_sum_square_grad(self, parameter: Matrix):
        if parameter not in self.sum_square_grad:
            self.sum_square_grad[parameter] = parameter.grad_ ** 2
        else:
            self.sum_square_grad[parameter] += parameter.grad_ ** 2
        return

    def optimize(self, parameters: Matrix):
        for param in parameters:
            if param.grad_ is None:
                continue
            if self.clip:
                self.clipping(param)
            self.update_sum_square_grad(param)
            param.value -= self.learning_rate / (np.sqrt(self.sum_square_grad[param] + EPS)) * param.grad_
        return self


class RMSProp(Adagrad):
    last_grad_moment: float

    def __init__(self, learning_rate: float, last_grad_moment: float, clip: float = None):
        Adagrad.__init__(self, learning_rate, clip=clip)
        self.last_grad_moment = last_grad_moment
        return

    def update_sum_square_grad(self, parameter: Matrix):
        if parameter not in self.sum_square_grad:
            self.sum_square_grad[parameter] = (1 - self.last_grad_moment) * parameter.grad_ ** 2
        else:
            self.sum_square_grad[parameter] = (1 - self.last_grad_moment) * parameter.grad_ ** 2 + \
                                          self.last_grad_moment * self.sum_square_grad[parameter]
        return


class Adam(NAD, RMSProp):
    iteration: int

    def __init__(self, learning_rate: float, beta1: float = 0.9, beta2: float = 0.99, clip: float = None):
        NAD.__init__(self, learning_rate, beta1, clip=clip)
        RMSProp.__init__(self, learning_rate, beta2, clip=clip)
        self.iteration = 0
        return

    def update_last_gradient(self, parameter: Matrix):
        if parameter not in self.last_gradients:
            self.last_gradients[parameter] = (1 - self.inertia_moment) * parameter.grad_
        else:
            self.last_gradients[parameter] = (1 - self.inertia_moment) * parameter.grad_ +\
                                             self.inertia_moment * self.last_gradients[parameter]
        return

    def optimize(self, parameters: Matrix):
        self.iteration += 1

        for param in parameters:
            if param.grad_ is None:
                continue
            if self.clip:
                self.clipping(param)
            self.update_last_gradient(param)
            self.update_sum_square_grad(param)

            new_gradient = self.last_gradients[param] / (1 - self.inertia_moment ** self.iteration)
            new_sum_square_grad = self.sum_square_grad[param] / (1 - self.last_grad_moment ** self.iteration)

            param.value -= self.learning_rate / (np.sqrt(new_sum_square_grad + EPS)) * new_gradient
        return self
