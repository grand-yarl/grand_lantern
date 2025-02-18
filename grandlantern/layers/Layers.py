import numpy as np
from grandlantern.matrix.Matrix import Matrix
from .Activation import ActivationFunction, Linear
from .Regularizers import BaseRegularizer


class Layer:
    parameters: list
    regularizer: BaseRegularizer

    def __init__(self):
        self.parameters = []
        self.regularizer = BaseRegularizer()

    def get_parameters(self):
        return self.parameters

    def get_regularizer(self):
        return self.regularizer

    def forward(self, X, train_mode):
        pass

    def make_constant(self):
        for param in self.parameters:
            param.require_grad = False
        self.parameters = []
        return self

    def __str__(self):
        return f"Base Layer."


class LinearLayer(Layer):
    W: Matrix
    bias: Matrix
    n_neurons: int
    biased: bool
    activation: ActivationFunction

    def __init__(self, n_neurons, activation, biased=False, regularizer=BaseRegularizer()):
        super().__init__()
        self.n_neurons = n_neurons
        self.activation = activation
        self.regularizer = regularizer
        self.biased = biased
        self.W = None
        return

    def initialize_weights(self, n_inputs):
        self.W = Matrix.normal(shape=(n_inputs, self.n_neurons),
                               require_grad=True)
        self.parameters = [self.W]
        if self.biased:
            self.bias = Matrix.normal(shape=(1, self.n_neurons),
                                      require_grad=True)
            self.parameters = [self.W, self.bias]
        self.regularizer.define_params(self.parameters)
        return

    def forward(self, X, train_mode):
        if self.W is None:
            self.initialize_weights(X.shape[-1])

        if self.biased:
            return self.activation(X @ self.W + self.bias)

        return self.activation(X @ self.W)

    def __str__(self):
        return f"Linear Layer with n_neurons {self.n_neurons}, " \
               f"biased {self.biased}, " \
               f"activation {self.activation}, " \
               f"regularizer {self.regularizer}."


class BatchNormLayer(Layer):
    gamma: Matrix
    beta: Matrix

    def __init__(self):
        super().__init__()
        self.gamma = None
        self.beta = None
        return

    def initialize_weights(self, n_inputs):
        self.gamma = Matrix.ones(shape=(n_inputs), require_grad=True)
        self.beta = Matrix.zeros(shape=(n_inputs), require_grad=True)
        self.parameters = [self.gamma, self.beta]
        return

    def forward(self, X, train_mode):
        if (self.gamma is None) or (self.beta is None):
            self.initialize_weights(X.shape[1:])

        mean = Matrix.mean(X, axis=0, keepdims=True)
        std = Matrix.std(X, axis=0, keepdims=True)

        X_normed = (X - mean) / (std ** 2 + 10e-5)

        return X_normed * self.gamma + self.beta

    def __str__(self):
        return f"Batch Norm Layer."


class DropOutLayer(Layer):
    prob: float

    def __init__(self, prob=0.1):
        super().__init__()
        self.prob = prob
        return

    def forward(self, X, train_mode):
        if train_mode:
            d_shape = (1,) + X.shape[1:]
            d = np.random.uniform(low=0, high=1, size=d_shape)
            D = Matrix(d > self.prob)
            return D * X
        else:
            return X

    def __str__(self):
        return f"Dropout Layer."


class Conv2DLayer(LinearLayer):
    kernel_size: np.array([int, int])
    dilation: np.array([int, int])

    def __init__(self, kernel_size, n_channels, activation, dilation=(1, 1), biased=False, regularizer=BaseRegularizer()):
        super().__init__(n_channels, activation, biased, regularizer)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.W = None
        return

    def initialize_weights(self, n_inputs):
        self.W = Matrix.normal(shape=(n_inputs, self.n_neurons, self.kernel_size[0], self.kernel_size[1]),
                               require_grad=True)
        self.parameters = [self.W]
        if self.biased:
            self.bias = Matrix.normal(shape=(self.n_neurons),
                                      require_grad=True)
            self.parameters = [self.W, self.bias]
        self.regularizer.define_params(self.parameters)
        return

    def forward(self, X, train_mode):
        if self.W is None:
            self.initialize_weights(X.shape[1])

        if self.biased:
            """
            need to fix
            """
            WX = Matrix.conv2d(X, self.W, self.dilation)
            for c in range(self.n_neurons):
                WX_bias = WX[:, c, :, :] + self.bias[c]

            return self.activation(WX_bias)

        return self.activation(Matrix.conv2d(X, self.W, self.dilation))

    def __str__(self):
        return f"Convolutional layer with kernel {self.kernel_size}, " \
               f"channels {self.n_neurons}, " \
               f"dilation {self.dilation}, " \
               f"biased {self.biased}, " \
               f"activation {self.activation}, "  \
               f"regularizer {self.regularizer}."


class RecursiveLayer(Layer):
    Wx: Matrix
    Wh: Matrix
    bias: Matrix
    biased: bool
    activation: ActivationFunction

    def __init__(self, n_neurons, activation, biased=False, regularizer=BaseRegularizer()):
        super().__init__()
        self.n_neurons = n_neurons
        self.activation = activation
        self.regularizer = regularizer
        self.biased = biased
        self.Wx = None
        self.Wh = None
        return

    def initialize_weights(self, n_inputs):
        self.Wx = Matrix.normal(shape=(n_inputs, self.n_neurons), require_grad=True)
        self.Wh = Matrix.normal(shape=(self.n_neurons, self.n_neurons), require_grad=True)
        self.parameters = [self.Wx, self.Wh]

        if self.biased:
            self.bias = Matrix.normal(shape=(1, self.n_neurons), require_grad=True)
            self.parameters = [self.Wx, self.Wh, self.bias]
        self.regularizer.define_params(self.parameters)
        return

    def forward(self, X, train_mode):
        if self.Wx is None:
            self.initialize_weights(X.shape[2])

        H = Matrix.zeros(shape=(X.shape[0], X.shape[1] + 1, self.n_neurons))

        for i in range(X.shape[1]):
            if self.biased:
                H[:, i + 1] = self.activation(X[:, i] @ self.Wx + H[:, i] @ self.Wh + self.bias)
            else:
                H[:, i + 1] = self.activation(X[:, i] @ self.Wx + H[:, i] @ self.Wh)
        return H[:, 1:]

    def __str__(self):
        return f"Recursive Layer with n_neurons {self.n_neurons}, " \
               f"biased {self.biased}, " \
               f"activation {self.activation}, "  \
               f"regularizer {self.regularizer}."


class RNNLayer(Layer):
    layers: list[Layer]
    n_inner_neurons: int
    n_out_neurons: int
    biased: bool
    activation: ActivationFunction

    def __init__(self, n_inner_neurons, n_out_neurons, activation, biased=False, regularizer=BaseRegularizer()):
        super().__init__()
        self.n_inner_neurons = n_inner_neurons
        self.n_out_neurons = n_out_neurons
        self.activation = activation
        self.biased = biased
        self.regularizer = regularizer
        self.layers = [
            RecursiveLayer(self.n_inner_neurons, self.activation, self.biased),
            LinearLayer(self.n_out_neurons, Linear(), self.biased)
        ]
        return

    def forward(self, X, train_mode):
        current = X
        self.parameters = []
        for layer in self.layers:
            current = layer.forward(current, train_mode)
            self.parameters += layer.get_parameters()
        self.regularizer.define_params(self.parameters)
        return current

    def __str__(self):
        return f"RNN Layer with n_inner_neurons {self.n_inner_neurons}, " \
               f"n_out_neurons {self.n_out_neurons}, " \
               f"biased {self.biased}, " \
               f"activation {self.activation}, "  \
               f"regularizer {self.regularizer}."


class FlattenLayer(Layer):
    input_shape: tuple

    def __init__(self):
        super().__init__()
        self.biased = False
        return

    def forward(self, X, train_mode):
        self.input_shape = np.array(X.shape)
        X_reshaped = X.reshape(shape=(X.shape[0], np.prod(self.input_shape[1:])))
        return X_reshaped

    def __str__(self):
        return f"Flatten layer."
