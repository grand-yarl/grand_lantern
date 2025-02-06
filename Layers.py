import numpy as np
from Matrix import *
from Activation import ActivationFunction


class Layer:
    parameters: list

    def get_parameters(self):
        return self.parameters

    def forward(self, X):
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
    parameters: list

    def __init__(self, n_neurons, activation, biased=False):
        self.n_neurons = n_neurons
        self.activation = activation
        self.biased = biased
        self.W = None
        self.parameters = []
        return

    def initialize_weights(self, n_inputs):
        self.W = Matrix.normal(shape=(n_inputs, self.n_neurons),
                               require_grad=True)
        self.parameters = [self.W]
        if self.biased:
            self.bias = Matrix.normal(shape=(1, self.n_neurons),
                                      require_grad=True)
            self.parameters = [self.W, self.bias]
        return

    def forward(self, X):
        if self.W is None:
            self.initialize_weights(X.shape[1])

        if self.biased:
            return self.activation(X @ self.W + self.bias)

        return self.activation(X @ self.W)

    def __str__(self):
        return f"Layer with n_neurons {self.n_neurons}, biased {self.biased}, activation {self.activation}."


class BatchNormLayer(Layer):
    gamma: Matrix
    beta: Matrix
    parameters: list

    def __init__(self):
        self.gamma = None
        self.beta = None
        self.parameters = []
        return

    def initialize_weights(self):
        self.gamma = Matrix(1., require_grad=True)
        self.beta = Matrix(0., require_grad=True)
        self.parameters = [self.gamma, self.beta]
        return

    def forward(self, X):
        if (self.gamma is None) or (self.beta is None):
            self.initialize_weights()

        mean = Matrix.mean(X, axis=0, keepdims=True)
        std = Matrix.std(X, axis=0, keepdims=True)

        X_normed = (X - mean) / (std ** 2 + 10e-3)

        return X_normed * self.gamma + self.beta

    def __str__(self):
        return f"Batch Norm Layer."


class Conv2DLayer(LinearLayer):
    kernel_size: np.array([int, int])
    n_channels: int
    dilation: np.array([int, int])

    def __init__(self, kernel_size, n_channels, activation, dilation=(1, 1), biased=False):
        self.kernel_size = kernel_size
        self.n_channels = n_channels
        self.dilation = dilation
        self.activation = activation
        self.biased = biased
        self.W = None
        self.parameters = []
        return

    def initialize_weights(self, n_inputs):
        self.W = Matrix.normal(shape=(n_inputs, self.n_channels, self.kernel_size[0], self.kernel_size[1]),
                               require_grad=True)
        self.parameters = [self.W]
        if self.biased:
            self.bias = Matrix.normal(shape=(self.n_channels),
                                      require_grad=True)
            self.parameters = [self.W, self.bias]
        return

    def forward(self, X):
        if self.W is None:
            self.initialize_weights(X.shape[1])

        if self.biased:
            """
            need to fix
            """
            WX = Matrix.conv2d(X, self.W, self.dilation)
            for c in range(self.n_channels):
                WX_bias = WX[:, c, :, :] + self.bias[c]

            return self.activation(WX_bias)

        return self.activation(Matrix.conv2d(X, self.W, self.dilation))

    def __str__(self):
        return f"Convolutional layer with kernel {self.kernel_size}, channels {self.n_channels}, dilation {self.dilation}, biased {self.biased}, activation {self.activation}."


class FlattenLayer(Layer):
    input_shape: tuple

    def __init__(self):
        self.biased = False
        self.parameters = []
        return

    def forward(self, X):
        self.input_shape = np.array(X.shape)
        X_reshaped = X.reshape(shape=(X.shape[0], np.prod(self.input_shape[1:])))
        return X_reshaped

    def __str__(self):
        return f"Flatten layer."
