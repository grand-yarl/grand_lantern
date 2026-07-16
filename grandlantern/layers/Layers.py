try:
    import cupy as np
except:
    import numpy as np
import math
from grandlantern.matrix.Matrix import Matrix
from .Activation import ActivationFunction, Linear, Sigmoid, Tanh
from .Regularizers import BaseRegularizer

EPS = 10e-5

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
        k = np.sqrt(1 / n_inputs)
        self.W = Matrix.uniform(low=-k, high=k, shape=(n_inputs, self.n_neurons), require_grad=True)
        self.parameters = [self.W]
        if self.biased:
            self.bias = Matrix.uniform(low=-k, high=k, shape=(self.n_neurons), require_grad=True)
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
    running_mean: np.ndarray
    running_std: np.ndarray

    def __init__(self, momentum = 0.9):
        super().__init__()
        self.gamma = None
        self.beta = None
        self.running_mean = None
        self.running_std = None
        self.momentum = momentum
        return

    def initialize_weights(self, n_inputs):
        self.gamma = Matrix.ones(shape=(n_inputs), require_grad=True)
        self.beta = Matrix.zeros(shape=(n_inputs), require_grad=True)
        self.running_mean = np.ones(shape=(n_inputs))
        self.running_std = np.zeros(shape=(n_inputs))
        self.parameters = [self.gamma, self.beta]
        return

    def forward(self, X, train_mode):
        if (self.gamma is None) or (self.beta is None):
            self.initialize_weights(X.shape[1:])

        if (train_mode):
            mean = Matrix.mean(X, axis=0, keepdims=True)
            std = Matrix.std(X, axis=0, keepdims=True)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean.value
            self.running_std = self.momentum * self.running_std + (1 - self.momentum) * std.value
        else:
            mean = Matrix(self.running_mean)
            std = Matrix(self.running_var)

        X_normed = (X - mean) / (std ** 2 + EPS)

        return X_normed * self.gamma + self.beta

    def __str__(self):
        return f"Batch Norm Layer with momentum {self.momentum}."


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
        return f"Dropout Layer with zero probability {self.prob}."


class Conv2DLayer(LinearLayer):
    kernel_size: tuple
    dilation: tuple

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
    train_init: bool
    activation: ActivationFunction

    def __init__(self, n_neurons, activation, biased=False, train_init=False, regularizer=BaseRegularizer()):
        super().__init__()
        self.n_neurons = n_neurons
        self.activation = activation
        self.regularizer = regularizer
        self.biased = biased
        self.train_init = train_init
        self.Wx = None
        self.Wh = None
        self.h0 = None
        return

    def initialize_weights(self, n_inputs):
        k = np.sqrt(1 / self.n_neurons)
        self.Wx = Matrix.uniform(low=-k, high=k, shape=(n_inputs, self.n_neurons), require_grad=True)
        self.Wh = Matrix.uniform(low=-k, high=k, shape=(self.n_neurons, self.n_neurons), require_grad=True)
        self.parameters = [self.Wx, self.Wh]

        if self.biased:
            self.bias = Matrix.uniform(low=-k, high=k, shape=(self.n_neurons), require_grad=True)
            self.parameters = [self.Wx, self.Wh, self.bias]
        self.regularizer.define_params(self.parameters)
        return

    def forward(self, X, train_mode):
        if self.Wx is None:
            self.initialize_weights(X.shape[2])
        if self.h0 is None:
            self.h0 = Matrix.zeros(shape=(self.n_neurons), require_grad=self.train_init)

        h0 = Matrix.stack([self.h0 for i in range(X.shape[0])])
        h = [h0]

        for i in range(X.shape[1]):
            if self.biased:
                h.append(self.activation(X[:, i] @ self.Wx + h[i] @ self.Wh + self.bias))
            else:
                h.append(self.activation(X[:, i] @ self.Wx + h[i] @ self.Wh + self.bias))

        H = Matrix.stack(h[1:], axis=1)
        return H

    def __str__(self):
        return f"Recursive Layer with n_neurons {self.n_neurons}, " \
               f"biased {self.biased}, " \
               f"activation {self.activation}, " \
               f"regularizer {self.regularizer}."


class LSTMLayer(Layer):
    # Forget gate
    Wfx: Matrix
    Wfh: Matrix
    bias_f: Matrix

    # Input gate
    Wix: Matrix
    Wih: Matrix
    bias_i: Matrix

    # Cell gate
    Wcx: Matrix
    Wch: Matrix
    bias_c: Matrix

    # Output gate
    Wox: Matrix
    Woh: Matrix
    bias_o: Matrix

    biased: bool

    def __init__(self, n_neurons, biased=False, regularizer=BaseRegularizer()):
        super().__init__()
        self.n_neurons = n_neurons
        self.regularizer = regularizer
        self.biased = biased

        self.Wfx = None
        self.Wfh = None
        self.Wix = None
        self.Wih = None
        self.Wcx = None
        self.Wch = None
        self.Wox = None
        self.Woh = None
        self.h0 = None
        return

    def initialize_weights(self, n_inputs):
        k = np.sqrt(1 / self.n_neurons)
        self.Wfx = Matrix.uniform(low=-k, high=k, shape=(n_inputs, self.n_neurons), require_grad=True)
        self.Wfh = Matrix.uniform(low=-k, high=k, shape=(self.n_neurons, self.n_neurons), require_grad=True)

        self.Wix = Matrix.uniform(low=-k, high=k, shape=(n_inputs, self.n_neurons), require_grad=True)
        self.Wih = Matrix.uniform(low=-k, high=k, shape=(self.n_neurons, self.n_neurons), require_grad=True)

        self.Wcx = Matrix.uniform(low=-k, high=k, shape=(n_inputs, self.n_neurons), require_grad=True)
        self.Wch = Matrix.uniform(low=-k, high=k, shape=(self.n_neurons, self.n_neurons), require_grad=True)

        self.Wox = Matrix.uniform(low=-k, high=k, shape=(n_inputs, self.n_neurons), require_grad=True)
        self.Woh = Matrix.uniform(low=-k, high=k, shape=(self.n_neurons, self.n_neurons), require_grad=True)
        self.parameters = [self.Wfx, self.Wfh, self.Wix, self.Wih, self.Wcx, self.Wch, self.Wox, self.Woh]

        if self.biased:
            self.bias_f = Matrix.uniform(low=-k, high=k, shape=(self.n_neurons), require_grad=True)
            self.bias_i = Matrix.uniform(low=-k, high=k, shape=(self.n_neurons), require_grad=True)
            self.bias_c = Matrix.uniform(low=-k, high=k, shape=(self.n_neurons), require_grad=True)
            self.bias_o = Matrix.uniform(low=-k, high=k, shape=(self.n_neurons), require_grad=True)
            self.parameters = [self.Wfx, self.Wfh, self.Wix, self.Wih, self.Wcx, self.Wch, self.Wox, self.Woh,
                               self.bias_f, self.bias_i, self.bias_c, self.bias_o]
        self.regularizer.define_params(self.parameters)
        return

    def forward(self, X, train_mode):
        if self.Wfx is None:
            self.initialize_weights(X.shape[2])
        if self.h0 is None:
            self.h0 = Matrix.zeros(shape=(X.shape[0], self.n_neurons), require_grad=True)

        c0 = Matrix.zeros(shape=(X.shape[0], self.n_neurons))
        h = [self.h0]
        c = [c0]

        sigmoid = Sigmoid()
        tanh = Tanh()
        for i in range(X.shape[1]):
            if self.biased:
                f_t = sigmoid(X[:, i] @ self.Wfx + h[i] @ self.Wfh + self.bias_f)
                i_t = sigmoid(X[:, i] @ self.Wix + h[i] @ self.Wih + self.bias_i)
                c_t = tanh(X[:, i] @ self.Wcx + h[i] @ self.Wch + self.bias_c)
                c.append(f_t * c[i] + i_t * c_t)

                o_t = sigmoid(X[:, i] @ self.Wox + h[i] @ self.Woh + self.bias_o)
                h.append(o_t * tanh(c[i + 1]))
            else:
                f_t = sigmoid(X[:, i] @ self.Wfx + h[i] @ self.Wfh)
                i_t = sigmoid(X[:, i] @ self.Wix + h[i] @ self.Wih)
                c_t = tanh(X[:, i] @ self.Wcx + h[i] @ self.Wch)
                c.append(f_t * c[i] + i_t * c_t)

                o_t = sigmoid(X[:, i] @ self.Wox + h[i] @ self.Woh)
                h.append(o_t * tanh(c[i + 1]))

        H = Matrix.stack(h[1:], axis=1)
        return H

    def __str__(self):
        return f"LSTM Layer with n_neurons {self.n_neurons}, " \
               f"biased {self.biased}, " \
               f"regularizer {self.regularizer}."


class GRULayer(Layer):
    # Update gate
    Wzx: Matrix
    Wzh: Matrix
    bias_z: Matrix

    # Reset gate
    Wrx: Matrix
    Wrh: Matrix
    bias_r: Matrix

    # Hidden gate
    Whx: Matrix
    Whh: Matrix
    bias_h: Matrix

    def __init__(self, n_neurons, biased=False, regularizer=BaseRegularizer()):
        super().__init__()
        self.n_neurons = n_neurons
        self.regularizer = regularizer
        self.biased = biased

        self.Wzx = None
        self.Wzh = None
        self.Wrx = None
        self.Wrh = None
        self.Whx = None
        self.Whh = None
        self.h0 = None
        return

    def initialize_weights(self, n_inputs):
        k = np.sqrt(1 / self.n_neurons)
        self.Wzx = Matrix.uniform(low=-k, high=k, shape=(n_inputs, self.n_neurons), require_grad=True)
        self.Wzh = Matrix.uniform(low=-k, high=k, shape=(self.n_neurons, self.n_neurons), require_grad=True)

        self.Wrx = Matrix.uniform(low=-k, high=k, shape=(n_inputs, self.n_neurons), require_grad=True)
        self.Wrh = Matrix.uniform(low=-k, high=k, shape=(self.n_neurons, self.n_neurons), require_grad=True)

        self.Whx = Matrix.uniform(low=-k, high=k, shape=(n_inputs, self.n_neurons), require_grad=True)
        self.Whh = Matrix.uniform(low=-k, high=k, shape=(self.n_neurons, self.n_neurons), require_grad=True)
        self.parameters = [self.Wzx, self.Wzh, self.Wrx, self.Wrh, self.Whx, self.Whh]

        if self.biased:
            self.bias_z = Matrix.uniform(low=-k, high=k, shape=(self.n_neurons), require_grad=True)
            self.bias_r = Matrix.uniform(low=-k, high=k, shape=(self.n_neurons), require_grad=True)
            self.bias_h = Matrix.uniform(low=-k, high=k, shape=(self.n_neurons), require_grad=True)
            self.parameters = [self.Wzx, self.Wzh, self.Wrx, self.Wrh, self.Whx, self.Whh,
                               self.bias_z, self.bias_r, self.bias_h]
        self.regularizer.define_params(self.parameters)
        return

    def forward(self, X, train_mode):
        if self.Wzx is None:
            self.initialize_weights(X.shape[2])
        if self.h0 is None:
            self.h0 = Matrix.zeros(shape=(X.shape[0], self.n_neurons), require_grad=True)

        h = [self.h0]

        sigmoid = Sigmoid()
        tanh = Tanh()
        for i in range(X.shape[1]):
            if self.biased:
                z_t = sigmoid(X[:, i] @ self.Wzx + h[i] @ self.Wzh + self.bias_z)
                r_t = sigmoid(X[:, i] @ self.Wrx + h[i] @ self.Wrh + self.bias_r)
                h_t = tanh((h[i] * r_t) @ self.Whh + X[:, i] @ self.Whx + self.bias_h)
                h.append((1 - z_t) * h[i] + z_t * h_t)
            else:
                z_t = sigmoid(X[:, i] @ self.Wzx + h[i] @ self.Wzh)
                r_t = sigmoid(X[:, i] @ self.Wrx + h[i] @ self.Wrh)
                h_t = tanh((h[i] * r_t) @ self.Whh + X[:, i] @ self.Whx)
                h.append((1 - z_t) * h[i] + z_t * h_t)

        H = Matrix.stack(h[1:], axis=1)
        return H

    def __str__(self):
        return f"GRU Layer with n_neurons {self.n_neurons}, " \
               f"biased {self.biased}, " \
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


class EmbeddingLayer(Layer):
    Emb: list[Matrix]
    emb_num: int
    emb_dim: int

    def __init__(self, emb_num, emb_dim):
        super().__init__()

        self.Emb = []
        for i in range(emb_num):
            self.Emb.append(Matrix.uniform(low=-1, high=1, shape=(emb_dim), require_grad=True))

        self.emb_num = emb_num
        self.emb_dim = emb_dim
        return

    def forward(self, X, train_mode):
        emb_list = []
        for index in X:
            emb_list.append(self.Emb[index])
        embeddings = Matrix.stack(emb_list)
        self.parameters = emb_list
        return embeddings

    def __str__(self):
        return f"Embedding Layer with number of embeddings {self.emb_num}, " \
               f"dimension {self.emb_dim}."


class FlattenLayer(Layer):
    input_shape: tuple

    def __init__(self):
        super().__init__()
        self.biased = False
        return

    def forward(self, X, train_mode):
        self.input_shape = X.shape
        X_reshaped = X.reshape(shape=(X.shape[0], math.prod(self.input_shape[1:])))
        return X_reshaped

    def __str__(self):
        return f"Flatten layer."


class ReshapeLayer(Layer):
    input_shape: tuple
    output_shape: tuple                                                                 

    def __init__(self, output_shape):
        super().__init__()                                              
        self.output_shape = output_shape                                                                                                                                                                                                                                                                                                  
        self.biased = False
        return

    def forward(self, X, train_mode):
        self.input_shape = np.array(X.shape)
        X_reshaped = X.reshape(shape=(X.shape[0], *self.output_shape))
        return X_reshaped

    def __str__(self):
        return f"Reshape layer with output shape {self.output_shape}."