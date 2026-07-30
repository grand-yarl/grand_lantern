"""
try:
    import cupy as np
except:
    import numpy as np
"""
import numpy as np
import math
from grandlantern.matrix.Matrix import Matrix
from .activation.Activation import ActivationFunction, Linear, Sigmoid, Tanh
from .regularizers.Regularizers import BaseRegularizer

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

    def __init__(self, momentum=0.9):
        super().__init__()
        self.momentum = momentum
        self.gamma = None
        self.beta = None
        self.running_mean = None
        self.running_var = None

    def initialize_weights(self, n_features):
        self.gamma = Matrix.ones((n_features,), require_grad=True)
        self.beta = Matrix.zeros((n_features,), require_grad=True)
        self.running_mean = np.zeros(n_features)
        self.running_var = np.ones(n_features)
        self.parameters = [self.gamma, self.beta]

    def forward(self, X, train_mode):
        if self.gamma is None:
            self.initialize_weights(X.shape[1])

        if train_mode:
            mean = Matrix.mean(X, axis=0, keepdims=True)
            var = Matrix.mean((X - mean) ** 2, axis=0, keepdims=True)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean.value.squeeze()
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var.value.squeeze()
        else:
            mean = Matrix(self.running_mean.reshape(1, -1))
            var = Matrix(self.running_var.reshape(1, -1))

        X_normed = (X - mean) / (var + EPS) ** 0.5
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
        return f"Convolutional Layer with kernel {self.kernel_size}, " \
               f"channels {self.n_neurons}, " \
               f"dilation {self.dilation}, " \
               f"biased {self.biased}, " \
               f"activation {self.activation}, "  \
               f"regularizer {self.regularizer}."


class MultiHeadAttentionLayer(Layer):
    Wq: Matrix
    Wk: Matrix
    Wv: Matrix
    Wo: Matrix
    n_neurons: int
    n_heads: int

    def __init__(self, n_neurons, n_heads, regularizer=BaseRegularizer()):
        super().__init__()
        self.n_neurons = n_neurons
        self.n_heads = n_heads
        self.regularizer = regularizer
        self.Wq = None
        self.Wk = None
        self.Wv = None
        self.Wo = None
    
    def initialize_weights(self, n_inputs):
        k = np.sqrt(1 / n_inputs)
        self.Wq = Matrix.uniform(low=-k, high=k, shape=(self.n_heads, n_inputs, self.n_neurons), require_grad=True)
        self.Wk = Matrix.uniform(low=-k, high=k, shape=(self.n_heads, n_inputs, self.n_neurons), require_grad=True)
        self.Wv = Matrix.uniform(low=-k, high=k, shape=(self.n_heads, n_inputs, self.n_neurons), require_grad=True)
        self.Wo = Matrix.uniform(low=-k, high=k, shape=(self.n_heads * self.n_neurons, n_inputs), require_grad=True)

        self.parameters = [self.Wq, self.Wk, self.Wv, self.Wo]
        self.regularizer.define_params(self.parameters)
        return
    
    def forward(self, XX, train_mode):
        Xq = XX[0]
        Xk = XX[1]
        Xv = XX[2]

        if Xq.shape[-1] != Xk.shape[-1] != Xv.shape[-1]:
            raise ValueError("Q, K, V must have same feature dimension")
        if self.Wq is None:
            self.initialize_weights(Xq.shape[-1])
        heads = []

        for i in range(self.n_heads):
            Q = Xq @ self.Wq[i]
            K = Xk @ self.Wk[i]
            V = Xv @ self.Wv[i]

            logits = (Q @ K.transpose() / np.sqrt(self.n_neurons))
            probas = Matrix.safe_softmax(logits, axis = -1)
            Z = probas @ V
            heads.append(Z)

        Z_full = Matrix.concat(heads, axis = -1)
        return Z_full @ self.Wo
    
    def __str__(self):
        return f"MultiHead Attention Layer with n_neurons {self.n_neurons}, " \
               f"n_heads {self.n_heads}, " \
               f"regularizer {self.regularizer}."


class EmbeddingLayer(Layer):
    Emb: list[Matrix]
    emb_num: int
    emb_dim: int

    def __init__(self, emb_num, emb_dim):
        super().__init__()

        self.Emb = [Matrix.uniform(low=-1, high=1, shape=(emb_dim), require_grad=True) for i in range(emb_num)]
        self.parameters = self.Emb

        self.emb_num = emb_num
        self.emb_dim = emb_dim
        return

    def forward(self, X, train_mode):
        indices = X.value.astype(int) 
        flat_indices = indices.flatten()
        emb_list = [self.Emb[idx] for idx in flat_indices]
        embeddings = Matrix.stack(emb_list, axis=0)
        new_shape = list(indices.shape) + [self.emb_dim]
        embeddings = embeddings.reshape(new_shape)
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