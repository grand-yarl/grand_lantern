import numpy as np
from grandlantern.matrix import Matrix
from grandlantern.layers import Layer
from grandlantern.layers import ActivationFunction, Tanh, Sigmoid
from grandlantern.layers import BaseRegularizer


class RecursiveLayer(Layer):
    cell_layer: Layer
    h0: Matrix
    return_last: bool
    train_init: bool

    def __init__(self, cell_layer: Layer, return_last=False, train_init=False):
        super().__init__()
        self.cell_layer = cell_layer
        self.train_init = train_init
        self.return_last = return_last 
        self.h0 = None
        self.n_neurons = cell_layer.n_neurons
        return

    def forward(self, X, train_mode):
        if not self.cell_layer.parameters:
            self.cell_layer.initialize_weights(X.shape[2])
        self.parameters = self.cell_layer.get_parameters()

        if self.h0 is None:
            self.h0 = Matrix.zeros(shape=(self.n_neurons), require_grad=self.train_init)
            if self.train_init:
                self.parameters += [self.h0]

        h0_batch = Matrix.stack([self.h0] * X.shape[0])
        h = [h0_batch]

        c = []
        if isinstance(self.cell_layer, LSTMCell):
            c0 = Matrix.zeros(shape=(self.n_neurons))
            c0_batch = Matrix.stack([c0] * X.shape[0])
            c = [c0_batch]

        for i in range(X.shape[1]):
            if isinstance(self.cell_layer, LSTMCell):
                h_i, c_i = self.cell_layer.forward(X[:, i], h[i], c[i], train_mode)
                h.append(h_i)
                c.append(c_i)
            else:
                h_i = self.cell_layer.forward(X[:, i], h[i], train_mode)
                h.append(h_i)
        
        if self.return_last:
            return h[-1]
        
        H = Matrix.stack(h[1:], axis=1)
        return H

    def __str__(self):
        return f"Recursive Layer with cell {self.cell}"
    

class RNNCell(Layer):
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
    
    def forward(self, X, H, train_mode):
        if self.Wx is None:
            self.initialize_weights(X.shape[2])

        if (self.biased):
            H_next = self.activation(X @ self.Wx + H @ self.Wh + self.bias)
        else:
            H_next = self.activation(X @ self.Wx + H @ self.Wh)
        return H_next
    
    def __str__(self):
        return f"Simple RNN cell with n_neurons {self.n_neurons}, " \
               f"biased {self.biased}, " \
               f"activation {self.activation}, " \
               f"regularizer {self.regularizer}."


class LSTMCell(Layer):
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

    def forward(self, X, H, C, train_mode):
        if self.Wfx is None:
            self.initialize_weights(X.shape[2])

        sigmoid = Sigmoid()
        tanh = Tanh()

        if self.biased:
            f_t = sigmoid(X @ self.Wfx + H @ self.Wfh + self.bias_f)
            i_t = sigmoid(X @ self.Wix + H @ self.Wih + self.bias_i)
            c_t = tanh(X @ self.Wcx + H @ self.Wch + self.bias_c)
            C_next = (f_t * C + i_t * c_t)

            o_t = sigmoid(X @ self.Wox + H @ self.Woh + self.bias_o)
            H_next = o_t * tanh(C_next)
        else:
            f_t = sigmoid(X @ self.Wfx + H @ self.Wfh)
            i_t = sigmoid(X @ self.Wix + H @ self.Wih)
            c_t = tanh(X @ self.Wcx + H @ self.Wch)
            C_next = (f_t * C + i_t * c_t)

            o_t = sigmoid(X @ self.Wox + H @ self.Woh)
            H_next = o_t * tanh(C_next)
        return H_next, C_next

    def __str__(self):
        return f"LSTM cell with n_neurons {self.n_neurons}, " \
               f"biased {self.biased}, " \
               f"regularizer {self.regularizer}."


class GRUCell(Layer):
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

    def forward(self, X, H, train_mode):
        if self.Wzx is None:
            self.initialize_weights(X.shape[2])

        sigmoid = Sigmoid()
        tanh = Tanh()
        if self.biased:
            z_t = sigmoid(X @ self.Wzx + H @ self.Wzh + self.bias_z)
            r_t = sigmoid(X @ self.Wrx + H @ self.Wrh + self.bias_r)
            h_t = tanh((H * r_t) @ self.Whh + X @ self.Whx + self.bias_h)
            H_next = (1 - z_t) * H + z_t * h_t
        else:
            z_t = sigmoid(X @ self.Wzx + H @ self.Wzh)
            r_t = sigmoid(X @ self.Wrx + H @ self.Wrh)
            h_t = tanh((H * r_t) @ self.Whh + X @ self.Whx)
            H_next = (1 - z_t) * H + z_t * h_t
        return H_next

    def __str__(self):
        return f"GRU cell with n_neurons {self.n_neurons}, " \
               f"biased {self.biased}, " \
               f"regularizer {self.regularizer}."