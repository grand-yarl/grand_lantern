import numpy as np
from grandlantern.matrix import Matrix
from grandlantern.layers import Layer, LinearLayer, LayerNormLayer, DropOutLayer, SequenceLayer, SkipConnectionLayer
from grandlantern.layers import Linear, ReLU
from grandlantern.layers import BaseRegularizer

NEG = -10e9


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
    
    def forward(self, XYZ, train_mode, mask=None):
        Xq = XYZ[0]
        Xk = XYZ[1]
        Xv = XYZ[2]

        if Xq.shape[-1] != Xk.shape[-1] != Xv.shape[-1]:
            raise ValueError("Q, K, V must have same feature dimension")
        if (mask is not None) and (mask.ndim == 2):
            mask = mask[:, np.newaxis, :]  
        if self.Wq is None:
            self.initialize_weights(Xq.shape[-1])
        heads = []

        for i in range(self.n_heads):
            Q = Xq @ self.Wq[i]
            K = Xk @ self.Wk[i]
            V = Xv @ self.Wv[i]

            logits = (Q @ K.transpose() / np.sqrt(self.n_neurons))
            if mask is not None:
                logits += (1 - mask) * NEG
            probas = Matrix.safe_softmax(logits, axis = -1)
            Z = probas @ V
            heads.append(Z)

        Z_full = Matrix.concat(heads, axis = -1)
        return Z_full @ self.Wo
    
    def __str__(self):
        return f"MultiHead Attention Layer with n_neurons {self.n_neurons}, " \
               f"n_heads {self.n_heads}, " \
               f"regularizer {self.regularizer}."
    

class TransformerEncoderLayer(Layer):
    emb_dim: int
    attn_neurons: int
    n_heads: int
    ffn_neurons: int
    dropout_prob: int

    attention_layer: MultiHeadAttentionLayer
    ffn_layers: SequenceLayer
    norm1: LayerNormLayer
    norm2: LayerNormLayer
    dropout1: DropOutLayer
    dropout2: DropOutLayer

    def __init__(self, emb_dim, attn_neurons, n_heads, ffn_neurons, dropout_prob = 0.1, activation=ReLU(), regularizer=BaseRegularizer()):
        super().__init__()
        self.emb_dim = emb_dim
        self.attn_neurons = attn_neurons
        self.n_heads = n_heads
        self.ffn_neurons = ffn_neurons
        self.dropout_prob = dropout_prob

        # SubLayers
        self.attention_layer = MultiHeadAttentionLayer(n_neurons=attn_neurons, n_heads=n_heads, regularizer=regularizer)
        ffn_layers = [
            LinearLayer(n_neurons=ffn_neurons, activation=activation, biased=True, regularizer=regularizer),
            LinearLayer(n_neurons=emb_dim, activation=Linear(), biased=True, regularizer=regularizer)
        ]
        self.ffn_layers = SequenceLayer(layers=ffn_layers)
        self.norm1 = LayerNormLayer()
        self.norm2 = LayerNormLayer()
        self.dropout1 = DropOutLayer(prob=dropout_prob)
        self.dropout2 = DropOutLayer(prob=dropout_prob)

    def forward(self, X, train_mode, mask=None):
        X_norm1 = self.norm1.forward(X, train_mode=train_mode)
        X_attn = self.attention_layer.forward((X_norm1, X_norm1, X_norm1), train_mode=train_mode, mask=mask)
        X_attn_drop = self.dropout1.forward(X_attn, train_mode=train_mode)
        Y_attn = X + X_attn_drop

        Y_norm2 = self.norm2.forward(Y_attn, train_mode=train_mode)
        Y_ffn = self.ffn_layers.forward(Y_norm2, train_mode=train_mode)
        Y_ffn_drop = self.dropout2.forward(Y_ffn, train_mode=train_mode)
        Z = Y_attn + Y_ffn_drop

        self.parameters = (self.norm1.get_parameters() + self.attention_layer.get_parameters() + self.norm2.get_parameters() + self.ffn_layers.get_parameters())
        self.regularizer.define_params(self.parameters)
        return Z
    
    def __str__(self):
        return f"Transformer Encoder Layer with dimension={self.emb_dim}, " \
               f"attention dimension {self.attn_neurons}, n_heads={self.n_heads}, " \
               f"ffn dimension {self.ffn_neurons}, activation={self.ffn_layers.layers[0].activation}) " \
               f"dropout proba {self.dropout_prob} and regularizer {self.regularizer}."


class TransformerEncoderStack(SequenceLayer):
    n_layers: int
    emb_dim: int
    attn_neurons: int
    n_heads: int
    ffn_neurons: int
    dropout_prob: int

    def __init__(self, n_layers, emb_dim, attn_neurons, n_heads, ffn_neurons, dropout_prob = 0.1, activation=ReLU(), regularizer=BaseRegularizer()):
        super().__init__([])
        self.n_layers = n_layers
        self.emb_dim = emb_dim
        self.attn_neurons = attn_neurons
        self.n_heads = n_heads
        self.ffn_neurons = ffn_neurons
        self.dropout_prob = dropout_prob

        self.layers = [
            TransformerEncoderLayer(emb_dim, attn_neurons, n_heads, ffn_neurons, dropout_prob = dropout_prob, activation=activation, regularizer=regularizer)
            for i in range(n_layers)
        ]
    
    def __str__(self):
        return f"Transformer Encoder Stack Layer with n_layers={self.n_layers} " \
               f"dimension={self.emb_dim}, " \
               f"attention dimension {self.attn_neurons}, n_heads={self.n_heads}, " \
               f"ffn dimension {self.ffn_neurons}, activation={self.ffn_layers.layers[0].activation}) " \
               f"dropout proba {self.dropout_prob} and regularizer {self.regularizer}."
