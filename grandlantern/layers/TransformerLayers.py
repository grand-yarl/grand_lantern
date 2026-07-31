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
    n_out: int

    def __init__(self, n_neurons, n_heads, n_out = None, regularizer=BaseRegularizer()):
        super().__init__()
        self.n_neurons = n_neurons
        self.n_heads = n_heads
        self.regularizer = regularizer
        self.n_out = n_out
        self.Wq = None
        self.Wk = None
        self.Wv = None
        self.Wo = None
    
    def initialize_weights(self, n_inputs_q, n_inputs_k, n_inputs_v):
        k = np.sqrt(1 / n_inputs_q)
        self.Wq = Matrix.uniform(low=-k, high=k, shape=(self.n_heads, n_inputs_q, self.n_neurons), require_grad=True)
        self.Wk = Matrix.uniform(low=-k, high=k, shape=(self.n_heads, n_inputs_k, self.n_neurons), require_grad=True)
        self.Wv = Matrix.uniform(low=-k, high=k, shape=(self.n_heads, n_inputs_v, self.n_neurons), require_grad=True)
        if self.n_out is None:
            self.n_out = n_inputs_v
        self.Wo = Matrix.uniform(low=-k, high=k, shape=(self.n_heads * self.n_neurons, self.n_out), require_grad=True)

        self.parameters = [self.Wq, self.Wk, self.Wv, self.Wo]
        self.regularizer.define_params(self.parameters)
        return
    
    def forward(self, XYZ, train_mode, mask=None):
        Xq = XYZ[0]
        Xk = XYZ[1]
        Xv = XYZ[2]

        if (mask is not None) and (mask.ndim == 2):
            mask = mask[:, np.newaxis, :]  
        if self.Wq is None:
            self.initialize_weights(Xq.shape[-1], Xk.shape[-1], Xv.shape[-1])
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
    

class TransformerDecoderLayer(Layer):
    emb_dim: int
    attn_neurons: int
    n_heads: int
    ffn_neurons: int
    dropout_prob: int

    self_attention_layer: MultiHeadAttentionLayer
    cross_attention_layer: MultiHeadAttentionLayer
    ffn_layers: SequenceLayer
    norm1: LayerNormLayer
    norm2: LayerNormLayer
    norm3: LayerNormLayer
    dropout1: DropOutLayer
    dropout2: DropOutLayer
    dropout3: DropOutLayer

    def __init__(self, emb_dim, attn_neurons, n_heads, ffn_neurons, dropout_prob = 0.1, activation=ReLU(), regularizer=BaseRegularizer()):
        super().__init__()
        self.emb_dim = emb_dim
        self.attn_neurons = attn_neurons
        self.n_heads = n_heads
        self.ffn_neurons = ffn_neurons
        self.dropout_prob = dropout_prob

        # SubLayers
        self.self_attention_layer = MultiHeadAttentionLayer(n_neurons=attn_neurons, n_heads=n_heads, n_out=emb_dim, regularizer=regularizer)
        self.cross_attention_layer = MultiHeadAttentionLayer(n_neurons=attn_neurons, n_heads=n_heads, n_out=emb_dim, regularizer=regularizer)
        ffn_layers = [
            LinearLayer(n_neurons=ffn_neurons, activation=activation, biased=True, regularizer=regularizer),
            LinearLayer(n_neurons=emb_dim, activation=Linear(), biased=True, regularizer=regularizer)
        ]
        self.ffn_layers = SequenceLayer(layers=ffn_layers)
        self.norm1 = LayerNormLayer()
        self.norm2 = LayerNormLayer()
        self.norm3 = LayerNormLayer()
        self.dropout1 = DropOutLayer(prob=dropout_prob)
        self.dropout2 = DropOutLayer(prob=dropout_prob)
        self.dropout3 = DropOutLayer(prob=dropout_prob)

    def forward(self, XH, train_mode, self_mask=None, cross_mask=None):
        X = XH[0]
        H = XH[1]

        causal_mask = np.stack(X.shape[0] * [np.tril(np.ones((X.shape[1], X.shape[1])))], axis=0)
        if self_mask is not None:
            causal_mask *= self_mask

        X_norm1 = self.norm1.forward(X, train_mode=train_mode)
        X_attn1 = self.self_attention_layer.forward((X_norm1, X_norm1, X_norm1), train_mode=train_mode, mask=causal_mask)
        X_attn1_drop = self.dropout1.forward(X_attn1, train_mode=train_mode)
        Y_attn1 = X + X_attn1_drop

        X_norm2 = self.norm2.forward(Y_attn1, train_mode=train_mode)
        X_attn2 = self.cross_attention_layer.forward((X_norm2, H, H), train_mode=train_mode, mask=cross_mask)
        X_attn2_drop = self.dropout2.forward(X_attn2, train_mode=train_mode)
        Y_attn2 = Y_attn1 + X_attn2_drop

        Y_norm2 = self.norm3.forward(Y_attn2, train_mode=train_mode)
        Y_ffn = self.ffn_layers.forward(Y_norm2, train_mode=train_mode)
        Y_ffn_drop = self.dropout3.forward(Y_ffn, train_mode=train_mode)
        Z = Y_attn2 + Y_ffn_drop

        self.parameters = (self.norm1.get_parameters() + self.self_attention_layer.get_parameters() +
                           self.norm2.get_parameters() + self.cross_attention_layer.get_parameters() +
                           self.norm3.get_parameters() + self.ffn_layers.get_parameters())
        self.regularizer.define_params(self.parameters)
        return Z
    
    def __str__(self):
        return f"Transformer Decoder Layer with dimension={self.emb_dim}, " \
               f"attention dimension {self.attn_neurons}, n_heads={self.n_heads}, " \
               f"ffn dimension {self.ffn_neurons}, activation={self.ffn_layers.layers[0].activation}) " \
               f"dropout proba {self.dropout_prob} and regularizer {self.regularizer}."


class TransformerDecoderStack(SequenceLayer):
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
            TransformerDecoderLayer(emb_dim, attn_neurons, n_heads, ffn_neurons, dropout_prob = dropout_prob, activation=activation, regularizer=regularizer)
            for i in range(n_layers)
        ]

    def forward(self, XH, train_mode, **kwargs):
        current = XH[0]
        H = XH[1]
        self.parameters = []
        for layer in self.layers:
            current = layer.forward((current, H), train_mode, **kwargs)
            self.parameters += layer.get_parameters()
        self.regularizer.define_params(self.parameters)
        return current
    
    def __str__(self):
        return f"Transformer Decoder Stack Layer with n_layers={self.n_layers} " \
               f"dimension={self.emb_dim}, " \
               f"attention dimension {self.attn_neurons}, n_heads={self.n_heads}, " \
               f"ffn dimension {self.ffn_neurons}, activation={self.ffn_layers.layers[0].activation}) " \
               f"dropout proba {self.dropout_prob} and regularizer {self.regularizer}."