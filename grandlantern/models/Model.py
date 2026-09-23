try:
    import cupy as np
except:
    import numpy as np

from copy import copy
from grandlantern.matrix import Matrix
from grandlantern.layers import Layer, EmbeddingLayer, CosSinPosEncoderLayer, TransformerEncoderStack, TransformerDecoderStack
from grandlantern.metrics import Metric, Loss
from grandlantern.optimizers import Optimizer
from grandlantern.dataiterators import DatasetIterator
from grandlantern.layers import BaseRegularizer
from grandlantern.dataiterators import TOKEN_BOS, TOKEN_EOS


class SequenceModel():
    layers: list[Layer]
    n_epochs: int
    dataset_it: DatasetIterator
    loss_fn: Loss
    metric_fn: Metric
    opimizer: Optimizer
    fit_error: np.ndarray
    val_error: np.ndarray
    parameters: list[Matrix]
    regularizators: list[BaseRegularizer]

    def __init__(self, n_epochs, dataset_iterator, loss_function, metric_function, optimizer):
        self.layers = []
        self.n_epochs = n_epochs
        self.dataset_it = dataset_iterator
        self.loss_fn = loss_function
        self.metric_fn = metric_function
        self.optimizer = optimizer
        self.parameters = []
        self.regularizators = []
        return

    def add_layer(self, layer):
        self.layers.append(layer)
        return

    def pop_layer(self, n_layer):
        self.layers.pop(n_layer)
        return

    def train_forward(self, X):
        self.parameters = []
        self.regularizators = []
        current_input = X
        for layer in self.layers:
            current_input = layer.forward(current_input, train_mode = True)
            self.parameters += layer.get_parameters()
            self.regularizators.append(layer.get_regularizer())
        return current_input

    def test_forward(self, X):
        current_input = X
        for layer in self.layers:
            current_input = layer.forward(current_input, train_mode = False)
        return current_input

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.grad_ = None
        return

    def train(self, dataset_iterator):
        sum_loss_train = 0
        sum_metric_train = 0

        for (X_batch, y_batch) in dataset_iterator():
            self.zero_grad()
            y_pred = self.train_forward(X_batch)

            loss = self.loss_fn(y_batch, y_pred)
            for regularizator in self.regularizators:
                loss += regularizator()
            loss.backward()
            self.optimizer.optimize(self.parameters)
            # print(Matrix.mean(loss))

            metric = self.metric_fn(y_batch, y_pred)

            sum_loss_train += float(loss.value)
            sum_metric_train += metric

        loss_train = sum_loss_train / dataset_iterator.n_batches
        metric_train = sum_metric_train / dataset_iterator.n_batches

        return loss_train, metric_train

    def test(self, dataset_iterator):
        sum_loss_val = 0
        sum_metric_val = 0

        for (X_batch, y_batch) in dataset_iterator():
            y_pred = self.test_forward(X_batch)

            loss = self.loss_fn(y_batch, y_pred)
            metric = self.metric_fn(y_batch, y_pred)

            sum_loss_val += float(loss.value)
            sum_metric_val += metric

        loss_val = sum_loss_val / dataset_iterator.n_batches
        metric_val = sum_metric_val / dataset_iterator.n_batches

        return loss_val, metric_val

    def fit(self, X, y, X_val=None, y_val=None):

        train_dataset_iterator = copy(self.dataset_it)
        train_dataset_iterator.fill(X, y)
        self.fit_error = np.zeros((self.n_epochs))

        val_dataset_iterator = None
        if (X_val is not None) and (y_val is not None):
            val_dataset_iterator = copy(self.dataset_it)
            val_dataset_iterator.fill(X_val, y_val)
            self.val_error = np.zeros((self.n_epochs))

        for epoch in range(self.n_epochs):
            loss_train, metric_train = self.train(train_dataset_iterator)

            loss_msg = f"Epoch {epoch + 1:>4d}: Train loss {self.loss_fn}: {loss_train:==7f} "
            metric_msg = f"Epoch {epoch + 1:>4d}: Train metric {self.metric_fn}: {metric_train:==7f} "
            self.fit_error[epoch] = loss_train

            if (X_val is not None) and (y_val is not None):
                loss_val, metric_val = self.test(val_dataset_iterator)

                loss_msg += f" Test loss {self.loss_fn}: {loss_val:==7f} "
                metric_msg += f" Test metric {self.metric_fn}: {metric_val:==7f} "
                self.val_error[epoch] = loss_val

            print(loss_msg)
            print(metric_msg)
            print(len(metric_msg) * "-")
        return self

    def predict(self, X):
        X = Matrix(X)
        y_pred = self.test_forward(X).value
        return y_pred

    def make_constant_layers(self):
        for layer in self.layers:
            layer.make_constant()
        return self

    def __str__(self):
        model_str = ""
        for layer in self.layers:
            model_str += (str(layer) + "\n")
        return model_str


class EncDecTransformerModel(SequenceModel):
    src_emb_layer: EmbeddingLayer
    tgt_emb_layer: EmbeddingLayer
    pos_layer: CosSinPosEncoderLayer
    encoder: TransformerEncoderStack
    decoder: TransformerDecoderStack
    out_layer: Layer
    max_len: int

    def __init__(self, n_epochs, dataset_iterator, loss_function, metric_function, optimizer,
                 src_emb_layer, tgt_emb_layer, pos_layer, encoder, decoder, out_layer, max_len = 1000):
        super().__init__(n_epochs, dataset_iterator, loss_function, metric_function, optimizer)
        self.src_emb_layer = src_emb_layer
        self.tgt_emb_layer = tgt_emb_layer
        self.pos_layer = pos_layer
        self.encoder = encoder
        self.decoder = decoder
        self.out_layer = out_layer
        self.max_len = max_len
        self.layers = [
            self.src_emb_layer,
            self.tgt_emb_layer,
            self.pos_layer,
            self.encoder,
            self.decoder,
            self.out_layer
        ]

    def train_forward(self, X, y, src_mask=None, tgt_mask=None, train_mode = True):
        n_batches = X.shape[0]

        src_emb = self.src_emb_layer.forward(X, train_mode=train_mode)
        src_pos = self.pos_layer.forward(X, train_mode=train_mode)
        src_emb_pos = src_emb + src_pos

        begin = Matrix(np.full(shape=(n_batches, 1), fill_value=TOKEN_BOS), require_grad=False)
        y_shift = Matrix.concat([begin, y[:, :-1]], axis=1)
        tgt_emb = self.tgt_emb_layer.forward(y_shift, train_mode=train_mode)
        tgt_pos = self.pos_layer.forward(y_shift, train_mode=train_mode)
        tgt_emb_pos = tgt_emb + tgt_pos

        context = self.encoder.forward(src_emb_pos, train_mode=train_mode, mask=src_mask)
        output = self.decoder.forward((tgt_emb_pos, context), train_mode=train_mode, self_mask=tgt_mask, cross_mask=src_mask)
        probas = self.out_layer.forward(output)

        self.parameters = []
        self.regularizators = []
        for layer in self.layers:
            self.parameters += layer.get_parameters()
            self.regularizators.append(layer.get_regularizer())
        return probas
    
    def test_forward(self, X, src_mask=None):
        n_batches = X.shape[0]

        src_emb = self.src_emb_layer.forward(X, train_mode=False)
        src_pos = self.pos_layer.forward(X, train_mode=False)
        src_emb_pos = src_emb + src_pos
        context = self.encoder.forward(src_emb_pos, train_mode=False, mask=src_mask)

        y_pred = Matrix(np.full(shape=(n_batches, 1), fill_value=TOKEN_BOS), require_grad=False)
        for i in range(self.max_len):
            tgt_emb = self.tgt_emb_layer.forward(y_pred, train_mode=False)
            tgt_pos = self.pos_layer.forward(y_pred, train_mode=False)
            tgt_emb_pos = tgt_emb + tgt_pos
            output = self.decoder.forward((tgt_emb_pos, context), train_mode=False, cross_mask=src_mask)
            probas = self.out_layer.forward(output)
            new_tokens = Matrix.argmax(probas[:, -1, :], axis=1).reshape((n_batches, 1))
            y_pred = Matrix.concat([y_pred, new_tokens], axis=1)

            if np.all(new_tokens.value == TOKEN_EOS):
                break
        return y_pred
    
    def train(self, dataset_iterator):
        sum_loss_train = 0
        sum_metric_train = 0
    
        for (X_batch, X_mask, y_batch, y_mask) in dataset_iterator():
            self.zero_grad()
            y_pred = self.train_forward(X_batch, y_batch, X_mask, y_mask)

            if y_mask is not None:
                loss = self.loss_fn(y_batch, y_pred, y_mask)
            else:
                loss = self.loss_fn(y_batch, y_pred)
            
            for regularizator in self.regularizators:
                loss += regularizator()
            loss.backward()
            self.optimizer.optimize(self.parameters)
            # print(Matrix.mean(loss))
    
            metric = self.metric_fn(y_batch, y_pred)
    
            sum_loss_train += float(loss.value)
            sum_metric_train += metric
    
        loss_train = sum_loss_train / dataset_iterator.n_batches
        metric_train = sum_metric_train / dataset_iterator.n_batches
    
        return loss_train, metric_train
    
    def test(self, dataset_iterator):
        sum_loss_val = 0
        sum_metric_val = 0
    
        for (X_batch, X_mask, y_batch, y_mask) in dataset_iterator():
            y_pred = self.train_forward(X_batch, y_batch, X_mask, y_mask, train_mode = False)
            y_gen = self.test_forward(X_batch, X_mask)
    
            if y_mask is not None:
                loss = self.loss_fn(y_batch, y_pred, y_mask)
            else:
                loss = self.loss_fn(y_batch, y_pred)
            metric = self.metric_fn(y_batch, y_gen)
    
            sum_loss_val += float(loss.value)
            sum_metric_val += metric
    
        loss_val = sum_loss_val / dataset_iterator.n_batches
        metric_val = sum_metric_val / dataset_iterator.n_batches
    
        return loss_val, metric_val
