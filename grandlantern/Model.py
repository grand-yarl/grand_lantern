import numpy as np
from copy import copy
from .matrix import Matrix
from .layers import Layer
from .metrics import Metric, Loss
from .optimizers import Optimizer
from .dataiterators import DatasetIterator
from .layers import BaseRegularizer


class model():
    layers: list[Layer]
    n_epochs: int
    dataset_it: DatasetIterator
    loss_fn: Loss
    metric_fn: Metric
    opimizer: Optimizer
    fit_error: np.array
    val_error: np.array
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
            parameter.local_gradients = []
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
            gradients = loss.backward()
            self.optimizer.optimize(self.parameters, gradients)
            # print(Matrix.mean(loss))

            metric = self.metric_fn(y_batch, y_pred)

            sum_loss_train += np.mean(loss.value)
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

            sum_loss_val += np.mean(loss.value)
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

            loss_msg = f"Epoch {epoch + 1:>4d}: Train {self.loss_fn}: {loss_train:==7f} "
            metric_msg = f"Epoch {epoch + 1:>4d}: Train {self.metric_fn}: {metric_train:==7f} "
            self.fit_error[epoch] = loss_train

            if (X_val is not None) and (y_val is not None):
                loss_val, metric_val = self.test(val_dataset_iterator)

                loss_msg += f" Test {self.loss_fn}: {loss_val:==7f} "
                metric_msg += f" Test {self.metric_fn}: {metric_val:==7f} "
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
