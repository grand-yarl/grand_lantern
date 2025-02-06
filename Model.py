import numpy as np
from Matrix import *
from Layers import *
from Loss import *
from Metric import *
from Optimizers import *


class model():
    layers: list[Layer]
    n_epochs: int
    batch_size: int
    loss_fn: Loss
    metric_fn: Metric
    optimizer: Optimizer
    fit_error: np.array
    val_error: np.array
    parameters = []

    def __init__(self, n_epochs, batch_size, loss_function, metric_function, optimizer):
        self.layers = []
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.loss_fn = loss_function
        self.metric_fn = metric_function
        self.optimizer = optimizer
        self.parameters = []
        return

    def add_layer(self, layer):
        self.layers.append(layer)
        return

    def forward(self, input_model):
        self.parameters = []
        current_input = input_model
        for layer in self.layers:
            current_input = layer.forward(current_input)
            self.parameters += layer.get_parameters()
        return current_input

    def batch(self, X, y, iteration):

        if (iteration + 1) * self.batch_size > X.shape[0] - 1:
            batch_slice = slice(iteration * self.batch_size, X.shape[0] - 1)
        else:
            batch_slice = slice(iteration * self.batch_size, (iteration + 1) * self.batch_size)
        X_batch = X[batch_slice][:]
        y_batch = y[batch_slice]
        return X_batch, y_batch

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.local_gradients = []
        return

    def fit(self, X, y, X_val=None, y_val=None):

        X = Matrix(X)
        y = Matrix(y)

        n_batches_train = int(np.ceil(X.shape[0] // self.batch_size))

        self.fit_error = np.zeros((self.n_epochs))
        if (X_val is not None) and (y_val is not None):
            self.val_error = np.zeros((self.n_epochs))

        for epoch in range(self.n_epochs):

            sum_loss_train = 0
            sum_metric_train = 0

            for it in range(n_batches_train):
                self.zero_grad()
                X_batch, y_batch = self.batch(X, y, it)
                y_pred = self.forward(X_batch)

                loss = self.loss_fn(y_batch, y_pred)
                metric = self.metric_fn(y_batch, y_pred)
                gradients = loss.backward()
                self.optimizer.optimize(self.parameters, gradients)
                # print(Matrix.mean(loss))

                sum_loss_train += np.mean(loss.value)
                sum_metric_train += metric

            loss_train = sum_loss_train / n_batches_train
            metric_train = sum_metric_train / n_batches_train

            loss_msg = f"Epoch {epoch + 1:>4d}: Train {self.loss_fn}: {loss_train:==7f} "
            metric_msg = f"Epoch {epoch + 1:>4d}: Train {self.metric_fn}: {metric_train:==7f} "
            self.fit_error[epoch] = loss_train

            if (X_val is not None) and (y_val is not None):

                X_val = Matrix(X_val)
                y_val = Matrix(y_val)

                n_batches_val = int(np.ceil(X_val.shape[0] // self.batch_size))

                sum_loss_val = 0
                sum_metric_val = 0

                for it in range(n_batches_val):
                    self.zero_grad()
                    X_batch, y_batch = self.batch(X, y, it)
                    y_pred = self.forward(X_batch)

                    loss = self.loss_fn(y_batch, y_pred)
                    metric = self.metric_fn(y_batch, y_pred)

                    sum_loss_val += np.mean(loss.value)
                    sum_metric_val += metric

                loss_val = sum_loss_val / n_batches_val
                metric_val = sum_metric_val / n_batches_val

                loss_msg += f" Test {self.loss_fn}: {loss_val:==7f} "
                metric_msg += f" Test {self.metric_fn}: {metric_val:==7f} "
                self.val_error[epoch] = loss_val
            print(loss_msg)
            print(metric_msg)
            print(len(metric_msg) * "-")
        return self

    def predict(self, X):
        X = Matrix(X)
        y_pred = self.forward(X).value
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
