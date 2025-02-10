from grand_lantern.layers.Layers import *
from grand_lantern.metrics import *
from grand_lantern.optimizers import *
from grand_lantern.dataiterators.DatasetIterators import *


class model():
    layers: list[Layer]
    n_epochs: int
    dataset_it: DatasetIterator
    loss_fn: Loss
    metric_fn: Metric
    opimizer: Optimizer
    fit_error: np.array
    val_error: np.array
    parameters = []

    def __init__(self, n_epochs, dataset_iterator, loss_function, metric_function, optimizer):
        self.layers = []
        self.n_epochs = n_epochs
        self.dataset_it = dataset_iterator
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

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.local_gradients = []
        return

    def fit(self, X, y, X_val=None, y_val=None):

        train_dataset_iterator = copy(self.dataset_it)
        train_dataset_iterator.fill(X, y)
        n_batches_train = train_dataset_iterator.n_batches
        self.fit_error = np.zeros((self.n_epochs))

        val_dataset_iterator = None
        n_batches_val = 0
        if (X_val is not None) and (y_val is not None):
            val_dataset_iterator = copy(self.dataset_it)
            val_dataset_iterator.fill(X_val, y_val)
            n_batches_val = val_dataset_iterator.n_batches
            self.val_error = np.zeros((self.n_epochs))

        for epoch in range(self.n_epochs):

            sum_loss_train = 0
            sum_metric_train = 0

            for (X_batch, y_batch) in train_dataset_iterator():
                self.zero_grad()
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

                sum_loss_val = 0
                sum_metric_val = 0

                for (X_batch, y_batch) in val_dataset_iterator():
                    self.zero_grad()
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
