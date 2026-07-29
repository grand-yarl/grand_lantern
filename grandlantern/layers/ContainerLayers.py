from copy import deepcopy
import numpy as np
from grandlantern.matrix import Matrix
from grandlantern.layers import Layer


class SequenceLayer(Layer):
    layers: list[Layer]

    def __init__(self, layers):
        super().__init__()
        self.layers = layers
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
        layer_str = "Sequence Layer with sublayers : \n"
        for layer in self.layers:
            layer_str += f"{str(layer)} \n"
        return layer_str
    

class SkipConnectionLayer(Layer):
    layer: Layer

    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        return

    def forward(self, X, train_mode):
        before = X
        after = self.layer.forward(X, train_mode)
        out = before + after

        self.parameters = self.layer.get_parameters()
        self.regularizer.define_params(self.parameters)
        return out

    def __str__(self):
        layer_str = "Skip Connection Layer with sublayer : \n\t {self.layer} \n"
        return layer_str