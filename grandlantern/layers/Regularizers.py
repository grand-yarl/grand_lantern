import numpy as np
from grandlantern.matrix.Matrix import Matrix


class BaseRegularizer:
    parameters: list

    def __init__(self):
        self.parameters = []
        return

    def define_params(self, parameters):
        self.parameters = parameters
        return

    def __call__(self):
        term = 0
        return term

    def __str__(self):
        return f"None"


class L1Regularizer(BaseRegularizer):
    l1: float

    def __init__(self, l1):
        super().__init__()
        self.l1 = l1
        return

    def __call__(self):
        term = 0
        for param in self.parameters:
            term += self.l1 * Matrix.sum(abs(param))
        return term

    def __str__(self):
        return f"L1 regularizer with coefficient {self.l1}"


class L2Regularizer(BaseRegularizer):
    l2: float

    def __init__(self, l2):
        super().__init__()
        self.l2 = l2
        return

    def __call__(self):
        term = 0
        for param in self.parameters:
            term += self.l2 * Matrix.sum(param ** 2)
        return term

    def __str__(self):
        return f"L2 regularizer with coefficient {self.l2}"


class ElasticNetRegularizer(L1Regularizer, L2Regularizer):
    
    def __init__(self, l1, l2):
        L1Regularizer.__init__(self, l1=l1)
        L2Regularizer.__init__(self, l2=l2)

    def __call__(self):
        term = 0
        for param in self.parameters:
            term += self.l1 * Matrix.sum(abs(param)) + self.l2 * Matrix.sum(param ** 2)
        return term

    def __str__(self):
        return f"Elastic Net regularizer with coefficient {self.l1} and {self.l2}"
