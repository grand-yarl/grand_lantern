from copy import copy
from collections import defaultdict
try:
    import cupy as np
except:
    import numpy as np
import math


class Matrix:

    value: np.ndarray[np.float64]
    grad_: np.ndarray[np.float64]
    local_gradients_: list
    require_grad: bool
    shape: tuple
    ndim: int

    def __init__(self, array, local_gradients=None, require_grad=False):
        if local_gradients is None:
            local_gradients = []
        if isinstance(array, Matrix):
            self.value = array.value
            self.local_gradients_ = array.local_gradients
        else:
            self.value = np.array(array)
            self.local_gradients_ = local_gradients
        self.grad_ = None
        self.shape = self.value.shape
        self.ndim = self.value.ndim
        self.require_grad = require_grad

    @staticmethod
    def _broadcast_gradient(grad, target_shape):
        if grad.shape == target_shape:
            return grad

        while grad.ndim < len(target_shape):
            grad = np.expand_dims(grad, axis=0)

        sum_axes = []
        for axis, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape)):
            if g_dim == 1 and t_dim > 1:
                sum_axes.append(axis)
            elif g_dim != t_dim:
                raise ValueError(f"Wrong shapes: grad.shape={grad.shape}, target_shape={target_shape}")
        if sum_axes:
            grad = np.sum(grad, axis=tuple(sum_axes), keepdims=True)
        return np.broadcast_to(grad, target_shape)

    def __add__(self, other):
        if not isinstance(other, Matrix):
            other = Matrix(other)

        self_val = self.value
        other_val = other.value
        new_value = self_val + other_val
        new_local_gradients = []
        new_require_grad = self.require_grad or other.require_grad

        if self.require_grad:
            def grad_self(grad, self_val = self_val, other_val = None):
                return Matrix._broadcast_gradient(grad, self_val.shape)
            
            new_local_gradients.append((self, grad_self, 'add'))

        if other.require_grad:
            def grad_other(grad, self_val = None, other_val = other_val):
                return Matrix._broadcast_gradient(grad, other_val.shape)
            
            new_local_gradients.append((other, grad_other, 'add'))
        
        return Matrix(new_value, new_local_gradients, new_require_grad)

    def __sub__(self, other):
        if not isinstance(other, Matrix):
            other = Matrix(other)

        self_val = self.value
        other_val = other.value
        new_value = self_val - other_val
        new_local_gradients = []
        new_require_grad = self.require_grad or other.require_grad

        if self.require_grad:
            def grad_self(grad, self_val = self_val, other_val = None):
                return Matrix._broadcast_gradient(grad, self_val.shape)
        
            new_local_gradients.append((self, grad_self, 'sub'))

        if other.require_grad:
            def grad_other(grad, self_val = None, other_val = other_val):
                return Matrix._broadcast_gradient(-grad, other_val.shape)
            
            new_local_gradients.append((other, grad_other, 'sub'))
        
        return Matrix(new_value, new_local_gradients, new_require_grad)

    def __mul__(self, other):
        if not isinstance(other, Matrix):
            other = Matrix(other)

        self_val = self.value
        other_val = other.value
        new_value = self_val * other_val
        new_local_gradients = []
        new_require_grad = self.require_grad or other.require_grad

        if self.require_grad:
            def grad_self(grad, self_val = self_val, other_val = other_val):
                return Matrix._broadcast_gradient(grad * other_val, self_val.shape)
            
            new_local_gradients.append((self, grad_self, 'mul'))

        if other.require_grad:
            def grad_other(grad, self_val = self_val, other_val = other_val):
                return Matrix._broadcast_gradient(grad * self_val, other_val.shape)
            
            new_local_gradients.append((other, grad_other, 'mul'))
        
        return Matrix(new_value, new_local_gradients, new_require_grad)

    def __truediv__(self, other):
        if not isinstance(other, Matrix):
            other = Matrix(other)

        self_val = self.value
        other_val = other.value
        new_value = self_val / other_val
        new_local_gradients = []
        new_require_grad = self.require_grad or other.require_grad
            
        if self.require_grad:
            def grad_self(grad, self_val = self_val, other_val = other_val):
                new_grad = grad / other_val
                return Matrix._broadcast_gradient(new_grad, self_val.shape)
            
            new_local_gradients.append((self, grad_self, 'div'))

        if other.require_grad:
            def grad_other(grad, self_val = self_val, other_val = other_val):
                new_grad = -grad * self_val / (other_val ** 2)
                return Matrix._broadcast_gradient(new_grad, other_val.shape)
            
            new_local_gradients.append((other, grad_other, 'div'))

        return Matrix(new_value, new_local_gradients, new_require_grad)

    def __radd__(self, other):
        if not isinstance(other, Matrix):
            other = Matrix(other)
        return other + self

    def __rsub__(self, other):
        if not isinstance(other, Matrix):
            other = Matrix(other)
        return other - self

    def __rmul__(self, other):
        if not isinstance(other, Matrix):
            other = Matrix(other)
        return other * self

    def __rtruediv__(self, other):
        if not isinstance(other, Matrix):
            other = Matrix(other)
        return other / self

    def __neg__(self):
        self_val = self.value
        new_value = -self.value
        new_local_gradients = []
        new_require_grad = self.require_grad

        if self.require_grad:
            def grad_self(grad, self_val = self_val):
                return Matrix._broadcast_gradient(-grad, self_val.shape)
            
            new_local_gradients.append((self, grad_self, 'neg'))

        return Matrix(new_value, new_local_gradients, new_require_grad)

    def __matmul__(self, other):
        if not isinstance(other, Matrix):
            other = Matrix(other)

        self_val = self.value
        other_val = other.value
        new_value = self_val @ other_val
        new_local_gradients = []
        new_require_grad = self.require_grad or other.require_grad

        if self.require_grad:
            def grad_self(grad, self_val = self_val, other_val = other_val):
                new_grad = grad @ np.moveaxis(other_val, -1, -2)
                return Matrix._broadcast_gradient(new_grad, self_val.shape)
            
            new_local_gradients.append((self, grad_self, 'matmul'))

        if other.require_grad:
            def grad_other(grad, self_val = self_val, other_val = other_val):
                new_grad = np.moveaxis(self_val, -1, -2) @ grad
                return Matrix._broadcast_gradient(new_grad, other_val.shape)
            
            new_local_gradients.append((other, grad_other, 'matmul'))

        return Matrix(new_value, new_local_gradients, new_require_grad)

    def __rmatmul__(self, other):
        if not isinstance(other, Matrix):
            other = Matrix(other)
        return other @ self

    def __pow__(self, power):
        self_val = self.value
        new_value = self_val ** power
        new_local_gradients = []
        new_require_grad = self.require_grad

        if self.require_grad:
            def grad_self(grad, self_val = self_val, power = power):
                new_grad = grad * power * (self_val ** (power - 1))
                return Matrix._broadcast_gradient(new_grad, self_val.shape)
            
            new_local_gradients.append((self, grad_self, 'pow'))
        return Matrix(new_value, new_local_gradients, new_require_grad)

    def __abs__(self):
        self_val = self.value
        new_value = np.abs(self_val)
        new_local_gradients = []
        new_require_grad = self.require_grad

        if self.require_grad:
            def grad_self(grad, self_val = self_val):
                new_grad = grad * np.sign(self_val)
                return Matrix._broadcast_gradient(new_grad, self_val.shape)
            
            new_local_gradients.append((self, grad_self, 'abs'))
        return Matrix(new_value, new_local_gradients, new_require_grad)

    def reshape(self, shape):
        self_val = self.value
        new_value = self_val.reshape(shape)
        new_local_gradients = []
        new_require_grad = self.require_grad

        if self.require_grad:
            def grad_self(grad, self_val = self_val):
                new_grad = grad.reshape(self_val.shape)
                return new_grad
            
            new_local_gradients.append((self, grad_self, 'reshape'))
        return Matrix(new_value, new_local_gradients, new_require_grad)

    def transpose(self):
        self_val = self.value
        new_value = np.moveaxis(self_val, -1, -2)
        new_local_gradients = []
        new_require_grad = self.require_grad

        if self.require_grad:
            def grad_self(grad, self_val = self_val):
                new_grad = np.moveaxis(grad, -1, -2)
                return new_grad
            
            new_local_gradients.append((self, grad_self, 'transpose'))
        return Matrix(new_value, new_local_gradients, new_require_grad)

    def T(self):
        return self.transpose()

    @classmethod
    def zeros(cls, shape, require_grad=False):
        new_value = np.zeros(shape)
        return Matrix(new_value, require_grad=require_grad)

    @classmethod
    def ones(cls, shape, require_grad=False):
        new_value = np.ones(shape)
        return Matrix(new_value, require_grad=require_grad)

    @classmethod
    def normal(cls, shape, mean=0, std=1, require_grad=False):
        new_value = np.random.normal(loc=mean, scale=std, size=shape)
        return Matrix(new_value, require_grad=require_grad)

    @classmethod
    def uniform(cls, shape, low=-1, high=1, require_grad=False):
        new_value = np.random.uniform(low=low, high=high, size=shape)
        return Matrix(new_value, require_grad=require_grad)

    @classmethod
    def stack(cls, stack_list, axis=0):
        full_shape = list(stack_list[0].shape)
        full_shape.insert(axis, len(stack_list))
        new_shape = list(stack_list[0].shape)
        new_shape.insert(axis, 1)
        stacked = Matrix.zeros(shape=full_shape)
        for i in range(len(stack_list)):

            slices = []
            for j in range(stack_list[0].ndim):
                slices.append(slice(None))
            slices.insert(axis, slice(i, i + 1, None))

            stacked[tuple(slices)] = stack_list[i].reshape(shape=new_shape)
        return stacked

    @classmethod
    def sin(cls, obj):
        new_value = np.sin(obj.value)
        new_local_gradients = []
        new_require_grad = obj.require_grad

        if obj.require_grad:
            new_local_gradients.append((obj, lambda x: x * np.cos(obj.value), 'sin'))
        return Matrix(new_value, new_local_gradients, new_require_grad)

    @classmethod
    def cos(cls, obj):
        new_value = np.cos(obj.value)
        new_local_gradients = []
        new_require_grad = obj.require_grad

        if obj.require_grad:
            new_local_gradients.append((obj, lambda x: -x * np.sin(obj.value), 'cos'))
        return Matrix(new_value, new_local_gradients, new_require_grad)

    @classmethod
    def exp(cls, obj):
        new_value = np.exp(obj.value)
        new_local_gradients = []
        new_require_grad = obj.require_grad

        if obj.require_grad:
            new_local_gradients.append((obj, lambda x: x * new_value, 'exp'))
        return Matrix(new_value, new_local_gradients, new_require_grad)

    @classmethod
    def log(cls, obj):
        new_value = np.log(obj.value)
        new_local_gradients = []
        new_require_grad = obj.require_grad

        if obj.require_grad:
            new_local_gradients.append((obj, lambda x: x / obj.value, 'log'))
        return Matrix(new_value, new_local_gradients, new_require_grad)

    @classmethod
    def sqrt(cls, obj):
        new_value = np.sqrt(obj.value)
        new_local_gradients = []
        new_require_grad = obj.require_grad

        if obj.require_grad:
            new_local_gradients.append((obj, lambda x: x / (2 * new_value + 10e-5), 'sqrt'))
        return Matrix(new_value, new_local_gradients, new_require_grad)

    @classmethod
    def sum(cls, obj, axis=None, keepdims=False):
        if not keepdims:
            if axis is not None:
                new_value = np.sum(obj.value, axis=axis)
                new_local_gradients = []
                new_require_grad = obj.require_grad

                if obj.require_grad:
                    new_local_gradients.append(
                        (obj, lambda x: np.expand_dims(np.array(x), axis=axis) * np.ones_like(obj.value), 'sum')
                    )
                return Matrix(new_value, new_local_gradients, new_require_grad)

            else:
                new_value = np.sum(obj.value, axis=axis)
                new_local_gradients = []
                new_require_grad = obj.require_grad

                if obj.require_grad:
                    new_local_gradients.append(
                        (obj, lambda x: x * np.ones_like(obj.value), 'sum')
                    )
                return Matrix(new_value, new_local_gradients, new_require_grad)

        else:
            new_value = np.sum(obj.value, axis=axis, keepdims=True) * np.ones_like(obj.value)
            new_local_gradients = []
            new_require_grad = obj.require_grad

            if obj.require_grad:
                new_local_gradients.append(
                    (obj, lambda x: x, 'sum')
                )
        return Matrix(new_value, new_local_gradients, new_require_grad)

    @classmethod
    def mean(cls, obj, axis=None, keepdims=False):
        if axis is not None:
            n_elements = obj.shape[axis]
        else:
            n_elements = math.prod(obj.shape)
        return Matrix.sum(obj, axis=axis, keepdims=keepdims) / n_elements

    @classmethod
    def std(cls, obj, axis=None, keepdims=False):
        mean = Matrix.mean(obj, axis=axis, keepdims=keepdims)
        sub = obj - mean
        square_sub = sub ** 2
        sum_square = Matrix.mean(square_sub, axis=axis, keepdims=keepdims)
        std = Matrix.sqrt(sum_square)
        return std

    @classmethod
    def sigmoid(cls, obj):
        new_value = 1 / (1 + np.exp(-1 * obj.value))
        new_local_gradients = []
        new_require_grad = obj.require_grad

        if obj.require_grad:
            new_local_gradients.append((obj, lambda x: x * new_value * (1 - new_value), 'sigmoid'))
        return Matrix(new_value, new_local_gradients, new_require_grad)

    @classmethod
    def sign(cls, obj):
        new_value = np.sign(obj.value)
        new_require_grad = obj.require_grad
        return Matrix(new_value, require_grad=new_require_grad)

    @classmethod
    def relu(cls, obj):
        zeros = np.zeros(obj.shape)
        new_value = np.maximum(zeros, obj.value)
        new_local_gradients = []
        new_require_grad = obj.require_grad

        if obj.require_grad:
            new_local_gradients.append((obj, lambda x: x * np.sign(new_value), 'relu'))
        return Matrix(new_value, new_local_gradients, new_require_grad)

    @classmethod
    def lrelu(cls, obj, alpha):
        new_value = np.maximum(alpha * obj.value, obj.value)
        new_local_gradients = []
        new_require_grad = obj.require_grad

        if obj.require_grad:
            sign = np.sign(new_value)
            sign[sign == 0] = alpha
            new_local_gradients.append((obj, lambda x: x * sign, 'lrelu'))
        return Matrix(new_value, new_local_gradients, new_require_grad)

    @classmethod
    def tanh(cls, obj):
        new_value = np.tanh(obj.value)
        new_local_gradients = []
        new_require_grad = obj.require_grad

        if obj.require_grad:
            new_local_gradients.append((obj, lambda x: x * (1 + new_value) * (1 - new_value), 'tanh'))
        return Matrix(new_value, new_local_gradients, new_require_grad)

    @classmethod
    def softmax(cls, obj, axis=-1):

        def compute_gradient(grad):
            prime = np.zeros(obj.shape)
            for i in range(obj.shape[0]):
                s = np.array([new_value[i]])
                g = np.array([grad[i]])
                prime[i] = -s * np.sum(s * g, axis=1) + s * g
            return prime

        new_value = np.exp(obj.value) / np.sum(np.exp(obj.value), axis=axis, keepdims=True)
        new_local_gradients = []
        new_require_grad = obj.require_grad

        if obj.require_grad:
            new_local_gradients.append((obj, lambda x: compute_gradient(x), 'softmax'))
        return Matrix(new_value, new_local_gradients, new_require_grad)

    @classmethod
    def safe_softmax(cls, obj, axis=-1):
        sub = Matrix(obj.value.max(axis=1, keepdims=True))
        return Matrix.softmax(obj - sub, axis=axis)

    @classmethod
    def conv2d(cls, matrix, kernel, dilation):

        def convolution(matrix, kernel, dilation):
            x_steps = (matrix.shape[2] - kernel.shape[2] + 1) // dilation[0]
            y_steps = (matrix.shape[3] - kernel.shape[3] + 1) // dilation[1]
            conv = np.zeros((matrix.shape[0], kernel.shape[1], x_steps, y_steps))
            for i in range(x_steps):
                for j in range(y_steps):
                    for c in range(kernel.shape[1]):
                        x_slice = slice(i * dilation[0], i * dilation[0] + kernel.shape[2])
                        y_slice = slice(j * dilation[1], j * dilation[1] + kernel.shape[3])
                        conv[:, c, i, j] = np.sum(matrix[:, :, x_slice, y_slice] * kernel[:, c, :, :])
            return conv

        def grad_dilate(grad, dilation):
            new_grad = np.zeros(
                (grad.shape[0], grad.shape[1], dilation[0] * grad.shape[2], dilation[1] * grad.shape[3]))
            new_grad[:, :, ::dilation[0], ::dilation[1]] = grad
            return new_grad

        def grad_pad(grad, matrix):
            pad_x = matrix.shape[2] - grad.shape[2]
            pad_y = matrix.shape[3] - grad.shape[3]
            return np.pad(grad, ((0, 0), (0, 0), (pad_x, pad_x), (pad_y, pad_y)))

        new_value = convolution(matrix.value, kernel.value, dilation)
        new_local_gradients = []
        new_require_grad = matrix.require_grad or kernel.require_grad

        if matrix.require_grad:
            new_local_gradients.append(
                (matrix, lambda x: convolution(
                    grad_pad(grad_dilate(x, dilation), matrix),
                    kernel.value[:, :, ::-1, ::-1].reshape(kernel.shape[1], kernel.shape[0], kernel.shape[2],
                                                           kernel.shape[3]),
                    dilation=(1, 1))[:, :, ::-1, ::-1],
                 'conv2d')
            )
        if kernel.require_grad:
            new_local_gradients.append(
                (kernel, lambda x: convolution(
                    matrix.value.reshape(matrix.shape[1], matrix.shape[0], matrix.shape[2], matrix.shape[3]),
                    grad_dilate(x[:, :, ::-1, ::-1], dilation),
                    dilation=(1, 1)),
                 'conv2d')
            )
        return Matrix(new_value, new_local_gradients, new_require_grad)

    def backward(self, verbose=False):

        gradients = defaultdict(lambda: 0)

        def compute_gradients(matrix, before_grads):
            if matrix.local_gradients:
                for (child, child_gradients_func, operation) in matrix.local_gradients:
                    if verbose:
                        print(f"{child} - {child_gradients_func} - {operation}")
                    new_child_grad = child_gradients_func(before_grads)
                    compute_gradients(child, new_child_grad)
                    gradients[child] += new_child_grad

        compute_gradients(self, np.ones(self.shape))

        return gradients

    def __getitem__(self, idx):
        def compute_gradient(grad, target):
            target_grad = np.zeros((target.shape))
            target_grad[idx] = grad
            return target_grad

        new_value = self.value[idx]
        new_local_gradients = []
        new_require_grad = self.require_grad

        if self.require_grad:
            new_local_gradients.append((self, lambda x: compute_gradient(x, self), 'getitem'))

        return Matrix(new_value, new_local_gradients, new_require_grad)

    def __setitem__(self, key, item):
        if isinstance(item, Matrix):
            self.value[key] = item.value
            if item.require_grad:
                self.require_grad = True
                if self.value[key].ndim == self.value.ndim:
                    self.local_gradients.append((item, lambda x: x[key], 'setitem'))
                else:
                    self.local_gradients.append((item, lambda x: x[key].reshape(1, -1), 'setitem'))
        else:
            self.value = np.array(item)
        return

    def __repr__(self):
        return f"{self.value}"

    def __str__(self):
        return f"{self.value}"
