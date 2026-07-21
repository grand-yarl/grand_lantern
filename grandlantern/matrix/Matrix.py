from copy import copy, deepcopy
from collections import defaultdict
"""try:
    import cupy as np
except:
"""
import numpy as np
import math

EPS = 10e-10

class Matrix:

    value: np.ndarray
    grad_: np.ndarray
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
            return grad.copy()
        if grad.ndim == 0:
            return np.full(target_shape, grad)

        # Удаляем все оси размерности 1 (squeeze)
        grad = grad.squeeze()
        if grad.shape == target_shape:
            return grad.copy()

        # Если осей больше, чем нужно, суммируем по первым лишним (слева)
        if grad.ndim > len(target_shape):
            axes_to_sum = tuple(range(grad.ndim - len(target_shape)))
            grad = np.sum(grad, axis=axes_to_sum)
            if grad.shape == target_shape:
                return grad.copy()

        # Если осей меньше, добавляем оси СПРАВА (в конец)
        while grad.ndim < len(target_shape):
            grad = np.expand_dims(grad, axis=-1)

        # Пытаемся применить broadcasting
        try:
            return np.broadcast_to(grad, target_shape).copy()
        except ValueError:
            # Если broadcast не удался, суммируем по осям, где grad > 1, target == 1
            sum_axes = []
            for axis, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape)):
                if g_dim == t_dim:
                    continue
                elif g_dim > 1 and t_dim == 1:
                    sum_axes.append(axis)
                elif g_dim == 1 and t_dim > 1:
                    continue
                else:
                    raise ValueError(f"Wrong shapes: grad.shape={grad.shape}, target_shape={target_shape}")
            if sum_axes:
                grad = np.sum(grad, axis=tuple(sum_axes), keepdims=True)
            return np.broadcast_to(grad, target_shape).copy()

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
            def grad_self(grad, self_val=self_val, other_val=other_val):
                if self_val.ndim == 1:
                    new_grad = grad @ other_val.T
                else:
                    new_grad = grad @ np.moveaxis(other_val, -1, -2)
                return Matrix._broadcast_gradient(new_grad, self_val.shape)
            
            new_local_gradients.append((self, grad_self, 'matmul'))

        if other.require_grad:
            def grad_other(grad, self_val=self_val, other_val=other_val):
                if self_val.ndim == 1:
                    new_grad = np.outer(self_val, grad)
                else:
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
        if not stack_list:
            raise ValueError("concat_list cannot be empty")
        
        full_shape = list(stack_list[0].shape)
        full_shape.insert(axis, len(stack_list))
        new_shape = list(stack_list[0].shape)
        new_shape.insert(axis, 1)
        new_value = np.zeros(shape=full_shape)

        new_local_gradients = []
        new_require_grad = False
        for i in range(len(stack_list)):
            obj_val = stack_list[i].value

            slices = []
            for j in range(stack_list[0].ndim):
                slices.append(slice(None))
            slices.insert(axis, slice(i, i + 1, None))
            tuple_slices = tuple(slices)

            new_value[tuple_slices] = obj_val.reshape(new_shape)

            if (stack_list[i].require_grad):
                def grad_self(grad, obj_val = obj_val, tuple_slices=tuple_slices):
                    new_grad = grad[tuple_slices]
                    return Matrix._broadcast_gradient(new_grad, obj_val.shape)

                new_require_grad = True
                new_local_gradients.append((stack_list[i], grad_self, 'stack'))

        return Matrix(new_value, new_local_gradients, new_require_grad)
    
    @classmethod
    def concat(cls, concat_list, axis=-1):
        if not concat_list:
            raise ValueError("concat_list cannot be empty")

        ndim = concat_list[0].ndim
        if axis < 0:
            axis = ndim + axis

        for t in concat_list:
            if t.ndim != ndim:
                raise ValueError(f"All tensors must have same ndim: {ndim} vs {t.ndim}")
            for ax in range(ndim):
                if ax != axis and t.shape[ax] != concat_list[0].shape[ax]:
                    raise ValueError(f"Incompatible shapes at axis {ax}: {t.shape} vs {concat_list[0].shape}")

        values = [t.value for t in concat_list]
        new_value = np.concatenate(values, axis=axis)

        new_local_gradients = []
        new_require_grad = False
        start = 0
        for t in concat_list:
            if t.require_grad:
                def grad_obj(grad, obj_val = t, axis=axis, start=start):
                    slices = [slice(None)] * grad.ndim
                    slices[axis] = slice(start, start + obj_val.shape[axis])
                    new_grad = grad[tuple(slices)]
                    return Matrix._broadcast_gradient(new_grad, obj_val.shape)
                
                new_local_gradients.append((t, grad_obj, 'concat'))
                new_require_grad = True
            start += t.shape[axis]

        return Matrix(new_value, new_local_gradients, new_require_grad)

    @classmethod
    def sin(cls, obj):
        obj_val = obj.value
        new_value = np.sin(obj_val)
        new_local_gradients = []
        new_require_grad = obj.require_grad

        if obj.require_grad:
            def grad_obj(grad, obj_val=obj_val):
                new_grad = grad * np.cos(obj_val)
                return Matrix._broadcast_gradient(new_grad, obj_val.shape)
            
            new_local_gradients.append((obj, grad_obj, 'sin'))

        return Matrix(new_value, new_local_gradients, new_require_grad)

    @classmethod
    def cos(cls, obj):
        obj_val = obj.value
        new_value = np.cos(obj_val)
        new_local_gradients = []
        new_require_grad = obj.require_grad

        if obj.require_grad:
            def grad_obj(grad, obj_val=obj_val):
                new_grad = -grad * np.sin(obj_val)
                return Matrix._broadcast_gradient(new_grad, obj_val.shape)
            
            new_local_gradients.append((obj, grad_obj, 'cos'))

        return Matrix(new_value, new_local_gradients, new_require_grad)

    @classmethod
    def exp(cls, obj):
        obj_val = obj.value
        new_value = np.exp(obj_val)
        new_local_gradients = []
        new_require_grad = obj.require_grad

        if obj.require_grad:
            def grad_obj(grad, obj_val=obj_val):
                new_grad = grad * np.exp(obj_val)
                return Matrix._broadcast_gradient(new_grad, obj_val.shape)
            
            new_local_gradients.append((obj, grad_obj, 'exp'))

        return Matrix(new_value, new_local_gradients, new_require_grad)

    @classmethod
    def log(cls, obj):
        obj_val = obj.value
        new_value = np.log(obj_val)
        new_local_gradients = []
        new_require_grad = obj.require_grad

        if obj.require_grad:
            def grad_obj(grad, obj_val=obj_val):
                new_grad = grad / obj_val
                return Matrix._broadcast_gradient(new_grad, obj_val.shape)
            
            new_local_gradients.append((obj, grad_obj, 'log'))

        return Matrix(new_value, new_local_gradients, new_require_grad)

    @classmethod
    def sqrt(cls, obj):
        obj_val = obj.value
        new_value = np.sqrt(obj.value)
        new_local_gradients = []
        new_require_grad = obj.require_grad

        if obj.require_grad:
            def grad_obj(grad, obj_val=obj_val):
                new_grad = grad / (2 * np.sqrt(obj_val) + EPS)
                return Matrix._broadcast_gradient(new_grad, obj_val.shape)
            
            new_local_gradients.append((obj, grad_obj, 'sqrt'))
        return Matrix(new_value, new_local_gradients, new_require_grad)

    @classmethod
    def sum(cls, obj, axis=None, keepdims=False):
        obj_val = obj.value
        new_value = np.sum(obj_val, axis=axis, keepdims=keepdims)
        new_local_gradients = []
        new_require_grad = obj.require_grad

        if obj.require_grad:
            if keepdims:
                def grad_obj(grad, obj_val=obj_val):
                    return Matrix._broadcast_gradient(grad, obj_val.shape)
                
            else:
                def grad_obj(grad, obj_val=obj_val, axis=axis):
                    if axis is None:
                        return  grad * np.ones_like(obj_val)
                    else:
                        if isinstance(axis, int):
                            axis = (axis,)
                        for ax in sorted(axis):
                            grad = np.expand_dims(grad, axis=ax)
                        return np.broadcast_to(grad, obj_val.shape)
                    
            new_local_gradients.append((obj, grad_obj, 'sum'))

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
        # Всегда используем keepdims=True для промежуточных операций
        mean = Matrix.mean(obj, axis=axis, keepdims=True)
        sub = obj - mean
        square_sub = sub ** 2
        sum_square = Matrix.mean(square_sub, axis=axis, keepdims=True)
        std = Matrix.sqrt(sum_square)
        
        if not keepdims and axis is not None:
            # Убираем оси, по которым усредняли
            axes = axis if isinstance(axis, (tuple, list)) else (axis,)
            new_shape = list(std.shape)
            for ax in sorted(axes, reverse=True):
                del new_shape[ax]
            std = std.reshape(new_shape)
        
        return std

    @classmethod
    def sigmoid(cls, obj):
        obj_val = obj.value
        new_value = 1 / (1 + np.exp(-1 * obj.value))
        new_local_gradients = []
        new_require_grad = obj.require_grad

        if obj.require_grad:
            def grad_obj(grad, obj_val = obj_val, new_val=new_value):
                new_grad = grad * new_val * (1 - new_val)
                return Matrix._broadcast_gradient(new_grad, obj_val.shape)
            
            new_local_gradients.append((obj, grad_obj, 'sigmoid'))
        return Matrix(new_value, new_local_gradients, new_require_grad)

    @classmethod
    def sign(cls, obj):
        obj_val = obj.value
        new_value = np.sign(obj_val)
        new_local_gradients = []
        new_require_grad = obj.require_grad

        if obj.require_grad:
            def grad_obj(grad, obj_val = obj_val):
                new_grad = np.zeros((obj_val.shape))
                return new_grad
            
            new_local_gradients.append((obj, grad_obj, 'sign'))
            
        return Matrix(new_value, require_grad=new_require_grad)

    @classmethod
    def relu(cls, obj):
        obj_val = obj.value
        zeros = np.zeros(obj.shape)
        new_value = np.maximum(zeros, obj_val)
        new_local_gradients = []
        new_require_grad = obj.require_grad

        if obj.require_grad:
            def grad_obj(grad, obj_val = obj_val, new_val=new_value):
                new_grad = grad * np.sign(new_val)
                return Matrix._broadcast_gradient(new_grad, obj_val.shape)
            
            new_local_gradients.append((obj, grad_obj, 'relu'))

        return Matrix(new_value, new_local_gradients, new_require_grad)

    @classmethod
    def lrelu(cls, obj, alpha):
        obj_val = obj.value
        new_value = np.maximum(alpha * obj_val, obj_val)
        new_local_gradients = []
        new_require_grad = obj.require_grad

        if obj.require_grad:
            def grad_obj(grad, obj_val = obj_val, new_val=new_value):
                sign = np.sign(new_val)
                sign[sign == 0] = alpha
                new_grad = grad * sign
                return Matrix._broadcast_gradient(new_grad, obj_val.shape)
            
            new_local_gradients.append((obj, grad_obj, 'lrelu'))

        return Matrix(new_value, new_local_gradients, new_require_grad)

    @classmethod
    def tanh(cls, obj):
        obj_val = obj.value
        new_value = np.tanh(obj_val)
        new_local_gradients = []
        new_require_grad = obj.require_grad

        if obj.require_grad:
            def grad_obj(grad, obj_val = obj_val, new_val=new_value):
                new_grad = grad * (1 + new_val) * (1 - new_val)
                return Matrix._broadcast_gradient(new_grad, obj_val.shape)
            
            new_local_gradients.append((obj, grad_obj, 'tanh'))
            
        return Matrix(new_value, new_local_gradients, new_require_grad)

    @classmethod
    def softmax(cls, obj, axis=-1):
        obj_val = obj.value
        new_value = np.exp(obj_val) / np.sum(np.exp(obj_val), axis=axis, keepdims=True)
        new_local_gradients = []
        new_require_grad = obj.require_grad

        if obj.require_grad:
            def grad_obj(grad, obj_val = obj_val, new_val=new_value):
                new_grad = np.zeros(obj_val.shape)
                for i in range(obj_val.shape[0]):
                    s = np.array([new_val[i]])
                    g = np.array([grad[i]])
                    new_grad[i] = -s * np.sum(s * g, axis=1) + s * g
                return Matrix._broadcast_gradient(new_grad, obj_val.shape)
            
            new_local_gradients.append((obj, grad_obj, 'softmax'))
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
        self.grad_ = np.ones_like(self.value)
        stack = [(self, np.ones_like(self.value))]
        while stack:
            node, current_grad = stack.pop()
            grad_accum = {}  # parent -> суммарный градиент от текущего узла
            for parent, grad_fn, operation in node.local_gradients_:
                if verbose:
                    print(f"Calculate {operation} gradient for {parent} from {node} ...")
                parent_grad = grad_fn(current_grad)
                if parent in grad_accum:
                    grad_accum[parent] += parent_grad
                else:
                    grad_accum[parent] = parent_grad
            for parent, parent_grad in grad_accum.items():
                if parent.grad_ is None:
                    parent.grad_ = parent_grad
                else:
                    parent.grad_ = parent.grad_ + parent_grad
                if verbose:
                    print(f"Gradient is {parent_grad}")
                    print(f"Pushing ({parent}, {parent_grad}) to stack")
                stack.append((parent, parent_grad))
        return

    def __getitem__(self, idx):
        self_val = self.value
        new_value = self.value[idx]
        new_local_gradients = []
        new_require_grad = self.require_grad

        if self.require_grad:
            def grad_self(grad, self_val = self_val, idx = idx):
                new_grad = np.zeros((self_val.shape))
                new_grad[idx] = grad
                return new_grad
            
            new_local_gradients.append((self, grad_self, 'getitem'))

        return Matrix(new_value, new_local_gradients, new_require_grad)

    def __setitem__(self, key, item):
        if isinstance(item, Matrix):
            self.value[key] = item.value
            if item.require_grad:
                raise AttributeError("No gradient operation is allowed")
        else:
            self.value[key] = np.array(item)
        return

    def __repr__(self):
        return f"{self.value}"

    def __str__(self):
        return f"{self.value}"
