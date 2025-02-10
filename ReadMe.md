# This is Grand Lantern neural network library 🏮
The main purpose of this library is to understand - what is happening under the hood of popular deep learning frameworks.
Also, it can be used to build your own ideas - no special knowledge for thst you need! 

This library is written only using numpy (maybe cupy in future), you can see every calculation in neural network!

## How to install ⛏️

Put just this command to terminal

```commandline
pip install grand_lantern
```

## Get started 🚀

### 1. Import library

```python
import grandlantern as gl
```

### 2. Define Data Iterator

```python
from grandlantern.dataiterators import DatasetIterator

batch_size = 100
my_dataset_iterator = DatasetIterator(dataset=gl.TableDataset(), batch_size=batch_size)
```

### 3. Choose optimizer

```python
from grandlantern.optimizers import SGD

my_optimizer = SGD(learning_rate=0.01)
```

### 4. Build model

```python
from grandlantern import model
from grandlantern.metrics import CrossEntropy, Accuracy

NN = model(n_epochs=100,
           dataset_iterator=my_dataset_iterator,
           loss_function=CrossEntropy(),
           metric_function=Accuracy(),
           optimizer=my_optimizer)
```

### 5. Add layers

```python
from grandlantern.layers import LinearLayer
from grandlantern.layers.Activation import Sigmoid, ReLU, SoftMax

NN.add_layer(LinearLayer(n_neurons=100, activation=Sigmoid(), biased=True))
NN.add_layer(LinearLayer(n_neurons=50, activation=ReLU(), biased=True))
NN.add_layer(LinearLayer(n_neurons=10, activation=SoftMax(), biased=True))
```

### 6. Train model

```python
NN.fit(X_train, y_train.reshape(-1, 1), X_test, y_test.reshape(-1, 1))
```

### 7. Use model

```python
y_pred = NN.predict(X_test)
```

## What is supported now ✅

### Layers

* Linear Layer
* Image Convolutional Layer (Conv2DLayer)
* Batch Normalization Layer (BatchNormLayer)
* RNN Layer (RNNLayer)

### Optimizers

* SGD
* NAG
* Adagrad
* Adam

## What will be done 📝

### New Layers

* LSTM, GRU layers
* Transformers

### GPU Accelaration using cupy

### Some more optimizers

### Regularization

* L1, L2 regularization
* Dropout
* Pooling

### Preprocessing modules
