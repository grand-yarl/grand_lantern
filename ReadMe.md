# This is Grand Lantern neural network library 🏮
The main purpose of this library is to understand - what is happening under the hood of popular deep learning frameworks.
Also, it can be used to build your own ideas - no special knowledge for thst you need! 

This library is written only using numpy (maybe cupy in future), you can see every calculation in neural network!

## How to install ⛏️

Just put this command to terminal

```commandline
pip install grandlantern
```

## Get started 🚀

### 1. Import library

```python
import grandlantern
```

### 2. Define Data Iterator

```python
from grandlantern.dataiterators import DatasetIterator, TableDataset

batch_size = 100
my_dataset_iterator = DatasetIterator(dataset=TableDataset(), batch_size=batch_size)
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
from grandlantern.layers import Sigmoid, ReLU, SoftMax

NN.add_layer(LinearLayer(n_neurons=100, activation=Sigmoid(), biased=True))
NN.add_layer(LinearLayer(n_neurons=50, activation=ReLU(), biased=True))
NN.add_layer(LinearLayer(n_neurons=10, activation=SoftMax(), biased=True))
```

Also layers can be added using attribute model.layers:
```python
from grandlantern.layers import LinearLayer
from grandlantern.layers import Sigmoid, ReLU, SoftMax

NN.layers = 
[
    LinearLayer(n_neurons=100, activation=Sigmoid(), biased=True),
    LinearLayer(n_neurons=50, activation=ReLU(), biased=True)
    LinearLayer(n_neurons=10, activation=SoftMax(), biased=True)
]
```

To look at the model structure the command print can be used:
```python
print(NN)
```


### 6. Train model

Inputs for training must be numpy arrays. There is also option to validate model on test data while training 
(X_test and y_test are optional).

```python
NN.fit(X_train, y_train, X_test, y_test)
```

### 7. Use model

Input for prediction must be numpy array.

```python
y_pred = NN.predict(X_test)
```

## What is supported now ✅

### Layers

* Linear Layer (LinearLayer)
* Image Convolutional Layer (Conv2DLayer)
* Batch Normalization Layer (BatchNormLayer)
* Recurrent Layer (RecurrentLayer)
* RNN Layer (RNNLayer)
* Flatten Layer (FlattenLayer)

### Optimizers

* SGD
* NAG
* Adagrad
* Adam

### Datasets and Iterators

* Dataset Iterator with shuffle
* Table Dataset
* Image Dataset
* Sequence Dataset

### Regularization

* L1, L2, Elastic Net
* Dropout Layer (DropOutLayer)

## What will be done 📝

### New Layers

* Bidirectional RNN
* LSTM, GRU layers
* Attention layers
* Seq2seq model
* Transformers
* Pooling layers

### GPU Accelaration using cupy

### Some more optimizers

### Preprocessing modules
