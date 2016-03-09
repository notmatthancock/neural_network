# Neural Network
A flexible neural network class for creating networks with arbitrary feedfoward architecture and wide variety of parameters. You can specify any number of layers with any number units each layer fed through any non-linearity you'd like (within your machine's limits of course). The non-linearity may be unit specific. Different layer types inlclude: Convolutional, LpPooling, and FullyConnected.Training options include: L1 and L2 regularize, momentum, early-stopping using variance of loss records, minibatch size, and, bootstrapping minibatches.

## Supervised Learning
Supervised classification or regression is done by instantiating a `NeuralNetworkClassifier` or `NeuralNetworkRegressor` object respectively. Classification assumes classes are mutually exclusive and uses the negative log-likelihood of the multinomial regression model as a loss function. Regression uses mean-squared error as a loss.

### Examples
Import theano for activation functions.
```
import neural_network as nn
import theano.tensor as T
```

Build the architecture next.

```
# Logistic regression:
arc = [
    ('F', dict(n_in=784, n_out=10, activation=T.nnet.softmax))
]

# One hidden layer fully connected
arc = [
    ('F', dict(n_in=784, n_out=392, activation=T.tanh))
    ('F', dict(n_in=784, n_out=10, activation=T.nnet.softmax))
]


# Convolutional network with max pooling
arc = [
    ('C', dict(channels_in=1, channels_out=5, input_shape=(28,28), filter_shape=(5,5))),
    ('P', dict(p=100, stride_shape=(2,2), window_shape=(2,2))),
    ('C', dict(channels_in=5, channels_out=50, filter_shape=(5,5))),
    ('P', dict(p=100, stride_shape=(2,2), window_shape=(2,2))),
    ('F', dict(n_in=800, n_out=100)),
    ('F', dict(n_in=100, n_out=10, activation=T.nnet.softmax))
]
```

After building the architecture, instantiate the network object, load relevant
datasets, and train.

```
net = nn.NeuralNetworkClassifier(arc)

net.load_training_set('training_inputs.npy', 'training_responses.npy')
net.load_validation_set('validation_inputs.npy', 'validation_responses.npy')

net.train(n_epochs=100, learning_rate=0.12, batch_size=1000)

net.load_testing_set('testing_inputs.npy', 'testing_responses.npy')
net.test()
```

Relevant statistics are stored in `net.training_stats` and `net.testing_stats` dictionaries after training and testing has been performed. 

## Unsupervised Learning
Unsupervised learning may be done by specifying the input and response data as the same for a `NeuralNetworkRegressor` model. This creates what's commmonly called an autoencoder.
