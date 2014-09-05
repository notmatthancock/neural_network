# Neural Network
A flexible neural network class for creating networks with arbitrary feedfoward architecture and wide variety of parameters. You can specify any number of layers with any number units each layer fed through any non-linearity you'd like (within your machine's limits of course). The non-linearity may be unit specific. Different layer types inlclude: Convolutional, LpPooling, and FullyConnected.Training options include: L1 and L2 regularize, momentum, early-stopping using variance of loss records, minibatch size, and, bootstrapping minibatches.

## Supervised Learning
Supervised classification or regression is done by instantiating a `NeuralNetworkClassifier` or `NeuralNetworkRegressor` object respectively. Classification assumes classes are mutually exclusive and uses the negative log-likelihood of the multinomial regression model as a loss function. Regression uses mean-squared error as a loss.

## Unsupervised Learning
Unsupervised learning may be done by specifying the input and response data as the same for a `NeuralNetworkRegressor` model. This creates what's commmonly called an autoencoder.

