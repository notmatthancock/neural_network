# Multilayer Perceptron
A flexible neural network class for creating networks with arbitrary feedfoward architecture and wide variety of parameters. You can specify any number of layers with any number units each layer fed through any non-linearity you'd like. The non-linearity may even be unit specific. More options include: L1 and L2 regularize, momentum, early-stopping using variance of loss records, minibatch size, and, bootstrapping minibatches.

## Supervised Learning
Supervised learning (classification or regression) is done by specifying the output layer as Logistic or Linear regression.

## Unsupervised Learning
Unsupervised learning may be done by specifying the input and response data as the same. Specifically, by specifying the output layer as a linear regression layer and regressing on the input data, you create an autoencoder. You can build hierarchical networks layer-wise by using this technique.

