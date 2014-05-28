import theano
import theano.tensor as T
import numpy as np

class LinearRegression(object):
    def __init__(self, input, n_in, n_out):
        """
        Initialize the parameters of linear regression.

        :type   input: theano.tensor.TensorType
        :param  input: symbolic variable that describes the input of the architecture
        
        :type   n_in: int
        :param  n_in: number of input units, the dimension of the space in which the datapoint lies

        :type   n_out: int
        :param  n_out: number of output units, the dimension of space in which the target lies
        """
        # Params to learn as shared variables
        self.W = theano.shared(np.zeros((n_in,n_out)).astype(np.float32), name='W', borrow=True)
        self.b = theano.shared(np.zeros(n_out).astype(np.float32), name='b', borrow=True)
        self.params = [self.W, self.b]

        # Regression prediction
        self.p_y_given_x = (T.dot(input,self.W)+self.b).flatten()

    def mse(self, y):
        """
        Predict mean squared error using current model params over y.
        """
        return ((self.p_y_given_x-y)**2).mean()

class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        """
        Initialize the parameters of logistic regression.

        :type   input: theano.tensor.TensorType
        :param  input: symbolic variable that describes the input of the architecture
        
        :type   n_in: int
        :param  n_in: number of input units, the dimension of the space in which the datapoint lies

        :type   n_out: int
        :param  n_out: number of output units, the dimension of space in which the target lies
        """
        # Params to learn as shared variables
        self.W = theano.shared(np.zeros((n_in,n_out)).astype(np.float32), name='W',borrow=True)
        self.b = theano.shared(np.zeros(n_out).astype(np.float32), name='b',borrow=True)
        self.params = [self.W, self.b]

        # Probability prediction
        self.p_y_given_x = T.nnet.softmax(T.dot(input,self.W)+self.b)

        # Class prediction
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

    def neg_log_likelihood(self,y):
        """
        Return the mean of the negative log-likelihood of the prediction of this model under a given target distribution.
        """
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),y])
    
    def misclass_error(self, y):
        """
        Return the number of examples in y misclassified / total number in y
        """
        # Check if input has correct dimension.
        if y.ndim != self.y_pred.ndim:
            raise TypeError('Input should have same dimension as self.y_pred.')
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s where 1 represents an error
            return T.mean(T.neq(self.y_pred,y))
        else:
            raise NotImplementedError()

class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, activation=T.tanh):
        """
        Construct a hidden layer of a multilayer perceptron.

        :type   rng:      numpy.Random.RandomState
        :param  rng:      random number generator for weight init

        :type   input:    theano.tensor.matrix
        :param  input:    symbolic tensor of shape (n_examples, n_in)

        :type   n_in:     int
        :param  n_in:     size of input dimension
        
        :type   n_out:    int
        :param  n_out:    size of output dimension

        :type   activation: theano.Op or function
        :param  activation: (non)linear activation function for hidden units
        """
        self.input = input

        W_vals = np.asarray(
            rng.uniform(
                low  = -np.sqrt(6. / (n_in+n_out)),
                high = np.sqrt(6. / (n_in+n_out)),
                size = (n_in, n_out)
            ).astype(theano.config.floatX)
        )
        if activation == theano.tensor.nnet.sigmoid:
            W_vals *= 4

        self.W = theano.shared(W_vals, name='W')
        self.b = theano.shared(np.zeros(n_out).astype(theano.config.floatX), name='b')
        self.params = [self.W, self.b]

        self.output= activation( T.dot(input,self.W) + self.b )

class NeuralNetwork(object):
    """
    Create a symbolic (theano) Feed-forward neural net with a logistic or linear regressor on top for classification or regression.
    """
    def __init__(self, rng, input, n_in, n_hidden, n_out, activation):
        """
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output unit
        """
        self.hiddenLayer = HiddenLayer(rng=rng, input=input, n_in=n_in, n_out=n_hidden, activation=activation)
        self.logregLayer = LogisticRegression(input=self.hiddenLayer.output, n_in=n_hidden, n_out=n_out)
        self.params      = self.hiddenLayer.params + self.logregLayer.params

        self.L1 = abs(self.hiddenLayer.W).sum() + abs(self.logregLayer.W).sum()
        self.L2 = (self.hiddenLayer.W**2).sum() + (self.logregLayer.W**2).sum()

        self.neg_log_likelihood = self.logregLayer.neg_log_likelihood
        self.misclass_error     = self.logregLayer.misclass_error
