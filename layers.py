import numpy as np
import theano
import theano.tensor as T

__all__ = [ 'FullyConnectedLayer',
            'ConvolutionalLayer',
            'LpPoolingLayer',
            'LAYER_TYPES' ]

LAYER_TYPES = {'F': 'FullyConnectedLayer',
               'C': 'ConvolutionalLayer',
               'P': 'LpPoolingLayer'}

# linear
#       self.response   = T.matrix('Linear regression response variable')
#       self.loss       = ((self.output-self.response)**2).mean() 
#       self.loss.name  = 'MSE loss'

# logistic
#        assert n_out > 1, "n_out should be at least 2"
#
#
#        self.y_pred = T.argmax( self.output, axis=1 )
#        self.output.name = 'Logistic regression softmax output'
#        self.y_pred.name = 'Logistic regression hard-assignment output' 
#
#        self.response =  T.ivector('Logistic regression response variable')
#        self.loss     = -T.mean(T.log(self.output)[T.arange(self.response.shape[0]), self.response])
#        self.loss.name= 'Negative loglikelihood loss'
#        self.miss     =  T.mean(T.neq(self.y_pred, self.response))
#        self.miss.name= 'Misclassification error'

class FullyConnectedLayer(object):
    def __init__(self,
                 input,
                 n_in,
                 n_out,
                 activation=T.tanh,
                 rng=np.random.RandomState(),
                 weight_paths=None
        ):
        """
        Construct a hidden layer of a neural network.
        
        input:    theano.tensor.matrix
        symbolic tensor of shape (n_examples, n_in)
        note that if input.ndim != 2, axes past the second will be flattened into the second.
        This is the case for pooling / convolutional layers coming into a fully connected layer.
        Note that the length of the second dimension must be n_in after flattening.
        
        n_in:     int
        size of input dimension
        
        n_out:    int
        size of output dimension
        
        activation: theano.Op or function
        (non)linear activation function for hidden units
        Default is T.tanh. Other possibilities are:
        T.nnet.sigmoid
        T.nnet.softplus
        T.nnet.softmax  (use this for output layer for logistic regression)
        lambda x: x     (use this for output layer of linear regression)

        rng:      numpy.Random.RandomState
        random number generator for weight init

        weight_paths: list of strings
        The first string should specify the path to the npy file where the W matrix located.
        The second should specify the path to the npy file where the b vector is located.
        None instead of a string skips that param. For instance:
        ['/weights.npy', None]
        Loads only the weight matrix and initializes the bias vector as normal.
        """
        assert weight_paths is None or ( weight_paths is not None and len(weight_paths)==2 ), "If weight paths is specified it must be a list of length 2."
        assert callable(activation), "activation must be callable"

        self.n_in   = n_in
        self.n_out  = n_out
        self.input  = input

        # set weights
        if weight_paths is None or (weight_paths is not None and weight_paths[0] is None):
            W_val = np.asarray(
                rng.uniform(
                    low = -np.sqrt(6.0 / (n_in + n_out)),
                    high=  np.sqrt(6.0 / (n_in + n_out)),
                    size=(n_in, n_out)
                ).astype(theano.config.floatX)
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_val *= 4
        else: # use path provided
            W_val = np.load(weight_paths[0])
            assert W_val.dtype == theano.config.floatX
            assert W_val.shape == (n_in, n_out)

        if weight_paths is None or (weight_paths is not None and weight_paths[1] is None):
            b_val = np.zeros((n_out,), dtype=theano.config.floatX)
        else:
            b_val = np.load(weight_paths[1])
            assert b_val.dtype == theano.config.floatX
            assert b_val.shape == (n_out, )
         
        self.W = theano.shared(W_val, name='W')
        self.b = theano.shared(b_val, name='b')
        self.params = [self.W, self.b]

        self.output = activation(T.dot(self.input if self.input == 2 else self.input.flatten(2), self.W) + self.b)
        self.output.name = 'Fully connected activation output' 

class ConvolutionalLayer(object):
    def __init__(self):
        pass 

class LpPoolingLayer(object):
    def __init__(self):
        pass
