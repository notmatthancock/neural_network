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

class FullyConnectedLayer(object):
    def __init__( self,
                  input,
                  n_in,
                  n_out,
                  activation=T.tanh,
                  rng=np.random.RandomState(),
                  weight_paths=None ):
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
        layer_type = 'F'

        # for building from layers
        self.arc_vars = locals()
        self.arc_vars.pop('self')

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

        self.output = activation(T.dot(self.input if self.input.ndim == 2 else self.input.flatten(2), self.W) + self.b)
        self.output.name = 'Fully connected activation output' 

class ConvolutionalLayer(object):
    def __init__( self,
                  input,
                  channels_in,
                  channels_out,
                  filter_shape,
                  activation=T.tanh,
                  rng=np.random.RandomState(),
                  weight_paths=None ):
        """
        Construct a convolutional layer of a neural network.
        
        input:    T.tensor4
        symbolic tensor of shape (examples, channels_in, height, width)
        
        channels_in:     int
        number of channels in the input.
        
        channels_out:    int
        number of channels in the output. (number of "feature maps" learned)
        
        filter_shape:    tuple of ints (height, width).
        Resulting filters combine over input channels, so
        actual filters will be (channels_in, height, width).

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
        ['./weights.npy', None]
        Loads only the weight matrix and initializes the bias vector as normal.
        """
        assert weight_paths is None or ( weight_paths is not None and len(weight_paths)==2 ), "If weight paths is specified it must be a list of length 2."
        assert callable(activation), "activation must be callable"

        self.channels_in  = channels_in
        self.channels_out = channels_out
        self.filter_shape = filter_shape
        self.input  = input
        layer_type = 'C'

        # for building from layers
        self.arc_vars = locals()
        self.arc_vars.pop('self')

        # set weights
        weight_shape = (channels_out, channels_in) + filter_shape
        if weight_paths is None or (weight_paths is not None and weight_paths[0] is None):
            W_val = np.asarray(
                rng.uniform(
                    low  = -np.sqrt(1.0 / np.prod(filter_shape) / channels_in ),
                    high =  np.sqrt(1.0 / np.prod(filter_shape) / channels_in ),
                    size = weight_shape
                ).astype(theano.config.floatX)
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_val *= 4
        else: # use path provided
            W_val = np.load(weight_paths[0])
            assert W_val.dtype == theano.config.floatX
            assert W_val.shape == weight_shape 

        if weight_paths is None or (weight_paths is not None and weight_paths[1] is None):
            b_val = np.zeros((channels_out,), dtype=theano.config.floatX)
        else:
            b_val = np.load(weight_paths[1])
            assert b_val.dtype == theano.config.floatX
            assert b_val.shape == (channels_out, )
         
        self.W = theano.shared(W_val, name='W')
        self.b = theano.shared(b_val, name='b')
        self.params = [self.W, self.b]

        # convolve, add bias to axis of output channels, apply activation.
        self.output = activation( T.nnet.conv2d(self.input, self.W) + self.b.dimshuffle('x', 0, 'x', 'x') ) 
        self.output.name = 'Convolutional activation output' 

class LpPoolingLayer(object):
    def __init__(self, input, p=2, stride_shape=(2,2), window_shape=(2,2), avg=False):
       """
       input: symbolic tensor4 variable
       The shape should be (n_examples, n_channels, height, width)
       
       p: float >= 1
       The "p" in the p norm. 2 is L2 pooling. If p >= 10, max-pooling is used.

       stride_shape: tuple (height, width)
       This determines the stride in both directions of the pooling operation.
       *Important* for max pooling (p>=10), stride_shape is ignored.

       window_shape: tuple (height, width)
       This is the region over which each pooling takes place. If stride_shape==window_shape,
       pooling is taken over non-overlapping regions.

       avg: bool
       If true, we divide by np.prod(window_shape) before taking the pth root.
       """
       assert p >= 1, "p must be greater than or equal to 1."
       self.p = p
       self.stride_shape = stride_shape
       self.window_shape = window_shape

       if p < 10:
           # We're basically doing the following:
           # for each channel create a 3D volume dirac delta of
           # shape (channels_in, window_shape[0], window_shape[1]).
           # This ith 3D volume has a 2D matrix of all ones when
           # i==j and matrics of zeros in all other channels.
           # This is done for each channel creating a 4D filter set.
           window = T.eye(input.shape[1], dtype=input.dtype)
           window = window.dimshuffle(0, 1, 'x', 'x')
           window = window.repeat(window_shape[0], axis=2).repeat(window_shape[1], axis=3)
           window /= 1. if not avg else float(np.prod(window_shape))
           print window.dtype
           self.output = ( T.nnet.conv2d(input**p, window, subsample=stride_shape) )**(1./p)
       else: # use the max pooling op
           self.output = T.signal.downsample.max_pool_2d(input, (window_shape))
       self.output.name = "Lp Pooling output"






