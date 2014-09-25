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
                  rng=np.random.RandomState() ):
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
        """
        assert callable(activation), "activation must be callable"

        self.n_in   = n_in
        self.n_out  = n_out
        self.input  = input
        layer_type = 'F'

        # for building from layers
        self.arc_vars = locals()
        self.arc_vars.pop('self')

        # set weights
        W_val = np.asarray(
            rng.uniform(
                low = -np.sqrt(6.0 / (n_in + n_out)),
                high=  np.sqrt(6.0 / (n_in + n_out)),
                size=(n_in, n_out)
            ).astype(theano.config.floatX)
        )
        if activation == theano.tensor.nnet.sigmoid:
            W_val *= 4

        b_val = np.zeros((n_out,), dtype=theano.config.floatX)
         
        self.W = theano.shared(W_val, name='W')
        self.b = theano.shared(b_val, name='b')
        self.params = [self.W, self.b]

        self.output = activation(T.dot(self.input if self.input.ndim == 2 else self.input.flatten(2), self.W) + self.b)
        self.output.name = 'Fully connected activation output' 

class ConvolutionalLayer(object):
    def __init__( self,
                  input=None,
                  channels_in=None,
                  input_shape=None,
                  channels_out=None,
                  filter_shape=None,
                  activation=T.tanh,
                  rng=np.random.RandomState(),
                  is_input=False ):
        """
        Construct a convolutional layer of a neural network.
        
        input:    T.tensor4
        symbolic tensor of shape (examples, channels_in, height, width)
        
        channels_in:     int
        number of channels in the input.
        
        channels_out:    int
        number of channels in the output. (number of "feature maps" learned)

        input_shape: tuple (height, width)
        
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
        """
        assert callable(activation), "activation must be callable"

        self.channels_in  = channels_in
        self.channels_out = channels_out
        self.filter_shape = filter_shape
        self.input_shape  = input_shape
        self.input  = input
        self.layer_type = 'C'

        # This is a bit of a workaround to manage input examples as flattened vectors,
        # while the conv op expects 4D.
        if is_input:
            self.input_reshape = self.input.reshape((
                self.input.shape[0],
                self.channels_in,
                self.input_shape[0],
                self.input_shape[1]
            ))
        else:
            self.input_reshape = self.input
        self.input_reshape.name = self.input.name if not is_input else "%s reshape"%self.input.name

        # for building from layers
        self.arc_vars = locals()
        self.arc_vars.pop('self')

        # set weights
        weight_shape = (channels_out, channels_in) + filter_shape
        W_val = np.asarray(
            rng.uniform(
                low  = -np.sqrt(1.0 / np.prod(filter_shape) / channels_in ),
                high =  np.sqrt(1.0 / np.prod(filter_shape) / channels_in ),
                size = weight_shape
            ).astype(theano.config.floatX)
        )
        if activation == theano.tensor.nnet.sigmoid:
            W_val *= 4

        b_val = np.zeros((channels_out,), dtype=theano.config.floatX)
         
        self.W = theano.shared(W_val, name='W')
        self.b = theano.shared(b_val, name='b')
        self.params = [self.W, self.b]

        # convolve, add bias to axis of output channels, apply activation.
        self.output = activation( T.nnet.conv2d(self.input_reshape, self.W) + self.b.dimshuffle('x', 0, 'x', 'x') ) 
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
            print "Lp Pooling not implemented yet. Using Max-pooling instead."
        if False:#p < 10:
            # TODO: efficient Lp pooling
            pass
        else: # use the max pooling op
            self.output = T.signal.downsample.max_pool_2d(input, (window_shape))
        self.output.name = "Lp Pooling output"
