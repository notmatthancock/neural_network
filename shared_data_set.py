import theano
import numpy as np

class SharedDataSet(object):
    def __init__(self, input, response=None):
        """
        SharedDataSet objects are containers for holding information about sets for training mlps.

        Input and output paths should point to a *.npy file
        input: string or ndarray
        /absolute/path/to/input_data/as/string/*.npy

        output: string or ndarray
        same as input but points to the datasets output (labels).
        If None, the input set is used for the output so that the network is trained in an unsupervised fashion.

        Example:
        tr = SharedDataSet(input='/mydata/x.npy', output_path='/mydata/y.npy')
        """
        response  = input if response is None else response
        
        x = np.load(input) if isinstance(input, str) else input
        assert isinstance(x, np.ndarray), "Input should be numpy.ndarray"
        assert x.ndim ==2, "Input variables should be shape (n_examples, n_features)"
        self.x = theano.shared(x.astype(theano.config.floatX), name = 'x')
        self.N = self.x.get_value().shape[0]

        y = np.load(response) if isinstance(response, str) else response
        assert isinstance(x, np.ndarray), "Response should be numpy.ndarray"
        self.y = theano.shared(y, name = 'y')
        assert self.N == self.y.get_value().shape[0], "Shape mismatch in data set."
