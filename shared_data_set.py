import theano
import numpy as np

class SharedDataSet(object):
    def __init__(self, input_path, response_path=None):
        """
        SharedDataSet objects are containers for holding information about sets for training mlps.

        Input and output paths should point to a *.npy file
        input_path: string
        /absolute/path/to/input_data/as/string/*.npy

        output_path: string
        same as input_path but points to the datasets output (labels).
        If None, the input set is used for the output so that the network is trained in an unsupervised fashion.

        Example:
        tr = SharedDataSet(input_path='/mydata/x.npy', output_path='/mydata/y.npy')
        """
        response_path  = input_path if response_path is None else response_path
        
        x = np.load(input_path)
        assert x.ndim ==2, "Input variables should be shape (n_examples, n_features)"
        self.x = theano.shared(x.astype(theano.config.floatX), name = 'x')
        self.N = self.x.get_value().shape[0]

        self.y = theano.shared(np.load( response_path ), name = 'y')
        assert self.N == self.y.get_value().shape[0], "Shape mismatch in data set."
