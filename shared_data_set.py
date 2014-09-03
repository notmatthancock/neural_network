import theano
import numpy as np

class SharedDataSet(object):
    def __init__(self, input_path, response_path=None, output_layer_type=None):
        """
        SharedDataSet objects are containers for holding information about sets for training mlps.

        Input and output paths should point to a *.npy file
        input_path: string
        /absolute/path/to/input_data/as/string/*.npy

        output_path: string
        same as input_path but points to the datasets output (labels). If None, the input set is used for the output so that the network is trained in an unsupervised fashion.

        Example:
        tr = SharedDataSet(input_path='/mydata/x.npy', output_path='/mydata/y.npy')
        """
        assert output_layer_type is not None and output_layer_type in OUTPUT_LAYER_TYPES, "output_layer_type_must be specified"
        response_path  = input_path if response_path is None else response_path
        
        self.x = theano.shared(np.load( input_path ).astype(theano.config.floatX)   , name = 'x')

        y = np.load( response_path )
        if y.ndim == 1 and output_layer_type == 'LinearRegression':
            y = y[:,np.newaxis]
            self.y = theano.shared(y, name = 'y')
        elif output_layer_type == 'LogisticRegression':
            self.y = theano.shared(np.load( response_path ), name = 'y')


        self.N = self.x.get_value().shape[0]
        assert self.N == self.y.get_value().shape[0], "Shape mismatch in data set."
        
        if output_layer_type == 'LogisticRegression':
            self.y = T.cast( self.y, 'int32' )
            assert self.y.ndim == 1, "Response variables should be contained in a one dimensional vector for Logistic Regression coded as unique integers per class label."
        elif output_layer_type == 'LinearRegression':
            assert self.y.ndim == 2, "Response variables should be contained in a matrix by row for Linear Regression"
