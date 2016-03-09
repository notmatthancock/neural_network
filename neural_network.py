import numpy as np
import theano, time, cPickle, uuid, sys
import theano.tensor as T
from .layers import *
from .shared_data_set import SharedDataSet 

class NeuralNetwork(object):
    """
    Parent class for creating a symbolic Feed forward neural network classifier or regressor.
    """
    def __init__(self, architecture, random_state=None):
        """
        architecture: list of tuples 
        
        random_state: numpy random state, default None

        Each item is of the form: (char, dict) which specifies architecture
        of each layer. The first item in each is one of LAYER_TYPES.keys()
        specifying the layer type. The second item must be a dictionary of
        args matching the call signature in the specified layer type with the
        exception of the 'input' argument which is set when the computation
        graph is built.
        
        Example: 
            [('C', {'channel_in': 3, 'channel_out': 6, 'filter_shape': (5,5), 'activation': T.tanh}),
             ('P', {'p': 2, 'stride_shape': (2,2), 'window_shape': (2,2)}),
             ('F', {'n_in': 200, 'n_out': 10, 'activation': T.nnet.softmax})]
                 
        would describe a network with a convolution over 3 input channels with
        filters of size (5,5) and tanh activation followed by L2 pooling over (2,2)
        windows. Lastly, there is a fully connected layer with softmax output
        for multinomial regression.
        """
        # Ugly one liner for checking each key corresponds to a valid type
        assert sum(map(lambda l: l in LAYER_TYPES.keys(), [a[0] for a in architecture])) == len(architecture), \
               "Invalid layer type key in architecture."

        self.input              = T.matrix('Network input')
        self.architecture       = architecture
        self.n_layers           = len(architecture)+1 # count the input layer
        self.layers             = []
        self.params             = []

        if random_state is None:
            self.random_state = np.random.RandomState()
        else:
            self.random_state = random_state

        for arc in self.architecture:
            # Add the random_state to the argument list
            arc[1].update([('random_state', random_state)])
            # Append the correct input variable to the arg dict.
            if len(self.layers)==0:
                #if arc[0] == 'C':
                #    # If the first layer is convolutional, we have to reshape the input
                #    # to the tensor4 shape. In particular, the input matrix is assumed to be
                #    # (n_examples, n_features) where n_features is flattened from the shape
                #    # (channels_in, height, width).
                #    arc[1].update([('input',
                #        self.input.reshape((
                #            self.input.shape[0],
                #            architecture[0][1]['channels_in'],
                #            architecture[0][1]['input_shape'][0],
                #            architecture[0][1]['input_shape'][1]
                #        ))
                #    )])
                #else:
                arc[1].update([('input', self.input)])
                if arc[0] == 'C': arc[1].update([('is_input', True)])
            else:
                # If not the first layer, the input is the output of previous.
                arc[1].update([('input', self.layers[-1].output)])

            # Create a new instance of current layer type with given args
            # and append it to the layer list.
            self.layers.append( globals()[LAYER_TYPES[arc[0]]](**arc[1]) )
            if arc[0] != 'P':
                self.params += self.layers[-1].params

        self.output = self.layers[-1].output
        self.predict = theano.function([self.input], self.output, allow_input_downcast=True)

    def set_params_from_list(self, P):
        """
        Set the parameters of the network using parameters stored in list, P.
        """
        for i in range(len(P)):
            self.params[i].set_value( P[i] )

    def _load_data_set(self, input, response, name ):
        setattr(self, name, SharedDataSet(input=input, response=response))
    def load_training_set(self, input, response=None):
        self._load_data_set(input, response, 'training_set')
    def load_validation_set(self, input, response=None):
        self._load_data_set(input, response, 'validation_set')
    def load_testing_set(self, input, response=None):
        self._load_data_set(input, response, 'testing_set')

    def _unload_data_set(self, name):
        delattr(self, name)
    def unload_training_set(self):
        self._unload_data_set('training_set')
    def unload_validation_set(self):
        self._unload_data_set('validation_set')
    def unload_testing_set(self):
        self._unload_data_set('testing_set')

    def train(   self,
                 step_size             = 0.1,
                 n_epochs              = 10,
                 batch_size            = None,
                 L1_coef               = None,
                 L2_coef               = None,
                 class_weight          = None,
                 start_rand            = False,
                 callback              = None
        ):
        """
        Train the network for a number of epochs. Training and validation sets must be loaded by calling load_..._set(...)

        step_size: float
        step size of gradient descent

        n_epochs: int
        Maximum number of epochs to cycle through gradient descent

        batch_size: int
        Size of batches used for grad descent updates. If this number does not divide the size of the
        training set exactly, then the remainder is not utilized (assuming bootstrap is not used). If None,
        the size of the training_set is used.

        L1_coef: float
        Amount of L1 regularization added weights 

        L2_coef: float
        Amount of L2 regularization added weights

        start_rand: bool
        If true, the network parameters are set to random values before initializing. False (default) uses the current network param values as starting points.

        class_weight: list, default None
        default None weights classes equally. Otherwise, the classes are weighted in the cost function by the class weight specified.
        """
        # Input checking.
        assert hasattr(self,'training_set') and hasattr(self,'validation_set'), \
               "Testing or validation set not present. You must load both via the \
               NeuralNetwork object's load_..._set(...) methods."
        assert batch_size <= self.training_set.N, \
               "Batch size cannot be greater than size of training set."

        # Class weights.
        ytr = self.training_set.y.eval()
        yva = self.validation_set.y.eval()
        classes = np.unique(ytr)
        if class_weight is not None:
            assert len(class_weight) == len(classes), \
            "class_weight list length doesn't match number of classes."
        else:
            class_weight = np.ones(len(classes))
        # Symbolic sample weight vector to use in the general case.
        sample_weight = T.fvector('sample_weight')
        # Shared variable sample weight.
        tr_sample_weight = theano.shared(
            np.array([class_weight[y] for y in ytr]).astype(np.float32),
            name='tr_sample_weight'
        )
        # Shared variable validation weight.
        va_sample_weight = theano.shared(
            np.array([class_weight[y] for y in yva]).astype(np.float32),
            name='va_sample_weight'
        )

        print("Beginning new trial.")

        batch_size      = self.training_set.N if batch_size is None else batch_size
        n_train_batches = int(np.floor(self.training_set.N / float(batch_size)))
        index = T.lscalar('index')

        if start_rand:
            print("... Randomizing network parameters")
            for param in self.params:
                param.set_value(self.random_state.randn( param.get_value().shape ).astype( param.dtype ))

        print("... Compiling")

        if isinstance(self, NeuralNetworkClassifier):
            self.loss      = -T.log(self.output)[T.arange(self.response.shape[0]), self.response]
            self.loss      = T.dot(self.loss, sample_weight) / sample_weight.shape[0]
            self.loss.name = "Negative log-likelihood loss"
            self.miss      = T.mean(T.neq(self.y_pred, self.response))
            self.miss.name = 'Misclassification error'
        elif isinstance(self, NeuralNetworkRegressor):
            self.loss       = ((self.output-self.response)**2).mean() 

        # Add regularizers to loss function.
        if L2_coef is not None:
            L2 = T.sum(self.params[0]**2)
            for j in range(2,len(self.params)):
                if j % 2 == 0: # This ignores intercept terms.
                    L2 += T.sum(self.params[j]**2)
            self.loss += L2*L2_coef
            self.loss.name += " L2 regularization"
        if L1_coef is not None:
            L1 = T.sum(T.abs_(self.params[0]))
            for j in range(2,len(self.params)):
                if j % 2 == 0: # This ignores intercept terms.
                    L1 += T.sum(T.abs_(self.params[j]))
            self.loss += L1*L1_coef
            self.loss.name += " L1 regularization"

        # Compute symbolic gradient of the cost with respect to params
        gparams = []
        for param in self.params:
            gparams.append( T.grad(cost=self.loss, wrt=param) )
        
        updates = []
        for i in range(len(self.params)):
            updates.append((self.params[i], self.params[i] - step_size*gparams[i]))

        train_model = theano.function(
            inputs=[index],
            outputs=self.loss,
            updates=updates,
            givens={
                self.input:    self.training_set.x[index*batch_size:(index+1)*batch_size],
                self.response: self.training_set.y[index*batch_size:(index+1)*batch_size],
                sample_weight: tr_sample_weight[index*batch_size:(index+1)*batch_size]
            }
        )

        has_miss = hasattr(self, 'miss')
        if has_miss:
            validation_cost_and_miss = theano.function(
                inputs=[],
                outputs=[self.loss, self.miss],
                givens={
                    self.input:    self.validation_set.x,
                    self.response: self.validation_set.y,
                    sample_weight: va_sample_weight
                }
            )
        else:
            validation_cost = theano.function(
                inputs=[],
                outputs=self.loss,
                givens={
                    self.input:     self.validation_set.x,
                    self.response:  self.validation_set.y,
                    sample_weight:  va_sample_weight
                }
            )

        print("... Beginning training\n")
        
        # setup variables before train loop
        start_time = time.clock()
        best_params = [None]*len(self.params)
        best_va = np.inf
        va_miss = None
        epoch = 0
        loss_records = np.zeros((n_epochs, 3 if has_miss else 2))

        while epoch < n_epochs:
            tr_cost = 0.
            if has_miss:
                va_cost, va_miss = validation_cost_and_miss()
            else:
                va_cost = validation_cost()
            
            for minibatch_index in xrange(n_train_batches):
                tr_cost += train_model(minibatch_index) / n_train_batches

            # record losses for epoch
            loss_records[epoch,0], loss_records[epoch,1] = tr_cost, va_cost
            if loss_records.shape[1]==3:
                loss_records[epoch,2] = va_miss
                sys.stdout.write(
                    "Epoch: %d || TrCost: %f || VaCost: %f || VaMiss: %f\r"
                    % (epoch+1, tr_cost, va_cost, va_miss)
                )
            else:
                sys.stdout.write(
                   "Epoch: %d || TrCost: %f || VaCost: %f\r"
                    % (epoch+1, tr_cost, va_cost)
                )
            sys.stdout.flush()
            
            # save on best missclass for classification and best cost for regression
            if (va_miss if has_miss else va_cost) < best_va:
                # record new best
                best_va = (va_miss if has_miss else va_cost) 
                best_epoch = epoch
                for i in range(len(self.params)):
                    best_params[i] = self.params[i].get_value().copy()

            # call the call back
            if callback is not None: callback(locals())

            epoch += 1
        end_time = time.clock()

        for i in range(len(self.params)):
            self.params[i].set_value( best_params[i] )

        self.training_stats = {
            'time':       end_time-start_time,
            'rate':       epoch/(end_time-start_time),
            'loss':       loss_records,
            'parameters': best_params,
#            'cost':       cost, ?????
            'epoch':      best_epoch
        }

        print ("\n\n... Training finished")
        print ("Running at ~ %f epochs / sec"%(epoch/(end_time-start_time)))

    def test(self):
        """Perform testing aftering having been trained."""
        assert hasattr(self,'testing_set'), "You must load the testing set via NeuralNetwork.load_testing_set()"
        assert hasattr(self, 'training_stats'), "Train before test!"
        test_loss = theano.function(
            inputs=[],
            outputs=self.training_stats['cost'],
            givens={
                self.input:     self.testing_set.x,
                self.response:  self.testing_set.y
            }
        )
        self.testing_stats = dict()
        self.testing_stats['loss'] = test_loss()

        if hasattr(self, 'miss'):
            test_miss = theano.function( 
                inputs=[],
                outputs=self.miss,
                givens={
                    self.input:     self.testing_set.x,
                    self.response:  self.testing_set.y
                }
            )
            self.testing_stats['miss'] = test_miss()

    def save_stats(self, save_path=None, prefix=None):
        if save_path is None:
            save_path = self.__str__().replace(' ', '').replace('\n',' ')+'---'+str(uuid.uuid4())+'.pkl'
        if prefix is not None:
            save_path = prefix+save_path

        assert hasattr(self,'training_stats'), "Train to get stats, then save."

        f=open(save_path,'wb')
        
        if hasattr(self,'training_stats') and hasattr(self,'testing_stats'):
            cPickle.dump({'training_stats': self.training_stats, 'testing_stats': self.testing_stats, 'arc': self.architecture}, f)
        else: # no testing stats
            cPickle.dump({'training_stats': self.training_stats, 'arc': self.architecture}, f)
        f.close()

    def __str__(self):
        return "TODO"
        # s = str(self.n_layers) + ' layered MLP:\n'
        # s += str(self.architecture[0]) + ' in => '
        # for i in range(self.n_hidden_layers):
        #     s += str(self.architecture[i+1]) + 'h '
        #     if hasattr(self.activation[i], '__name__'):
        #         s += '( ' + self.activation[i].__name__ + ' ) '
        #     elif hasattr(self.activation[i], '__str__'):
        #         s += '( ' + self.activation[i].__str__() + ' ) '
        #     s += ' => '
        # return s + str(self.architecture[-1]) + 'out ( ' + self.output_layer_type + ' )'

class NeuralNetworkClassifier(NeuralNetwork):
    """Create a feed forward neural network for multinomial regression (multiclass classification with mutually exclusive classes)"""
    def __init__(self, architecture, random_state=None):
        __doc__ = super(NeuralNetworkClassifier, self).__init__.__doc__
        super(NeuralNetworkClassifier, self).__init__(architecture, random_state=random_state)
        self.response  = T.ivector('Multinomial regression response variables (labels)')

        if architecture[-1][1]['activation'] != T.nnet.softmax:
            print "Warning: activation function of output layer should be T.nnet.softmax for Classification."

        self.y_pred = T.argmax( self.output, axis=1 )
        self.output.name = 'Multinomial regression softmax output'
        self.y_pred.name = 'Multinomial regression hard-assignment output' 

    def load_training_set(self, input, response=None):
        super(NeuralNetworkClassifier, self).load_training_set(input, response)
        self._validate_set('training_set')
    def load_validation_set(self, input, response=None):
        super(NeuralNetworkClassifier, self).load_validation_set(input, response)
        self._validate_set('validation_set')
    def load_testing_set(self, input, response=None):
        super(NeuralNetworkClassifier, self).load_testing_set(input, response)
        self._validate_set('testing_set')
    def _validate_set(self, set):
        # We must cast the labels as integers
        setattr(getattr(self, set), 'y', T.cast( getattr(self, set).y, 'int32'))
        # And check that the reponse labels are contained in a 1d vector
        assert getattr(self, set).y.ndim == 1, \
        "Response variables should be contained in a one dimensional vector \
        for classification and coded as unique integers per class label."

class NeuralNetworkRegressor(NeuralNetwork):
    """Create a feed forward neural network for regression."""
    def __init__(self, architecture, random_state=None):
        super(NeuralNetworkRegressor, self).__init__(architecture, random_state=random_state)
        self.response   = T.matrix('Regression response variable')

    def load_training_set(self, input, response=None):
        super(NeuralNetworkRegressor, self).load_training_set(input, response)
        self._validate_set('training_set')
    def load_validation_set(self, input, response=None):
        super(NeuralNetworkRegressor, self).load_validation_set(input, response)
        self._validate_set('validation_set')
    def load_testing_set(self, input, response=None):
        super(NeuralNetworkRegressor, self).load_testing_set(input, response)
        self._validate_set('testing_set')
    def _validate_set(self, set):
        assert getattr(self, set).y.ndim == 2, \
        "Response variables should be contained in a two dimensional matrix \
        for regression. If the response is 1D, response should be a matrix of \
        with shape (n_examples, 1)"
