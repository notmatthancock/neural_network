import numpy as np
import theano, time, cPickle, uuid, sys
import theano.tensor as T
from .layers import *
from .shared_data_set import SharedDataSet 

class NeuralNetwork(object):
    """
    Parent class for creating a symbolic Feed forward neural network classifier or regressor.
    """
    def __init__(self, architecture):
        """
        architecture: list of tuples 

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
        # one liner for checking each key corresponds to a valid type
        assert sum(map(lambda l: l in LAYER_TYPES.keys(), [a[0] for a in architecture])) == len(architecture), \
               "Invalid layer type key in architecture."

        self.input              = T.matrix('Network input')
        self.architecture       = architecture
        self.n_layers           = len(architecture)+1 # count the input layer
        self.layers             = []
        self.params             = []

        for arc in self.architecture:
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
                 learning_rate         = 0.1,
                 n_epochs              = 10,
                 batch_size            = None,
                 L1_coef               = None,
                 L2_coef               = None,
                 act_sparse_func       = None,
                 act_sparse_list       = None,
                 act_sparse_coef       = None,
                 momentum              = None,
                 rand_seed             = None,
                 start_rand            = False,
                 use_early_stopping    = False,
                 variance_window       = 20,
                 variance_threshold    = 1e-3,
                 bootstrap             = False,
                 callback              = None
        ):
        """
        Train the network for a number of epochs (or use early variance stopping). Training and validation sets must be loaded by calling load_..._set(...)

        learning_rate: float
        step size of gradient descent

        n_epochs: int
        maximum number of epochs to cycle through gradient descent

        batch_size: int
        size of batches used for grad descent updates. If this number does not divide the size of the
        training set exactly, then the remainder is not utilized (assuming bootstrap is not used). If None,
        the size of the training_set is used.

        L1_coef: float
        amount of L1 regularization added weights 

        act_sparse_func: theano op
        Default None.

        act_sparse_list: list of bools
        list should be length #hidden layers + 1 for output.
        Each item in the list is a bool specifying if the sparsity is
        applied to the activation of units in that layer. Default: [True]*length

        act_sparse_coef: float
        amount of activation sparsity

        L2_coef: float
        amount of L2 regularization added weights

        momentum: float
        size of momentum coefficient. should be < 1

        rand_seed: int
        seed for the random number generator if start_rand is True. default None uses random rand_seed

        start_rand: bool
        If true, the network parameters are set to random values before initializing. False (default) uses the current network param values as starting points.

        use_early_stopping: bool
        if True, training will stop (at an epoch < n_epoch possibly) according to the variance of the
        the log loss of the training set for the last variance_window epochs

        variance_window: int
        see use_early_stopping.

        variance_treshold: float
        the tolerance that determines when the earliness of early stopping. I.e. if var_window=20
        and var_tol=1e-3, at each epoch, we look at the variance of the loss from the last 20 epochs
        on the training set. If the variance is less than var_tol, we exit.

        bootstrap: bool
        if True, bootstrap (sub)samples are used for each minibatch of grad descent. Note that this method is slower.
        """
        # print params
        opts = locals()
        p = "\n"
        for key in opts:
            if opts[key] is None or opts[key] is False: continue
            if (key == 'variance_window' or key == 'variance_threshold') and opts['use_early_stopping'] is False: continue
            p += key+": "+str(opts[key])+"\n"
        opts.pop('self')
        

        #################
        # input checking
        assert hasattr(self,'training_set') and hasattr(self,'validation_set'), \
               "Testing or validation set not present. You must load both via the \
               NeuralNetwork object's load_..._set(...) methods."
        assert batch_size <= self.training_set.N, \
               "Batch size cannot be greater than size of training set."
        if act_sparse_func is not None:
            assert act_sparse_coef is not None, \
                   "activation sparsity coefficient must be set"
            # check if list length matches
            assert len(act_sparse_list) == self.n_layers-1
        #################

        print ("\nBeginning new trial. Params:")
        print ( p )

        batch_size          = self.training_set.N if batch_size is None else batch_size
        n_train_batches     = int(np.floor(self.training_set.N / float(batch_size)))

        if not bootstrap: index   = T.lscalar('index')
        else:             indexes = T.ivector('indexes')

        if start_rand:
            print "... Randomizing network parameters"
            for param in self.params:
                param.set_value( (np.random.random( param.get_value().shape )-0.5).astype( param.dtype ))

        print "... Compiling"

        # Add any extra terms to the loss function (regularizers, etc...)
        cost = self.loss
        if L2_coef is not None:
            L2 = reduce(lambda a,b: a+b, map(T.sum, map(lambda x: x**2, self.params)))
            cost += L2_coef*L2
        if L1_coef is not None:
            L1 = reduce(lambda a,b: a+b, map(T.sum, map(T.abs_, self.params)))
            cost += L1_coef*L1
        if act_sparse_func is not None:
            for i in range(len(act_sparse_list)-1):
                if act_sparse_list[i]:
                    cost += act_sparse_coef*act_sparse_func(self.layers[i].output).mean()
            if act_sparse_list[-1]:
                cost += act_sparse_coef*act_sparse_func(self.output).mean()

        # compute symbolic gradient of the cost with respect to params
        gparams = []
        oparams = [] # old params for momentum
        for param in self.params:
            gparams.append( T.grad(cost=cost, wrt=param) )
            if momentum is not None:
                oparams.append(theano.shared(np.zeros(param.get_value().shape, dtype=param.dtype)))
        
        updates = []
        for i in xrange(len(self.params)):
            if momentum is not None:
                updates.append((self.params[i],
                                self.params[i] - learning_rate*(gparams[i] + momentum*oparams[i])))
            else:
                updates.append((self.params[i], self.params[i] - learning_rate*gparams[i]))
        if momentum is not None:
            for i in xrange(len(self.params)):
                updates.append((oparams[i], gparams[i]))

        if bootstrap:
            train_model = theano.function(
                inputs=[indexes],
                outputs=cost,
                updates=updates,
                givens={
                    self.input:    self.training.x[indexes],
                    self.response: self.training.y[indexes]
                }
            )
        else:
            train_model = theano.function(
                inputs=[index],
                outputs=cost,
                updates=updates,
                givens={
                    self.input:     self.training_set.x[index*batch_size:(index+1)*batch_size],
                    self.response:  self.training_set.y[index*batch_size:(index+1)*batch_size]
                }
            )

        has_miss = hasattr(self, 'miss')
        if has_miss:
            validation_cost_and_miss = theano.function(
                inputs=[],
                outputs=[cost, self.miss],
                givens={
                    self.input:     self.validation_set.x,
                    self.response:  self.validation_set.y
                }
            )
        else:
            validation_cost = theano.function(
                inputs=[],
                outputs=cost,
                givens={
                    self.input:     self.validation_set.x,
                    self.response:  self.validation_set.y
                }
            )

        print ("... Beginning training\n")
        
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
            
            if bootstrap:
                for minibatch_index in xrange(n_train_batches):
                    indices = np.random.randint(0, self.training_set.N, size=batch_size).astype(np.int32)
                    tr_cost += train_model(indices) / n_train_batches
            else:
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
                for i in xrange(len(self.params)):
                    best_params[i] = self.params[i].get_value().copy()

            # Early stopping condition:
            # If the variance in the validation curve has not changed significantly
            # over the most recent variance window, then quit.
            if use_early_stopping and epoch > variance_window:
                if np.var(loss_records[epoch-variance_window+1:epoch+1,0]) < variance_threshold:
                    print ("Variance threshold of validation record reached. Quitting.")
                    epoch +=1
                    break

            # call the call back
            if callback is not None: callback(locals())

            epoch += 1
        end_time = time.clock()

        for i in range(len(self.params)):
            self.params[i].set_value( best_params[i] )
        loss_records = loss_records[:epoch,:]

        self.training_stats = {
            'time': end_time-start_time,
            'rate': epoch/(end_time-start_time),
            'loss': loss_records,
            'para': best_params,
            'cost': cost, 
            'opts': opts,
            'epoch':best_epoch
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
    def __init__(self, architecture):
        __doc__ = super(NeuralNetworkClassifier, self).__init__.__doc__
        super(NeuralNetworkClassifier, self).__init__(architecture)

        if architecture[-1][1]['activation'] != T.nnet.softmax:
            print "Warning: activation function of output layer should be T.nnet.softmax for Classification."

        self.y_pred = T.argmax( self.output, axis=1 )
        self.output.name = 'Multinomial regression softmax output'
        self.y_pred.name = 'Multinomial regression hard-assignment output' 

        self.response  =  T.ivector('Mulinomial regression response variables (labels)')
        self.loss      = -T.mean(T.log(self.output)[T.arange(self.response.shape[0]), self.response])
        self.loss.name = 'Negative loglikelihood loss'
        self.miss      =  T.mean(T.neq(self.y_pred, self.response))
        self.miss.name = 'Misclassification error'

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
    def __init__(self, architecture):
        super(NeuralNetworkRegressor, self).__init__(architecture)

        self.response   = T.matrix('Regression response variable')
        self.loss       = ((self.output-self.response)**2).mean() 
        self.loss.name  = 'MSE loss'

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
