import numpy as np
import theano, time, cPickle, uuid, sys
import theano.tensor as T

OUTPUT_LAYER_TYPES = ['LinearRegression', 'LogisticRegression']

class LinearRegression(object):
    def __init__(self, input, n_in, n_out):
        """
        Initialize the symbolic theano parameters for a linear regression layer.
        
        input: theano.tensor.TensorType
        symbolic variable that describes the input of the architecture
        
        n_in: int
        number of input units, the dimension of the space in which the datapoint lies
        
        n_out: int
        number of output units, the dimension of space in which the target lies
        """
        self.n_in  = n_in
        self.n_out = n_out

        self.W = theano.shared(
            np.zeros( (n_in, n_out)
        ).astype(theano.config.floatX), name='W', borrow=True)

        self.b      = theano.shared(np.zeros( n_out ).astype(theano.config.floatX), name='b', borrow=True)
        self.params = [self.W, self.b]

        self.output      = T.dot(input, self.W) + self.b
        self.output.name = 'Linear regression output'

        self.response   = T.matrix('Linear regression response variable')
        self.loss       = ((self.output-self.response)**2).mean() 
        self.loss.name  = 'MSE loss'


class LogisticRegression(object):
    def __init__(self, input, n_in, n_out):
        """
        Initialize symbolic theano parameters for logistic regression model.
        
        input: theano.tensor.TensorType
        symbolic variable that describes the input of the architecture
        
        n_in: int
        number of input units
        
        n_out: int
        number of output units
        """
        assert n_out > 1, "n_out should be at least 2 (this is binary classification!)."

        self.n_in  = n_in
        self.n_out = n_out

        self.W = theano.shared(
            np.zeros( (n_in, n_out)
        ).astype(theano.config.floatX), name='W', borrow=True)

        self.b      = theano.shared(np.zeros( n_out ).astype(theano.config.floatX), name='b', borrow=True)
        self.params = [self.W, self.b]

        self.output = T.nnet.softmax( T.dot(input, self.W) + self.b )
        self.y_pred = T.argmax( self.output, axis=1 )
        self.output.name = 'Logistic regression softmax output'
        self.y_pred.name = 'Logistic regression hard-assignment output' 

        self.response =  T.ivector('Logistic regression response variable')
        self.loss     = -T.mean(T.log(self.output)[T.arange(self.response.shape[0]), self.response])
        self.loss.name= 'Negative loglikelihood loss'
        self.miss     =  T.mean(T.neq(self.y_pred, self.response))
        self.miss.name= 'Misclassification error'

class HiddenLayer(object):
    def __init__(self, input, n_in, n_out, activation, rng=None, weight_paths=None):
        """
        Construct a hidden layer of a neural network.
        
        input:    theano.tensor.matrix
        symbolic tensor of shape (n_examples, n_in)
        
        n_in:     int
        size of input dimension
        
        n_out:    int
        size of output dimension
        
        activation: theano.Op or function
        (non)linear activation function for hidden units

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

        self.n_in   = n_in
        self.n_out  = n_out
        self.input  = input

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

        self.output = activation(T.dot(input, self.W) + self.b)
        self.output.name = 'Hidden activation output' 

class NeuralNetwork(object):
    """
    Create a symbolic (theano) Feed-forward neural net with a logistic or linear regressor on top for classification or regression.
    """
    def __init__(self, rng, architecture, activation, output_layer_type, hidden_weight_paths=None):
        """
        rng: numpy.random.RandomState
        A random number generator used to initialize weights
        
        architecture: list
        list of ints describing the size of each layer and hence the arch(itecture) of the network. For instance:
        
        architecture = [4,3,2,1]
        
        would describe a network with 4 inputs, 3 hidden units in the first hidden layer, 2 inputs in the second hidden layer, and one output.
        
        activation: list/function.
        activation is of size len(arch)-2 or a single function. If a list is given, elements are functions or theano ops that specify the activation function for each hidden layer (respectively). For instance:
        
        act = [theano.tensor.nnet.sigmoid, lambda x: theano.tensor.exp(-x**2)]
        
        would yield a sigmoid activation from the first hidden layer, and a gaussian for the second.
        Alternatively, if a single function is given, it is applied to all hidden layers.
        
        output_layer_type: string
        A string from mlp._ouputs(). This specifies the final output object of the network. In doing so, it also specifies the loss function. For instance if output_layer_type = 'LinearRegression', the output layer will have linear units and use mean squared error for the loss. 

        hidden_weight_paths: list of lists
        You can specify the values of hidden weights. This list should have as many elements as hidden layers, so:
        len(hidden_weight_paths) == len(architecture)-2
        Each element can be NoneType or a len 2 list specifying weight paths. See HiddenLayer weight_paths argument for more detail.
        For instance:
        hidden_weight_paths = [['/my_W_matrix.npy', None], None]
        initializes the weight matrix (and not the bias) of the first hidden layer only for a Net with 2 hidden layers.
        """
        # Error checks
        assert len(architecture) > 2, 'arc should be greater than two for MLP'
        for i in range(len(architecture)):
            assert architecture[i] > 0 and isinstance(architecture[i], int), 'elements of arc should be ints > 0'
        if isinstance(activation, list):
            assert len(activation) == len(architecture) - 2, 'list of activations should correspond to # hidden layers'
        else:
            activation = [activation] * (len(architecture) - 2)
        assert output_layer_type in OUTPUT_LAYER_TYPES, 'output should be one of %s' % str(OUTPUT_LAYER_TYPES)
        if hidden_weight_paths is not None:
            assert len(hidden_weight_paths) == len(architecture)-2, "weight path list be have as many elements as hidden layers"
            # assert sum(map(len,hidden_weight_paths)) == 2*(len(architecture)-2), "Each element of weight paths must len 2 list. See HiddenLayer object."
        else:
            hidden_weight_paths = [None]*(len(architecture)-2)
        # End error checks

        self.rng                = rng
        self.input              = T.matrix('Network input')
        self.architecture       = architecture
        self.activation         = activation
        self.output_layer_type  = output_layer_type
        self.n_layers           = len(architecture)
        self.n_hidden_layers    = len(architecture)-2
        self.hidden_layer       = []
        self.params             = []

        # Build the symbolic architecture
        for h in range(self.n_hidden_layers):
            self.hidden_layer.append(
                HiddenLayer(
                    rng         = rng,
                    input       = self.input if h == 0 else self.hidden_layer[h-1].output,
                    n_in        = architecture[h],
                    n_out       = architecture[h+1],
                    activation  = activation[h],
                    weight_paths= hidden_weight_paths[h]
                )
            )
            self.params += self.hidden_layer[h].params
        self.output_layer = globals()[self.output_layer_type](
            input   = self.hidden_layer[-1].output,
            n_in    = self.hidden_layer[-1].n_out,
            n_out   = self.architecture[-1]
        )
        self.params += self.output_layer.params
        self.output  = self.output_layer.output
        self.response= self.output_layer.response
        self.loss    = self.output_layer.loss

    def load_training_set(self, input_path, response_path=None):
        self.training_set = SharedDataSet( input_path=input_path,
                                           response_path=response_path,
                                           output_layer_type=self.output_layer_type
                                         )
    def load_validation_set(self, input_path, response_path=None):
        self.validation_set = SharedDataSet( input_path=input_path,
                                               response_path=response_path,
                                               output_layer_type=self.output_layer_type
                                              )

    def load_testing_set(self, input_path, response_path=None):
        self.testing_set = SharedDataSet( input_path=input_path,
                                          response_path=response_path,
                                          output_layer_type=self.output_layer_type
                                        )
    def train(   self,
                 learning_rate         = 0.1,
                 n_epochs              = 10,
                 batch_size            = None,
                 L1_coef               = None,
                 L2_coef               = None,
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
        amount of L1 regularization added to the cost function

        L2_coef: float
        amount of L2 regularization added to the cost function

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
        opts = locals()
        p = "\n"
        for key in opts:
            if opts[key] is None or opts[key] is False: continue
            if (key == 'variance_window' or key == 'variance_threshold') and opts['use_early_stopping'] is False: continue
            p += key+": "+str(opts[key])+"\n"
        opts.pop('self')
        
        assert hasattr(self,'training_set') and hasattr(self,'validation_set'), "Testing or validation set not present. You must load both via the NeuralNetork object's load_..._set(...) methods."
        assert batch_size <= self.training_set.N, "Batch size cannot be greater than size of training set."

        print ("\nBeginning new trial. Params:")
        print ( p )

        # Data formatting requirements are unique to the output layer type
        batch_size          = self.training_set.N if batch_size is None else batch_size
        n_train_batches     = int(np.floor(self.training_set.N / float(batch_size)))

        if not bootstrap:
            index = T.lscalar('index')
        else:
            indexes = T.ivector('indexes')

        if start_rand:
            print "... Randomizing network parameters"
            for param in self.params:
                param.set_value( (np.random.random( param.get_value().shape )-0.5).astype( param.dtype ))

        print "... Compiling"

        # Create symbolic cost function for gradient descent
        cost = self.loss
        if L2_coef is not None:
            L2 = reduce(lambda a,b: a+b, map(T.sum, map(lambda x: x**2, self.params)))
            cost += L2_coef*L2
        if L1_coef is not None:
            L1 = reduce(lambda a,b: a+b, map(T.sum, map(T.abs_, self.params)))
            cost += L1_coef*L1

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

        if self.output_layer_type == 'LogisticRegression':
            validation_cost_and_miss = theano.function(
                inputs=[],
                outputs=[cost,self.output_layer.miss],
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
        
        start_time = time.clock()
        best_params = [None]*len(self.params)
        best_va = np.inf
        va_miss = None
        epoch = 0
        loss_records = np.zeros((n_epochs,3 if self.output_layer_type == 'LogisticRegression' else 2))
        #from scipy.misc import imsave,imresize
        #from image_from_weights import image_from_weights as ifw
        while epoch < n_epochs:
            tr_cost = 0.
            if self.output_layer_type == 'LogisticRegression':
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
            #if epoch%100 == 0:
            #    imsave(
            #        './W1/'+'0'*(6-len(str(epoch)))+str(epoch)+'.png',
            #        imresize(ifw(self.params[0].get_value(),28,28,20,20), size=(700,700))
            #    )
            #    imsave(
            #        './W2/'+'0'*(6-len(str(epoch)))+str(epoch)+'.png',
            #        imresize(ifw(self.params[2].get_value(),20,20,10,10), size=(700,700))
            #    )

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
            
            if (va_miss if self.output_layer_type=='LogisticRegression' else va_cost) < best_va:
                # record new best
                best_va = (va_miss if self.output_layer_type=='LogisticRegression' else va_cost) 
                best_epoch = epoch
                for i in xrange(len(self.params)):
                    best_params[i] = self.params[i].get_value().copy()

            # Early stopping condition:
            # If the variance in the validation curve has not changed significantly over the most recent variance window, then quit.
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
        if self.output_layer_type == 'LogisticRegression':
            test_miss = theano.function( 
                inputs=[],
                outputs=self.output_layer.miss,
                givens={
                    self.input:     self.testing_set.x,
                    self.response:  self.testing_set.y
                }
            )
            self.testing_stats['miss'] = test_miss()
    def save_stats(self, save_path=None, prefix=None):
        if save_path is None:
            save_path = self.__str__().replace(' ', '').replace('\n',' ')+'---'+str(uuid.uuid4())+'.dat'
        if prefix is not None:
            save_path = prefix+save_path

        assert hasattr(self,'training_stats'), "Train to get stats, then save."

        f=open(save_path,'wb')
        
        if hasattr(self,'training_stats') and hasattr(self,'testing_stats'):
            cPickle.dump({'training_stats': self.training_stats, 'testing_stats': self.testing_stats, 'arc': self.architecture, 'act': self.activation}, f)
        else: # no testing stats
            cPickle.dump({'training_stats': self.training_stats, 'arc': self.architecture, 'act': self.activation}, f)
        f.close()

    def __str__(self):
        s = str(self.n_layers) + ' layered MLP:\n'
        s += str(self.architecture[0]) + ' in => '
        for i in range(self.n_hidden_layers):
            s += str(self.architecture[i+1]) + 'h '
            if hasattr(self.activation[i], '__name__'):
                s += '( ' + self.activation[i].__name__ + ' ) '
            elif hasattr(self.activation[i], '__str__'):
                s += '( ' + self.activation[i].__str__() + ' ) '
            s += ' => '
        return s + str(self.architecture[-1]) + 'out ( ' + self.output_layer_type + ' )'

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
        
        self.x = theano.shared(np.load( input_path )   , name = 'x')
        self.y = theano.shared(np.load( response_path ), name = 'y')

        self.N = self.x.get_value().shape[0]
        assert self.N == self.y.get_value().shape[0], "Shape mismatch in data set."

        if output_layer_type == 'LogisticRegression':
            self.y = T.cast( self.y, 'int32' )
            assert self.y.ndim == 1, "Response variables should be contained in a one dimensional vector for Logistic Regression coded as unique integers per class label."
        elif output_layer_type == 'LinearRegression':
            assert self.y.ndim == 2, "Response variables should be contained in a matrix by row for Linear Regression"
