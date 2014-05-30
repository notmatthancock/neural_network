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
        self.loss     = -T.mean(T.log(self.output)[T.arange(y.shape[0]), self.response])
        self.loss.name= 'Negative loglikelihood loss'
        self.miss     =  T.mean(T.neq(self.y_pred, self.response))
        self.miss.name= 'Misclassification error'
    
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, activation = T.tanh):
        """
        Construct a hidden layer of a multilayer perceptron.
        
        rng:      numpy.Random.RandomState
        random number generator for weight init
        
        input:    theano.tensor.matrix
        symbolic tensor of shape (n_examples, n_in)
        
        n_in:     int
        size of input dimension
        
        n_out:    int
        size of output dimension
        
        activation: theano.Op or function
        (non)linear activation function for hidden units
        """
        self.n_in   = n_in
        self.n_out  = n_out
        self.input  = input
        W_vals      = np.asarray(
            rng.uniform(
                low = -np.sqrt(6.0 / (n_in + n_out)),
                high=  np.sqrt(6.0 / (n_in + n_out)),
                size=(n_in, n_out)
            ).astype(theano.config.floatX)
        )
        if activation == theano.tensor.nnet.sigmoid:
            W_vals *= 4
        self.W = theano.shared(W_vals, name='W')
        self.b = theano.shared(np.zeros(n_out).astype(theano.config.floatX), name='b')
        self.params = [self.W, self.b]

        self.output = activation(T.dot(input, self.W) + self.b)
        self.output.name = 'Hidden activation output' 

class NeuralNetwork(object):
    """
    Create a symbolic (theano) Feed-forward neural net with a logistic or linear regressor on top for classification or regression.
    """
    def __init__(self, rng, architecture, activation, output_layer_type):
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
                    input       = self.input if h is 0 else self.hidden_layer[h-1].output,
                    n_in        = architecture[h],
                    n_out       = architecture[h+1],
                    activation  = activation[h]
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

    def train(   self,
                 train,
                 valid, 
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
                 bootstrap             = False
        ):
        """
        train: DataSet
        DataSet object holding the training data.

        valid: DataSet
        DataSet object holding the validation data.

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
        seed for the random number generator. default None uses random rand_seed

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
            p += key+": "+str(opts[key])+"\n"
        
        print ("\nBeginning new trial. Params:")
        print ( p )

        # Data formatting requirements are unique to the output layer type
        batch_size          = train.N if batch_size is None else batch_size
        n_train_batches     = int(np.floor(train.N / float(batch_size)))

        print ("... Constructing model")

        if not bootstrap:
            index = T.lscalar('index')
        else:
            indexes = T.ivector('indexes')

        if start_rand:
            print "... Randomizing network parameters"
            for param in self.params:
                param.set_value( (np.random.random( param.get_value().shape )-0.5).astype( param.dtype ))

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
                    self.input:    train.x[indexes],
                    self.response: train.y[indexes]
                }
            )
        else:
            train_model = theano.function(
                inputs=[index],
                outputs=cost,
                updates=updates,
                givens={
                    self.input:     train.x[index*batch_size:(index+1)*batch_size],
                    self.response:  train.y[index*batch_size:(index+1)*batch_size]
                }
            )

        validation_cost = theano.function(
            inputs=[],
            outputs=cost,
            givens={
                self.input:     valid.x,
                self.response:  valid.y
            }
        )

        print ("... Beginning training\n")
        
        start_time = time.clock()
        best_params = [None]*len(self.params)
        best_va_cost = np.inf
        epoch = 0
        loss_records = np.zeros((n_epochs,2))

        while epoch < n_epochs:
            tr_cost, va_cost = 0., validation_cost()
            
            if bootstrap:
                for minibatch_index in xrange(n_train_batches):
                    indices = np.random.randint(0, train.N, size=batch_size).astype(np.int32)
                    tr_cost += train_model(indices) / n_train_batches
            else:
                for minibatch_index in xrange(n_train_batches):
                    tr_cost += train_model(minibatch_index) / n_train_batches

            # record losses for epoch
            loss_records[epoch,0], loss_records[epoch,1] = tr_cost, va_cost
            sys.stdout.write("Epoch: %d || Trloss: %f || VaLoss: %f\r"%(epoch+1, tr_cost, va_cost))
            sys.stdout.flush()
            
            if va_cost < best_va_cost:
                # record new best
                best_va_cost = va_cost
                best_epoch = epoch
                for i in xrange(len(self.params)):
                    best_params[i] = self.params[i].get_value().copy()

            # Early stopping condition:
            # If the variance in the validation curve has not changed significantly over the most recent variance window, then quit.
            if use_early_stopping and \
               epoch > variance_window and \
               (np.var(loss_records[epoch-variance_window+1:epoch+1,0]) < variance_threshold or \
               (loss_records[epoch-variance_window+1,1] - loss_records[-1,1]) / variance_window > 0.01 ):
                print ("Variance threshold of validation record reached. Quitting.")
                epoch +=1
                break
            
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
            'epoch':best_epoch
        }

        print ("\n\n... Training finished")
        print ("Running at ~ %f epochs / sec"%(epoch/(end_time-start_time)))

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

class DataSet(object):
    def __init__(self, input_path, response_path=None, output_layer_type=None):
        """
        DataSet objects are containers for holding information about sets for training mlps.

        Input and output paths should point to a *.npy file
        input_path: string
        /absolute/path/to/input_data/as/string/*.npy

        output_path: string
        same as input_path but points to the datasets output (labels). If None, the input set is used for the output so that the network is trained in an unsupervised fashion.

        Example:
        tr = DataSet(input_path='/mydata/x.npy', output_path='/mydata/y.npy')
        """
        assert output_layer_type is not None and output_layer_type in OUTPUT_LAYER_TYPES, "output_layer_type_must be specified"
        response_path  = input_path if response_path is None else response_path
        
        self.x = theano.shared(np.load( input_path )   , name = 'x')
        self.y = theano.shared(np.load( response_path ), name = 'y')

        self.N = self.x.get_value().shape[0]
        assert self.N == self.y.get_value().shape[0], "Shape mismatch in data set."

        if output_layer_type is 'LogisticRegression':
            self.y = T.cast( self.y, 'int32' )
            assert self.y.ndim is 1, "Response variables should be contained in a one dimensional vector for Logistic Regression coded as unique integers per class label."
        elif output_layer_type is 'LinearRegression':
            assert self.y.ndim is 2, "Response variables should be contained in a matrix by row for Linear Regression"
