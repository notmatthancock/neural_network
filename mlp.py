import numpy as np
import theano
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
        # Params
        self.W = theano.shared(
            np.zeros( (n_in, n_out)
        ).astype(theano.config.floatX), name='W', borrow=True)
        self.b = theano.shared(np.zeros( n_out ).astype(theano.config.floatX), name='b', borrow=True)
        self.params = [self.W, self.b]

        self.output = (T.dot(input, self.W) + self.b).flatten()

    def loss(self, y):
        """
        Predict mean squared error using current model params over y.
        """
        return ((self.output - y) ** 2).mean()

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
        # Params
        self.W = theano.shared(
            np.zeros( (n_in, n_out)
        ).astype(theano.config.floatX), name='W', borrow=True)
        self.b = theano.shared(np.zeros( n_out ).astype(theano.config.floatX), name='b', borrow=True)
        self.params = [self.W, self.b]

        
        self.output = T.nnet.softmax( T.dot(input, self.W) + self.b )
        self.y_pred = T.argmax( self.output, axis=1 )

    def loss(self, y):
        """
        Return the mean of the negative log-likelihood of the prediction of this model under a given target distribution.
        """
        return -T.mean(T.log(self.output)[T.arange(y.shape[0]), y])

    def misclass_error(self, y):
        """
        Return the number of examples in y misclassified / total number in y
        """
        if y.ndim != self.y_pred.ndim:
            raise TypeError('Input should have same dimension as self.y_pred.')
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        raise NotImplementedError()


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

class NeuralNetwork(object):
    """
    Create a symbolic (theano) Feed-forward neural net with a logistic or linear regressor on top for classification or regression.
    """
    def __init__(self, rng, input, arc, act, output_layer_type):
        """
        rng: numpy.random.RandomState
        A random number generator used to initialize weights
        
        input: theano.tensor.TensorType
        Symbolic variable that describes the input of the architecture (one minibatch)
        
        arc: list
        list of ints describing the size of each layer and hence the arch(itecture) of the network. For instance:
        
        arc = [4,3,2,1]
        
        would describe a network with 4 inputs, 3 hidden units in the first hidden layer, 2 inputs in the second hidden layer, and one output.
        
        act: list/function.
        act(ivations) is of size len(arch)-2 or a single function. If a list is given, elements are functions or theano ops that specify the activation function for each hidden layer (respectively). For instance:
        
        act = [theano.tensor.nnet.sigmoid, lambda x: theano.tensor.exp(-x**2)]
        
        would yield a sigmoid activation from the first hidden layer, and a gaussian for the second.
        Alternatively, if a single function is given, it is applied to all hidden layers.
        
        output_layer_type: string
        A string from mlp._ouputs(). This specifies the final output object of the network. In doing so, it also specifies the loss function. For instance if output_layer_type = 'LinearRegression', the output layer will have linear units and use mean squared error for the loss. 
        """
        # Error checks
        assert len(arc) > 2, 'arc should be greater than two for MLP'
        for i in range(len(arc)):
            assert arc[i] > 0 and isinstance(arc[i], int), 'elements of arc should be ints > 0'
        if isinstance(act, list):
            assert len(act) == len(arc) - 2, 'list of activations should correspond to # hidden layers'
        else:
            act = [act] * (len(arc) - 2)
        assert output_layer_type in OUTPUT_LAYER_TYPES, 'output should be one of %s' % str(OUTPUT_LAYER_TYPES)
        # End error checks

        self.arc                = arc
        self.act                = act
        self.output_layer_type  = output_layer_type
        self.n_layers           = len(arc)
        self.n_hidden_layers    = len(arc)-2
        self.hidden_layer       = []
        self.params             = []

        # Build the symbolic architechture
        for h in range(self.n_hidden_layers):
            self.hidden_layer.append(
                HiddenLayer(
                    rng         = rng,
                    input       = input if h is 0 else self.hidden_layer[h-1].output,
                    n_in        = arc[h],
                    n_out       = arc[h+1],
                    activation  = act[h]
                )
            )
            self.params += self.hidden_layer[h].params
        self.output_layer = globals()[self.output_layer_type](
            input   = self.hidden_layer[-1].output,
            n_in    = self.hidden_layer[-1].n_out,
            n_out   = self.arc[-1]
        )
        self.params += self.output_layer.params
        self.output  = self.output_layer.output
        self.output.name = "Neural network feed-forward output"

        if False:
            self.hiddenLayer = HiddenLayer(rng=rng, input=input, n_in=n_in, n_out=n_hidden, activation=activation)
            self.logregLayer = LogisticRegression(input=self.hiddenLayer.output, n_in=n_hidden, n_out=n_out)
            self.params = self.hiddenLayer.params + self.logregLayer.params
            self.L1 = abs(self.hiddenLayer.W).sum() + abs(self.logregLayer.W).sum()
            self.L2 = (self.hiddenLayer.W ** 2).sum() + (self.logregLayer.W ** 2).sum()
            self.neg_log_likelihood = self.logregLayer.neg_log_likelihood
            self.misclass_error = self.logregLayer.misclass_error

    def train(   self,
                 data_path, 
                 learning_rate         = 0.1,
                 n_epochs              = 1000,
                 batch_size            = 1000,
                 L1_reg                = 0.,
                 L2_reg                = 0.,
                 momentum              = 0.,
                 rand_seed             = None,
                 start_rand            = False,
                 use_early_stopping    = False,
                 variance_window       = 20,
                 variance_threshold    = 1e-3,
                 bootstrap             = False,
                 verbose               = True ):
        ):
        """
        data_path: string
        /absolute/path/to/data/ as string
        In this dir, there should be a training, xtr.npy, ytr.npy, validation, xtv.npy, ytv.npy, and test, xts.npy, yts.npy, sets. 

        learning_rate: float
        step size of gradient descent

        n_epochs: int
        maximum number of epochs to cycle through gradient descent

        batch_size: int
        size of batches used for grad descent updates. If this number does not divide the size of the
        training set exactly, then the remainder is not utilized (assuming bootstrap is not used).

        L1_reg: float
        amount of L1 regularization added to the cost function

        L2_reg: float
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

        verbose: bool
        if True, print updates of training process.
        """
        loc = locals()
        p = "\n"
        for key in loc:
            p += key+": "+str(loc[key])+"\n"
        
        def print_if_verbose(s):
            if verbose:
                print s

        print_if_verbose("\nBeginning new trial. Params:")
        print_if_verbose(p)
        print_if_verbose("... Loading data from %s"%path)

        xtr = theano.shared(np.load(data_path+'xtr.npy')
        ytr = T.cast( theano.shared(data_path+'ytr.npy', 'int32') )

        xtv = theano.shared(np.load(data_path+'xtv.npy')
        ytv = T.cast( theano.shared(data_path+'ytv.npy', 'int32') )

        xts = theano.shared(np.load(data_path+'xts.npy')
        yts = T.cast( theano.shared(data_path+'yts.npy', 'int32') )

        training_set_size = xtr.get_value(borrow=True).shape[0]
        n_train_batches = int(np.ceil(training_set_size / float(batch_size)))

        print_if_verbose("... Constructing model")

        # allocate symbolic vars for data
        index = T.lscalar('index')
        indexes = T.ivector('indexes')
        x = T.matrix('x') # input
        y = T.ivector('y') # response

        rng = np.random.RandomState(rand_seed)

        classifier = NNet.NeuralNetwork(rng=rng, input=x, n_in=28**2, n_hidden=n_hidden, n_out=10, activation=hidden_activation)
        cost = classifier.neg_log_likelihood(y) + L1_reg*classifier.L1 + L2_reg*classifier.L2

        # compute gradient of the cost with respect to params
        gparams = []
        oparams = [] # old params for momentum
        for param in classifier.params:
            gparams.append(T.grad(cost=cost, wrt=param))
            oparams.append(theano.shared(np.zeros(param.get_value().shape, dtype=theano.config.floatX)))
        
        updates = []
        for i in xrange(len(classifier.params)):
            updates.append((classifier.params[i],
                            classifier.params[i] - learning_rate*(gparams[i] + momentum*oparams[i])))
        for i in xrange(len(classifier.params)):
            updates.append((oparams[i], gparams[i]))

        if bootstrap:
            train_model = theano.function(
                inputs=[indexes],
                outputs=cost,
                updates=updates,
                givens={
                    x: xtr[indexes],
                    y: ytr[indexes]
                }
            )
        else:
            train_model = theano.function(
                inputs=[index],
                outputs=cost,
                updates=updates,
                givens={
                    x: xtr[index*batch_size:(index+1)*batch_size],
                    y: ytr[index*batch_size:(index+1)*batch_size]
                }
            )

        get_tr_loss = theano.function(
            inputs=[],
            outputs=cost,
            givens={
                x: xtr,
                y: ytr
            }
        )

        get_va_loss = theano.function(
            inputs=[],
            outputs=cost,
            givens={
                x: xtv,
                y: ytv
            }
        )

        valid_misclass = theano.function(
            inputs=[],
            outputs=classifier.misclass_error(y),
            givens={
                x: xtv,
                y: ytv
            }
        )

        test_misclass = theano.function(
            inputs=[],
            outputs=classifier.misclass_error(y),
            givens={
                x: xts,
                y: yts
            }
        )
        
        print_if_verbose("... Beginning training\n\n")
        
        start_time = time.clock()
        best_params = [None]*len(classifier.params)
        best_va_loss = np.inf
        best_ts_misclass = None
        epoch = 0
        loss_records = np.zeros((n_epochs,2))

        while epoch < n_epochs:
            tr_loss, va_loss = 0., get_va_loss()
            
            if bootstrap:
                for minibatch_index in xrange(n_train_batches):
                    indices = np.random.randint(0, training_set_size, size=batch_size).astype(np.int32)
                    tr_loss += train_model(indices) / n_train_batches
            else:
                for minibatch_index in xrange(n_train_batches):
                    tr_loss += train_model(minibatch_index) / n_train_batches

            # record losses for epoch
            loss_records[epoch,0], loss_records[epoch,1] = tr_loss, va_loss
            if verbose:
                sys.stdout.write("Epoch: %d || Trloss: %f || VaLoss: %f\r"%(epoch+1, tr_loss, va_loss))
                sys.stdout.flush()
            
            if va_loss < best_va_loss:
                # record new best
                best_va_loss = va_loss
                best_ts_misclass = test_misclass()
                best_epoch = epoch
                for i in xrange(len(classifier.params)):
                    best_params[i] = classifier.params[i].get_value().copy()

            # Early stopping condition:
            # If the variance in the validation curve has not changed significantly over the most recent variance window, then quit.
            if use_early_stopping and \
               epoch > variance_window and \
               (np.var(loss_records[epoch-variance_window+1:epoch+1,0]) < variance_threshold or \
               (loss_records[epoch-variance_window+1,1] - loss_records[-1,1]) / variance_window > 0.01 ):
                print_if_verbose("Variance threshold of validation record reached. Quitting.")
                epoch +=1
                break
            
            epoch += 1
        end_time = time.clock()

        for i in range(len(classifier.params)):
            classifier.params[i].set_value( best_params[i] )
        loss_records = loss_records[:epoch,:]

        classifier.training_stats = {
            'test': best_ts_misclass,
            'time': end_time-start_time,
            'rate': epoch/(end_time-start_time),
            'loss': loss_records,
            'para': best_params,
            'epoch':best_epoch
        }

        print_if_verbose("\n\n... Training finished")
        print_if_verbose("Best Test misclass: %f%% found at epoch, %d."%(best_ts_misclass*100, epoch) )
        print_if_verbose("Running at ~ %f epochs / sec"%(epoch/(end_time-start_time)))

        return classifier

    def __str__(self):
        s = str(self.n_layers) + ' layered MLP:\n'
        s += str(self.arc[0]) + ' in => '
        for i in range(self.n_hidden_layers):
            s += str(self.arc[i+1]) + 'h '
            if hasattr(self.act[i], '__name__'):
                s += '( ' + self.act[i].__name__ + ' ) '
            elif hasattr(self.act[i], '__str__'):
                s += '( ' + self.act[i].__str__() + ' ) '
            s += ' => '
        return s + str(self.arc[-1]) + 'out ( ' + self.output_layer_type + ' )'
