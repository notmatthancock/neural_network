import numpy as np
import theano
from theano import tensor as T


def image_from_weights(W, height_in, width_in, height_out, width_out):
    """
    W: 2d ndarray
    Matrix of filters flattened and stored in each column.

    height_in: int
    width_in: int
    Each column of W is reshaped to size, (height_in, height_out).

    height_out: int
    width_out: int
    The columns of W, now rectangular blocks, are tiled into a mosaic which is
    height_out blocks tall and width_out blocks wide.
    """
    assert W.shape[0] == height_in * width_in
    assert W.shape[1] == height_out * width_out
    I = np.zeros((height_in * height_out, width_in * width_out), dtype=W.dtype)
    n = 0
    mx = W.max()
    mn = W.min()
    for i in range(height_out):
        for j in range(width_out):
            I[i * height_in:(i + 1) * height_in, j * width_in:(j + 1) * width_in] = W[:, n].reshape(height_in, width_in)
            n += 1

    return (I - I.min()) / (I.max() - I.min())

def activation_maximization(input, output, direction, x0, bounds=(0., 1.)):
    """
    input: symbolic input that was used to construct output

    output: symbolic output function of some network layer

    direction: direction in which to be maximized.

    x0: initial guess for optimization routine

    Returns the x value such that dot(f(x), n) is maximal.
    
    Example:

    result = activation_maximization(net.input, net.layers[0].output, eye(input_shape)[:,0], rand(input_shape))

    This would find the input, x, which maximizes the response of the first unit in the first
    hidden layer.
    """
    if abs(np.linalg.norm(direction)-1) > 1e-6:
        direction /= np.linalg.norm(direction)
    direction = theano.shared(direction.astype(input.dtype), name='direction')

    cost_ = -theano.tensor.dot(output.flatten(), direction)
    cost_.name = 'cost'
    cost  = theano.function([input], cost_, allow_input_downcast=True)
    grad_ = theano.grad(cost=cost_, wrt=input)
    grad_.name = 'grad'
    grad  = theano.function([input], grad_, allow_input_downcast=True)

    x0 = x0.astype(input.dtype)

    from scipy.optimize import fmin_l_bfgs_b as fmin
    bounds = [bounds for i in range(x0.shape[0])]
    result = fmin(
        func    = lambda x: np.asscalar(cost(x[np.newaxis,:].astype(np.float32))),
        x0      = x0,
        fprime  = lambda x: grad(x[np.newaxis,:].astype(np.float32)).flatten().astype(np.float64),
        bounds  = bounds,
        disp    = 1
    )

    return result
