import numpy as np

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

def activation_maximization(P,l,j,act_list):
    """
    activation_maximization finds a vector from the input space which maximizes the activation
    of the jth neuron in layer l of a network.

    P: list of ndarrays
    This should be from or formatted as the para attribute from the NeuralNetwork save_stats()
    dictionary.

    l: int
    layer num > 0

    j: int
    neuron num > 0

    act_list: list of ops
    """
    assert l > 0, "layer 0 is the input layer."
    assert l < len(P)/2, "you've entered a layer greater than how many exist"
    assert j > 0, "neuron 0 is the bias unit"
    assert j <= P[2*l].shape[1], "you've enter a neuron greater than how many exist"
    assert len(act_list) == len(P)/2
    
    from scipy.optimize import fmin_slsqp
    import theano
    input = theano.tensor.vector('input')
    for k in range(len(act_list)-1):
        output = act_list[k](theano.tensor.dot((input if k==0 else output),P[2*k]) + P[2*k+1])
    ft = theano.function(inputs=[input], outputs=output)

    gr = theano.grad(cost=output[j-1], wrt=input)
    gt = theano.function(inputs=[input], outputs=gr)

    f = lambda x: ft(x.astype(np.float32))
    g = lambda x: gt(x.astype(np.float32))
    
    constraint = lambda x: np.dot(x,x) - 1
    return fmin_slsqp(func=f, x0=np.ones((P[0].shape[0],)).astype(np.float32), fprime=g, f_eqcons=constraint)





