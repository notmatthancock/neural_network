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
