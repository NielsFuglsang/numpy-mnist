import numpy as np


def read_mnist():
    """Read file and labels."""
    f = open('mnist/t10k-images-idx3-ubyte', 'r')
    # data type is signed integer big-endian
    a = np.fromfile(f, dtype='>i4', count=4)
    images = np.fromfile(f, dtype=np.uint8)
    f.close()

    f = open('mnist/t10k-labels-idx1-ubyte', 'r')
    # data type is signed integer big-endian
    t = np.fromfile(f, count=2, dtype='>i4')
    labels = np.fromfile(f, dtype=np.uint8)
    f.close()
    
    ims = images.reshape(a[1:])

    return ims, labels, a, t


def ReLU(x, derivative=False):
    if derivative:
        return (x > 0).astype(int)
    else:
        return np.maximum(x, 0)


def stable_softmax(X):
    z = X - np.max(X, axis=-1, keepdims=True)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=-1, keepdims=True)
    softmax = numerator / denominator
    return softmax


def cross_entropy_loss(t, y, derivative=False):
    if np.shape(t) != np.shape(y):
        print("t and y have different shapes")
    if derivative:  # Return the derivative of the function
        return y-t
    else:
        return t * np.log(y)


def iterate_minibatches(inputs, targets, batchsize):
    """Generator for getting minibatches."""
    indices = np.arange(inputs.shape[0])
    np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        yield inputs[excerpt], targets[excerpt]
