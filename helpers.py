import numpy as np
import pickle


def read_mnist(im_path='mnist/train-images-idx3-ubyte', 
               lab_path='mnist/train-labels-idx1-ubyte'):
    """Read file and labels."""

    f = open(im_path, 'r')
    # data type is signed integer big-endian
    a = np.fromfile(f, dtype='>i4', count=4)
    images = np.fromfile(f, dtype=np.uint8)
    f.close()

    f = open(lab_path, 'r')
    # data type is signed integer big-endian
    t = np.fromfile(f, count=2, dtype='>i4')
    labels = np.fromfile(f, dtype=np.uint8)
    f.close()
    
    ims = images.reshape(a[1:])

    return ims, labels, a, t


def format_data(ims, labels, num_classes=10):

    X = ims.reshape(ims.shape[0], -1).astype(np.float32) / 255
    y = np.eye(num_classes)[labels]
    
    return X, y


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

        
def split_data(X, y, percentage=80):
    """Split data into train and validation set."""
    np.random.seed(42)
    N = X.shape[0]

    arr_rand = np.random.rand(X.shape[0])
    train_idx = arr_rand < np.percentile(arr_rand, percentage)

    X_train, y_train = X[train_idx], y[train_idx]
    
    mask = np.ones(N, dtype=np.bool)
    mask[train_idx] = False
    
    X_val, y_val = X[mask], y[mask]

    return X_train, y_train, X_val, y_val


def accuracy(pred, y):
    """Get prediction accuracy from network output and one-hot-encoded labels."""
    return np.mean(np.argmax(pred,axis=1)==np.argmax(y, axis=1))


def dump_nn(nn, nn_path='models/nn.pkl'):
    with open(nn_path, 'wb') as file:
        pickle.dump(nn, file, pickle.HIGHEST_PROTOCOL)
    
def load_nn(nn_path):
    with open(nn_path, 'rb') as file:
        nn = pickle.load(file)
    return nn
