import numpy as np

from nn import NeuralNetwork
from helpers import read_mnist, format_data, iterate_minibatches, split_data, accuracy, dump_nn
from augmentation import load_augmented

from deskew import deskew_all

def train_final(nn, X_train, y_train, epochs, batch_size):
    """Training loop."""
    
    t = range(epochs)
    for i in t:
        # Iterate over minibatches (backward includes both forward and backward step).
        for batch in iterate_minibatches(X_train, y_train, batch_size):
            x_batch, y_batch = batch
            nn.backward(x_batch, y_batch)

        print(f"Epoch {i}", flush=True)

    return None

# Read and format data.
ims, labels, a, t = read_mnist()
ims_aug, labels_aug = load_augmented()
X, y = format_data(ims, labels)
X = deskew_all(X)
X_aug, y_aug = format_data(ims_aug, labels_aug)
X = np.append(X, X_aug, axis=0)
y = np.append(y, y_aug, axis=0)

# Create nn.
nn = NeuralNetwork(layers=[784, 800, 10], momentum=0.9)

# Train.
epochs = 300
batch_size = 200
train_final(nn, X, y, epochs, batch_size)

dump_nn(nn, f'models/final_1h_deskew_aug.pkl')

