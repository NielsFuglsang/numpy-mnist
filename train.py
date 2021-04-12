import numpy as np

from nn import NeuralNetwork
from helpers import read_mnist, format_data, iterate_minibatches, split_data, accuracy, dump_nn
from augmentation import load_augmented

from deskew import deskew_all

def train(nn, X_train, y_train, X_val, y_val, epochs, batch_size):
    """Training loop."""
     
    accuracy_train = []
    accuracy_val = []
    
    t = range(epochs)
    for i in t:
        # Iterate over minibatches (backward includes both forward and backward step).
        for batch in iterate_minibatches(X_train, y_train, batch_size):
            x_batch, y_batch = batch
            nn.backward(x_batch, y_batch)

        # Train accuracy.
        yhat_train, _, _ = nn.forward(X_train)
        acc_train = accuracy(yhat_train, y_train)
        accuracy_train.append(acc_train)

        # Validation accuracy.
        yhat_val, _, _ = nn.forward(X_val)
        acc_val = accuracy(yhat_val, y_val)
        accuracy_val.append(acc_val)
        print(f"Epoch {i}, Train acc: {acc_train}, Val acc: {acc_val}", flush=True)

    return accuracy_train, accuracy_val


# Read and format data.
ims, labels, a, t = read_mnist()
# ims_aug, labels_aug = load_augmented()
X, y = format_data(ims, labels)
X = deskew_all(X)
# X_aug, y_aug = format_data(ims_aug, labels_aug)
X_train, y_train, X_val, y_val = split_data(X, y, percentage=80)
# X_train = np.append(X_train, X_aug, axis=0)
# y_train = np.append(y_train, y_aug, axis=0)

# Create nn.
nn = NeuralNetwork(layers=[784, 800, 10], momentum=0.9)

# Train.
epochs = 300
batch_size = 200
accuracy_train, accuracy_val = train(nn, X_train, y_train, X_val, y_val, epochs, batch_size)

yhat_val, _, _ = nn.forward(X_val)
acc = accuracy(yhat_val, y_val)

dump_nn(nn, f'models/{acc:.3f}.pkl')

