import numpy as np
from helpers import stable_softmax, iterate_minibatches

class NeuralNetwork:
    """ Neural network class"""
    
    def __init__(self, layers = [2, 3, 2], lr=0.001):
        """Initialize network.
        
        Args:
            layers (list of ints): Number of layers and hidden nodes.
            lr: Learning Rate

        """
        self.layers = layers
        self.num_layers = len(layers)
        self.W = self.init_weights(layers)
        self.lr = lr

        # Momentum velocity initalization.
        self.velocity = [np.zeros(self.W[i].shape) for i in range(self.num_layers - 1)]
    
    def init_weights(self, layers):
        """Initialize normal distributed weights.

        Args:
            layers (list of ints): Number of layers and hidden nodes.
        
        Returns:
            Weigths as a list of num_layers-1 elements.

        """
        W = []
        for i in range(self.num_layers-1):
            W.append(np.random.normal(loc=0, scale=np.sqrt(2/layers[i]), size=(layers[i]+1, layers[i+1])))
        return W
    
    def forward(self, X):
        """Forward pass through network.

        Args:
            X (np array): Observations of size (n, p) with n samples and p features.
        
        Returns:
            Tuple of (y, a_s, z_s):
                y: network output.
                a_s: list of activations.
                z_s: list of hidden node values.
        
        """
        # Number of observations.
        n_pts = X.shape[0]
        
        a_s = [X]
        z_s = []
        
        # Hidden layers.
        for i in range(self.num_layers - 2):
            
            z = np.c_[a_s[i], np.ones(n_pts)] @ self.W[i]
            a = np.maximum(z, 0)
            a_s.append(a)
            z_s.append(z)
        
        # Output layer.
        yh = np.c_[a_s[-1], np.ones(n_pts)] @ self.W[-1]
        z_s.append(yh)
        # Softmax
        y = stable_softmax(yh)
        a_s.append(y)
        
        return y, a_s, z_s
    
    def backward(self, X, y, momentum=0.9):
        """Backpropagation

        Args:
            X (np array): Observations of size (n, p) with n samples and p features.
            y (np array): Targets with n samples.

        Returns:
            Nothing. Updates network weights.
        
        """
        n = X.shape[0]
        dWs = [0 for _ in range(len(self.W))]
        
        yhat, a_s, z_s = self.forward(X)
        # Last layer
        delta = (yhat-y).T
        Q = np.c_[a_s[-2], np.ones(a_s[-2].shape[0])].T @ delta.T
        dWs[-1] = Q
        
        for l in range(2, self.num_layers):
            d_z = (z_s[-l]>0).astype(int).T
            delta = d_z * (self.W[-l+1][:-1,:] @ delta)
        
            Q = np.c_[a_s[-l-1], np.ones(a_s[-l-1].shape[0])].T @ delta.T
            dWs[-l] = Q
        
        # Update W
        for i, dW in enumerate(dWs):
            self.velocity[i] = momentum * self.velocity[i] + self.lr * dW/n 
            self.W[i] -= self.velocity[i]
    
    def train(self, X, y, batch_size=100, epochs=1):
        """Train neural network.
        
        Args:
            X (np array): Observations of size (n, p) with n samples and p features.
            y (np array): Targets with n samples.
            batch_size (int): Size of minibatches.
            epochs (int): Number of times to go through observations.
        
        Returns:
            Nothing. Updates model weights.
        
        """
        for i in range(epochs):
            for batch in iterate_minibatches(X, y, batch_size):
                x_batch, y_batch = batch
                self.backward(x_batch, y_batch)
