import numpy as np


def ReLU(x, derivative=False):
    """Compute the ReLU activation funciton and derivative."""
    if derivative:
        return (x > 0).astype(int).T
    else:
        return np.maximum(x, 0)


def stable_softmax(X):
    """Numerically more stable softmax."""
    z = X - np.max(X, axis=-1, keepdims=True)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=-1, keepdims=True)
    softmax = numerator / denominator
    return softmax


def cross_entropy_loss(t, y, derivative=False):
    """Compute cross entropy loss and derivative."""
    if np.shape(t) != np.shape(y):
        print("t and y have different shapes")
    if derivative:  # Return the derivative of the function
        return (y-t).T
    else:
        return t * np.log(y)


class NeuralNetwork:
    """ Neural network class"""
    
    def __init__(self, layers = [2, 3, 2], lr=0.001, momentum=0):
        """Initialize network.
        
        Args:
            layers (list of ints): Number of layers and hidden nodes.
            lr: Learning Rate

        """
        self.layers = layers
        self.momentum = momentum
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
            a = ReLU(z)
            a_s.append(a)
            z_s.append(z)
        
        # Output layer.
        yh = np.c_[a_s[-1], np.ones(n_pts)] @ self.W[-1]
        z_s.append(yh)
        # Softmax
        y = stable_softmax(yh)
        a_s.append(y)
        
        return y, a_s, z_s
    
    def backward(self, X, y):
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
        delta = cross_entropy_loss(y, yhat, derivative=True)
        Q = np.c_[a_s[-2], np.ones(a_s[-2].shape[0])].T @ delta.T
        dWs[-1] = Q
        
        for l in range(2, self.num_layers):
            d_z = ReLU(z_s[-l], derivative=True)
            delta = d_z * (self.W[-l+1][:-1,:] @ delta)
        
            Q = np.c_[a_s[-l-1], np.ones(a_s[-l-1].shape[0])].T @ delta.T
            dWs[-l] = Q
        
        # Update W
        for i, dW in enumerate(dWs):
            self.velocity[i] = self.momentum * self.velocity[i] + self.lr * dW/n 
            self.W[i] -= self.velocity[i]
    