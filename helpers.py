import numpy as np

def ReLU(x, derivative=False):
    if derivative:
        return (x>0).astype(int)
    else:
        return np.maximum(x, 0)

def softmax(x, derivative=False):
    if derivative:
        return np.diagflat(x) 
    else:
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

def cross_entropy_loss(t, y, derivative=False):
    if np.shape(t)!=np.shape(y):
        print("t and y have different shapes")
    if derivative: # Return the derivative of the function
        return y-t
    else:
        return t * np.log(y)