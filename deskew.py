"""Code for deskewing MNIST images. Inspired by https://fsix.github.io/mnist/Deskewing.html."""
from scipy.ndimage import interpolation
import numpy as np

def moments(image):
    """Calculate mean and covariance matrix of image."""
    c0,c1 = np.mgrid[:image.shape[0],:image.shape[1]] # A trick in numPy to create a mesh grid
    totalImage = np.sum(image) #sum of pixels
    m0 = np.sum(c0*image)/totalImage #mu_x
    m1 = np.sum(c1*image)/totalImage #mu_y
    m00 = np.sum((c0-m0)**2*image)/totalImage #var(x)
    m11 = np.sum((c1-m1)**2*image)/totalImage #var(y)
    m01 = np.sum((c0-m0)*(c1-m1)*image)/totalImage #covariance(x,y)
    mu_vector = np.array([m0,m1]) # Notice that these are \mu_x, \mu_y respectively
    covariance_matrix = np.array([[m00,m01],[m01,m11]]) # Do you see a similarity between the covariance matrix
    return mu_vector, covariance_matrix


def deskew(image):
    """Deskew image based on the image moments."""
    c,v = moments(image)
    alpha = v[0,1]/v[0,0]
    affine = np.array([[1,0],[alpha,1]])
    ocenter = np.array(image.shape)/2.0
    offset = c-np.dot(affine,ocenter)
    return interpolation.affine_transform(image,affine,offset=offset)

def deskew_all(X):
    """
    Deskew multiple images in np array stored as (n, p) where n
    is observations and p is pixels. This assumes a 28x28 image.
    """
    if X.ndim<2:
        return deskew(X.reshape(28,28)).ravel()

    deskewed = np.empty_like(X)
    for i in range(X.shape[0]):
        deskewed[i,:] = deskew(X[i].reshape(28,28)).ravel()
    
    return deskewed
