# Values: [-1, 1]
# Problems:
# Still suffers from gradient directioning
import numpy as np

def tanh(x, derivative=False):

    if(derivative == True):
        return (1-(x**2))
    return np.tanh(x)
