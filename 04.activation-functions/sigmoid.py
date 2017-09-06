# Problems:
# Gradient directions go too far with [sigmoids]
# Values are between [0] and [1]. It is not zero

import numpy as np

def sigmoid(x, derivative=False):
    if (derivative == True):
        return x * (1-x)

    return 1 / (1 + np.exp(-x))
