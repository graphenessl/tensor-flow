# It is used as an activation function only for the hidden layers
# Problems:
# It can result in a lot of dead neurons (which dont pass values anymore)
def relu(x):

    if x < 0:
        x = 0

    return x
