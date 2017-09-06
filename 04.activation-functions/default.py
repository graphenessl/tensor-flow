def normal(x, w, b, t):

    weight  = w
    bias    = b
    input   = x

    sum = (input*weight) + bias
    activation_threshold = t

    if (sum > activation_threshold):
        return sum
