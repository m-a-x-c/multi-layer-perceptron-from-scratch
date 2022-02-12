import numpy as np

def sigmoid(x):
    return ( 1 / ( 1+np.exp(-1*x) ) )

def sigmoid_derivative(x):
    f = sigmoid(x)
    return np.multiply(f, (1-f))

def mse(y, y_pred):
    return (np.subtract(y, y_pred) ** 2)

def mse_derivative(y, y_pred):
    return (-2 * np.subtract(y, y_pred))

def hadamard_product(args):
    product = args[0]

    for i in range(1, len(args)):
        product = np.multiply(product, args[i])
    
    return product