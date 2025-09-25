import numpy as np

# Activation functions
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

----------------------------------------------------
# Derivatives
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2



'''✅ Notes:

These derivatives are needed by backprop.py.

Keep all standard activations in one place for modularity.'''