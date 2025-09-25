import numpy as np
from activations import sigmoid, relu, tanh, sigmoid_derivative, relu_derivative, tanh_derivative

# Map activation names to derivative functions
activation_derivative_map = {
    'sigmoid': sigmoid_derivative,
    'relu': relu_derivative,
    'tanh': tanh_derivative
}

activation_map = {
    'sigmoid': sigmoid,
    'relu': relu,
    'tanh': tanh
}

# -------------------------------
# Backpropagation Function
# -------------------------------
def backpropagate(network, inputs, targets, learning_rate=0.01):
    """
    Performs a single backpropagation pass and updates network weights and biases.

    network: dict from initialize_network
    inputs: array-like input values
    targets: array-like true output values
    learning_rate: step size for gradient descent
    """

    # -------------------------------
    # 1. Forward pass (store z and a)
    # -------------------------------
    layer_inputs = list(inputs)
    zs = {}  # weighted sums
    activations_cache = {}  # activations

    for layer in network:
        layer_data = network[layer]
        act_func = activation_map[layer_data['activation']]
        layer_outputs = []
        layer_z = []

        for node in layer_data['nodes']:
            node_data = layer_data['nodes'][node]
            z = np.sum(layer_inputs * node_data['weights']) + node_data['bias']
            a = act_func(z)
            layer_z.append(z)
            layer_outputs.append(a)

        zs[layer] = np.array(layer_z)
        activations_cache[layer] = np.array(layer_outputs)
        layer_inputs = layer_outputs

    # -------------------------------
    # 2. Compute output layer delta
    # -------------------------------
    layers = list(network.keys())
    output_layer = layers[-1]
    y_pred = activations_cache[output_layer]
    y_true = np.array(targets)
    act_deriv_func = activation_derivative_map[network[output_layer]['activation']]
    delta = (y_pred - y_true) * act_deriv_func(zs[output_layer])
    deltas = {output_layer: delta}

    # -------------------------------
    # 3. Backpropagate deltas through hidden layers
    # -------------------------------
    for i in reversed(range(len(layers) - 1)):
        current_layer = layers[i]
        next_layer = layers[i + 1]

        weights_next = np.array([network[next_layer]['nodes'][n]['weights'] for n in network[next_layer]['nodes']])
        delta_next = deltas[next_layer]
        act_deriv_func = activation_derivative_map[network[current_layer]['activation']]
        deltas[current_layer] = (delta_next @ weights_next) * act_deriv_func(zs[current_layer])

    # -------------------------------
    # 4. Update weights and biases
    # -------------------------------
    for i, layer in enumerate(layers):
        inputs_to_use = np.array(inputs if i == 0 else activations_cache[layers[i - 1]])
        for j, node in enumerate(network[layer]['nodes']):
            network[layer]['nodes'][node]['weights'] -= learning_rate * deltas[layer][j] * inputs_to_use
            network[layer]['nodes'][node]['bias'] -= learning_rate * deltas[layer][j]

    return network
