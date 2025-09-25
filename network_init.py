import numpy as np
from activations import sigmoid, relu, tanh  # import multiple activations

# -------------------------------
# Network Initialization
# -------------------------------
def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output, activations=None):
    """
    Initialize network weights, biases, and per-layer activation functions.
    activations: list of activation names for each layer (hidden + output)
    """
    num_nodes_previous = num_inputs
    network = {}

    if activations is None:
        activations = ['sigmoid'] * (num_hidden_layers + 1)  # default all layers to sigmoid

    for layer in range(num_hidden_layers + 1):
        layer_name = "output" if layer == num_hidden_layers else f"layer_{layer+1}"
        num_nodes = num_nodes_output if layer == num_hidden_layers else num_nodes_hidden[layer]

        network[layer_name] = {
            'nodes': {},
            'activation': activations[layer]
        }

        for node in range(num_nodes):
            node_name = f"node_{node+1}"
            network[layer_name]['nodes'][node_name] = {
                'weights': np.around(np.random.uniform(size=num_nodes_previous), 2),
                'bias': np.around(np.random.uniform(size=1), 2)
            }

        num_nodes_previous = num_nodes

    return network

# -------------------------------
# Forward Propagation
# -------------------------------
activation_map = {
    'sigmoid': sigmoid,
    'relu': relu,
    'tanh': tanh
}

def compute_weighted_sum(inputs, weights, bias):
    return np.sum(inputs * weights) + bias

def forward_propagate(network, inputs, verbose=True):
    """
    Forward propagate inputs through the network.
    Returns final outputs, weighted sums (zs), and activations (a) for all layers.
    """
    layer_inputs = list(inputs)
    layer_zs = {}
    layer_activations = {}

    for layer in network:
        layer_data = network[layer]
        act_func = activation_map[layer_data['activation']]

        outputs = []
        zs = []

        for node in layer_data['nodes']:
            node_data = layer_data['nodes'][node]
            z = compute_weighted_sum(layer_inputs, node_data['weights'], node_data['bias'])
            a = act_func(z)
            zs.append(z)
            outputs.append(np.around(a[0], 4))

        if verbose and layer != 'output':
            print(f"Outputs of {layer}: {outputs}")

        layer_inputs = outputs
        layer_zs[layer] = np.array(zs)
        layer_activations[layer] = np.array(outputs)

    return layer_inputs, layer_zs, layer_activations
