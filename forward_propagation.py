"""
=========================================
🧠 Neural Network Template (From Scratch)
=========================================

Fill in the blanks below to quickly build and test different architectures.
"""

import numpy as np

# -------------------------------
# 1. Define your network structure
# -------------------------------

config = {
    "num_inputs": ___,              # e.g. 5
    "num_hidden_layers": ___,       # e.g. 3
    "hidden_nodes": [___, ___, ___],# e.g. [3, 2, 3]
    "num_outputs": ___              # e.g. 1
}


# -------------------------------
# 2. Initialize Network
# -------------------------------

def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
    num_nodes_previous = num_inputs
    network = {}

    for layer in range(num_hidden_layers + 1):

        # Name layers
        if layer == num_hidden_layers:
            layer_name = "output"
            num_nodes = num_nodes_output
        else:
            layer_name = f"layer_{layer + 1}"
            num_nodes = num_nodes_hidden[layer]

        # Initialize weights & bias
        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = f"node_{node + 1}"
            network[layer_name][node_name] = {
                "weights": np.around(np.random.uniform(size=num_nodes_previous), 2),
                "bias": np.around(np.random.uniform(size=1), 2),
            }

        num_nodes_previous = num_nodes

    return network


# -------------------------------
# 3. Core Math Functions
# -------------------------------

def compute_weighted_sum(inputs, weights, bias):
    return np.sum(inputs * weights) + bias

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

# Add more activations as needed...


# -------------------------------
# 4. Forward Propagation
# -------------------------------

def forward_propagate(network, inputs, activation=sigmoid, verbose=True):
    layer_inputs = list(inputs)

    for layer in network:
        layer_data = network[layer]
        layer_outputs = []

        for layer_node in layer_data:
            node_data = layer_data[layer_node]
            node_output = activation(
                compute_weighted_sum(layer_inputs, node_data["weights"], node_data["bias"])
            )
            layer_outputs.append(np.around(node_output[0], 4))

        if verbose and layer != "output":
            print(f"Outputs of {layer}: {layer_outputs}")

        layer_inputs = layer_outputs

    return layer_outputs


# -------------------------------
# 5. Try it out
# -------------------------------

# Initialize your network
my_network = initialize_network(
    config["num_inputs"],
    config["num_hidden_layers"],
    config["hidden_nodes"],
    config["num_outputs"]
)

# Generate random inputs
inputs = np.around(np.random.uniform(size=config["num_inputs"]), 2)
print(f"\nInputs: {inputs}")

# Forward propagate
predictions = forward_propagate(my_network, inputs, activation=sigmoid)
print(f"\nFinal predictions: {predictions}")
