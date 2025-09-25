import numpy as np
from network_init import initialize_network, forward_propagate
from backprop import backpropagate
from activations import sigmoid
from loss_functions import loss_map  # contains 'mse' and 'cross_entropy'

# -------------------------------
# Training Loop
# -------------------------------
def train_network(network, X, y, epochs=1000, learning_rate=0.01, loss_type='mse'):
    """
    Trains a neural network using backpropagation.

    network: dict from initialize_network
    X: 2D array of inputs (num_samples x num_features)
    y: 2D array of true outputs (num_samples x num_outputs)
    epochs: number of training iterations
    learning_rate: step size for gradient descent
    loss_type: 'mse' or 'cross_entropy'
    """
    if loss_type not in loss_map:
        raise ValueError(f"Unsupported loss_type '{loss_type}'. Supported: {list(loss_map.keys())}")
    loss_func = loss_map[loss_type]

    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(len(X)):
            inputs = np.array(X[i])
            targets = np.array(y[i])

            # Forward propagation
            predictions, zs, activations = forward_propagate(network, inputs, verbose=False)

            # Compute loss
            loss = loss_func(targets, predictions)
            epoch_loss += loss

            # Backpropagation and weight update
            network = backpropagate(network, inputs, targets, learning_rate=learning_rate)

        # Print average loss every 100 epochs
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss / len(X):.6f}")

    return network


# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    # XOR dataset
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])

    # Initialize network with per-layer activations
    network = initialize_network(
        num_inputs=2,
        num_hidden_layers=2,
        num_nodes_hidden=[2,2],
        num_nodes_output=1,
        activations=['sigmoid', 'sigmoid', 'sigmoid']
    )

    # Train network using cross-entropy loss
    trained_network = train_network(network, X, y, epochs=1000, learning_rate=0.5, loss_type='cross_entropy')

    # Test network
    print("\nPredictions after training:")
    for i in range(len(X)):
        pred, _, _ = forward_propagate(trained_network, X[i], verbose=False)
        print(f"Input: {X[i]} - Predicted: {np.around(pred[0],4)} - True: {y[i][0]}")
