import numpy as np

# ============================================================================
# WHAT WE'RE BUILDING:
# A tiny neural network that learns XOR (exclusive OR)
# XOR truth table: [0,0]→0, [0,1]→1, [1,0]→1, [1,1]→0
# 
# Architecture: 2 inputs → 3 hidden neurons → 1 output
# ============================================================================

# Initialize random parameters (the "knobs" we'll adjust during training)
np.random.seed(42)  # Use same random numbers each time for reproducibility

# W1: "weights" for first layer - controls how inputs affect hidden neurons
# Shape (2, 3) means: 2 inputs connecting to 3 hidden neurons
W1 = np.random.randn(2, 3) * 0.5

# b1: "biases" for first layer - shifts the hidden neuron activations
# Shape (3,) means: one bias value per hidden neuron
b1 = np.random.randn(3) * 0.5

# W2: weights for second layer - controls how hidden neurons affect output
# Shape (3, 1) means: 3 hidden neurons connecting to 1 output
W2 = np.random.randn(3, 1) * 0.5

# b2: bias for output layer
b2 = np.random.randn(1) * 0.5


def relu(x):
    """
    ReLU activation function: keeps positive values, zeros out negatives
    Example: relu([-1, 2, -3]) = [0, 2, 0]
    This adds nonlinearity - without it, the network could only learn linear functions
    """
    return np.maximum(0, x)


def forward(x):
    """
    Pass an input through the network to get output
    
    x: input array [a, b]
    Returns: (output, hidden) where
        - output is the final result
        - hidden is the intermediate layer activations (useful for debugging)
    
    Steps:
    1. First layer: x @ W1 + b1  (linear transformation)
    2. Apply ReLU (nonlinearity)
    3. Second layer: hidden @ W2 + b2 (linear transformation)
    """
    hidden = relu(x @ W1 + b1)  # First layer with activation
    output = hidden @ W2 + b2    # Second layer (no activation on output)
    return output, hidden


# ============================================================================
# TRAINING DATA
# ============================================================================
# X: all possible 2-bit inputs
X = np.array([[0,0], [0,1], [1,0], [1,1]])

# y: corresponding XOR outputs
y = np.array([[0], [1], [1], [0]])


# ============================================================================
# SHOW INITIAL STATE (before training)
# ============================================================================
print("BEFORE TRAINING:")
print("Parameters W1:")
print(W1)
print("\nTesting on XOR inputs:")
for i in range(4):
    out, hidden = forward(X[i])
    print(f"Input {X[i]} → Hidden {hidden} → Output {out[0]:.3f} (want {y[i][0]})")


# ============================================================================
# TRAINING LOOP
# ============================================================================
learning_rate = 0.1  # How big of steps to take when adjusting parameters

for epoch in range(5000):  # Run through all training examples 5000 times
    total_loss = 0  # Track how wrong we are overall
    
    # Process each training example
    for i in range(4):
        # ---- FORWARD PASS ----
        # Calculate what the network currently outputs for this input
        x = X[i]
        hidden = relu(x @ W1 + b1)  # Hidden layer activations
        output = hidden @ W2 + b2    # Final output
        
        # ---- CALCULATE ERROR ----
        # How far off are we from the correct answer?
        error = output - y[i]
        total_loss += error**2  # Square it (standard loss function)
        
        # ---- BACKWARD PASS (Backpropagation) ----
        # Calculate gradients: how should we adjust each parameter to reduce error?
        
        # Output layer gradients
        d_output = 2 * error  # Derivative of squared error
        d_W2 = np.outer(hidden, d_output)  # How much W2 contributed to error
        d_b2 = d_output  # How much b2 contributed to error
        
        # Hidden layer gradients
        d_hidden = (d_output * W2.T).flatten()  # Error flowing back to hidden layer
        d_hidden[hidden <= 0] = 0  # ReLU gradient: zero out where neurons were inactive
        d_W1 = np.outer(x, d_hidden)  # How much W1 contributed to error
        d_b1 = d_hidden  # How much b1 contributed to error
        
        # ---- UPDATE PARAMETERS ----
        # Adjust parameters in direction that reduces error
        W1 -= learning_rate * d_W1  # Move W1 opposite to gradient
        b1 -= learning_rate * d_b1
        W2 -= learning_rate * d_W2
        b2 -= learning_rate * d_b2
    
    # Print progress every 1000 iterations
    if epoch % 1000 == 0:
        print(f"\nEpoch {epoch}, Loss: {total_loss[0]:.4f}")


# ============================================================================
# SHOW FINAL STATE (after training)
# ============================================================================
print("\n\nAFTER TRAINING:")
print("Parameters W1:")
print(W1)
print("\nTesting on XOR inputs:")
for i in range(4):
    out, hidden = forward(X[i])
    print(f"Input {X[i]} → Hidden {hidden} → Output {out[0]:.3f} (want {y[i][0]})")