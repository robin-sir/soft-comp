import numpy as np

# --- 1. Define Activation and Derivative Functions (Sigmoid) ---

def sigmoid(x):
    """Sigmoid activation function: 1 / (1 + e^(-x))"""
    # Prevents overflow in e^(-x) for very large negative numbers
    x = np.clip(x, -500, 500) 
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(output):
    """Derivative of the sigmoid function, calculated using the output of the sigmoid."""
    # Derivative is O * (1 - O), where O is the output of the sigmoid.
    return output * (1 - output)

# --- 2. Define the Multilayer Perceptron Class ---
class MLP_Backpropagation:
    """
    Implements a simple two-layer Multilayer Perceptron (Input -> Hidden -> Output).
    """
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, max_epochs=10000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        
        # Initialize Weights and Biases randomly (W_ih: Input to Hidden, W_ho: Hidden to Output)
        # Weights should be initialized small to prevent saturation of the sigmoid function.
        self.W_ih = np.random.uniform(low=-0.5, high=0.5, size=(input_size, hidden_size))
        self.b_h = np.zeros((1, hidden_size)) # Bias for hidden layer
        
        self.W_ho = np.random.uniform(low=-0.5, high=0.5, size=(hidden_size, output_size))
        self.b_o = np.zeros((1, output_size)) # Bias for output layer
        
        self.errors = [] # To track mean squared error

    def forward_pass(self, X):
        """
        Calculates the output for a given input X.
        Z = WX + b; A = f(Z)
        """
        # Hidden Layer Calculation (Input -> Hidden)
        self.net_h = np.dot(X, self.W_ih) + self.b_h
        self.out_h = sigmoid(self.net_h)
        
        # Output Layer Calculation (Hidden -> Output)
        self.net_o = np.dot(self.out_h, self.W_ho) + self.b_o
        self.out_o = sigmoid(self.net_o)
        
        return self.out_o

    def backward_pass(self, X, y, out_o, out_h):
        """
        Calculates and applies weight updates based on the error.
        This is the core of the Backpropagation algorithm.
        """
        # --- A. Output Layer Error and Delta ---
        # Error = Target - Output
        error_o = y - out_o
        # Output Delta (d_o) = Error * f'(net_o)
        d_o = error_o * sigmoid_derivative(out_o)
        
        # --- B. Hidden Layer Error and Delta (Error Backpropagation) ---
        # Error_h = Delta_o * W_ho^T 
        # The error is propagated back using the weights W_ho.
        error_h = d_o.dot(self.W_ho.T)
        # Hidden Delta (d_h) = Error_h * f'(net_h)
        d_h = error_h * sigmoid_derivative(out_h)
        
        # --- C. Weight and Bias Updates (Gradient Descent) ---
        
        # Update W_ho (Hidden to Output Weights)
        # dW_ho = out_h.T . d_o
        self.W_ho += self.out_h.T.dot(d_o) * self.learning_rate
        self.b_o += np.sum(d_o, axis=0, keepdims=True) * self.learning_rate
        
        # Update W_ih (Input to Hidden Weights)
        # dW_ih = X.T . d_h
        self.W_ih += X.T.dot(d_h) * self.learning_rate
        self.b_h += np.sum(d_h, axis=0, keepdims=True) * self.learning_rate
        
        return np.mean(error_o**2)

    def train(self, X_train, y_train):
        """
        Main training loop.
        """
        print(f"--- Training MLP (Hidden Size: {self.hidden_size}, Rate: {self.learning_rate}) ---")
        
        for epoch in range(self.max_epochs):
            # 1. Forward Pass
            out_o = self.forward_pass(X_train)
            
            # 2. Backward Pass (Calculate Error and Update Weights)
            mse = self.backward_pass(X_train, y_train, out_o, self.out_h)
            
            self.errors.append(mse)
            
            if (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch + 1}/{self.max_epochs}, Mean Squared Error: {mse:.6f}")

        print("\n--- Training Complete ---")

# --- 3. Prepare Data for XOR Gate (A Non-Linearly Separable Problem) ---

# Input data (A, B) - 4 samples, 2 features
X_train = np.array([
    [0, 0], 
    [0, 1], 
    [1, 0], 
    [1, 1]  
])

# Output labels (A XOR B) - 4 samples, 1 output
y_train = np.array([
    [0], 
    [1], 
    [1], 
    [0]
])

# --- 4. Run the MLP Model ---

# Model Configuration: 2 inputs, 4 hidden neurons, 1 output
mlp = MLP_Backpropagation(
    input_size=2, 
    hidden_size=4, 
    output_size=1, 
    learning_rate=0.2, 
    max_epochs=10000
)

# Train the model
mlp.train(X_train, y_train)

# --- 5. Test the Trained Model ---

print("\n--- Testing Model Predictions ---")

# Pass the training data through the trained network
predictions = mlp.forward_pass(X_train)

for inputs, prediction, expected in zip(X_train, predictions, y_train):
    # Apply a threshold of 0.5 to convert the sigmoid output (0 to 1) into a binary prediction (0 or 1)
    predicted_class = 1 if prediction[0] >= 0.5 else 0
    status = "Correct" if predicted_class == expected[0] else "Incorrect"
    
    # Print prediction as a continuous value and its thresholded class
    print(f"Input: {inputs}, Output: {prediction[0]:.4f}, Predicted Class: {predicted_class}, Expected: {expected[0]} ({status})")

# Optional: Visualize the error history
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(mlp.errors)
plt.title('MLP Training Error (Mean Squared Error) Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.grid(True)
plt.show()