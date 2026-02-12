import numpy as np
import matplotlib.pyplot as plt

# --- 1. Define the Activation Function (Step Function) ---
def step_function(weighted_sum):
    """
    The Perceptron uses a Heaviside step function as its activation.
    It returns 1 (activate) if the weighted sum is non-negative, else 0 (deactivate).
    """
    return 1 if weighted_sum >= 0 else 0

# --- 2. Define the Perceptron Class ---
class Perceptron:
    """
    Implements the core learning logic of a Single-Layer Perceptron.
    """
    def __init__(self, num_inputs, learning_rate=0.1, max_epochs=100):
        # Initialize weights (W) and bias (b). We start with small random weights.
        # W has size (num_inputs) and b is a single scalar.
        self.weights = np.random.uniform(low=-0.5, high=0.5, size=num_inputs)
        self.bias = np.random.uniform(low=-0.5, high=0.5, size=1)
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        
        # History to track error over epochs
        self.errors = []

    def predict(self, inputs):
        """
        Calculates the weighted sum and applies the step function.
        Output = step( (W . X) + b )
        """
        # (W . X) is the dot product of weights and input features
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        
        # Apply the activation function
        return step_function(weighted_sum)

    def train(self, training_inputs, labels):
        """
        Trains the perceptron using the Perceptron Learning Rule.
        Weights are updated only when a misclassification occurs.
        """
        print(f"--- Training Perceptron (Epochs: {self.max_epochs}, Rate: {self.learning_rate}) ---")
        
        for epoch in range(self.max_epochs):
            total_error = 0
            
            # Iterate through each training example
            for inputs, label in zip(training_inputs, labels):
                
                # Forward Pass: Predict the output
                prediction = self.predict(inputs)
                
                # Calculate the error
                error = label - prediction
                total_error += abs(error)
                
                # Backward Pass: Update weights and bias only if error is non-zero
                if error != 0:
                    # Perceptron Update Rule: 
                    # W_new = W_old + LR * error * X
                    # b_new = b_old + LR * error * 1
                    self.weights += self.learning_rate * error * inputs
                    self.bias += self.learning_rate * error * 1 # Bias update
                    
            # Record error for tracking convergence
            self.errors.append(total_error)
            
            # Check for convergence: Stop if all samples are classified correctly
            if total_error == 0:
                print(f"Converged successfully at Epoch {epoch + 1}.")
                break
                
            if (epoch + 1) % 10 == 0:
                 print(f"Epoch {epoch + 1}/{self.max_epochs}, Total Error: {total_error}")

        print("\n--- Training Complete ---")
        print(f"Final Weights: {self.weights}")
        print(f"Final Bias: {self.bias[0]:.4f}")


# --- 3. Prepare Data for AND Gate (A Linearly Separable Problem) ---

# Input data (A, B)
X_train = np.array([
    [0, 0], # Input 1
    [0, 1], # Input 2
    [1, 0], # Input 3
    [1, 1]  # Input 4
])

# Output labels (A AND B)
y_train = np.array([0, 0, 0, 1])

# --- 4. Run the Perceptron Model ---

# Initialize the Perceptron with 2 inputs (features)
perceptron = Perceptron(num_inputs=X_train.shape[1], learning_rate=0.1, max_epochs=50)

# Train the model
perceptron.train(X_train, y_train)

# --- 5. Test the Trained Model ---

print("\n--- Testing Model Predictions ---")
test_cases = X_train
test_labels = y_train

for inputs, expected in zip(test_cases, test_labels):
    prediction = perceptron.predict(inputs)
    status = "Correct" if prediction == expected else "Incorrect"
    print(f"Input: {inputs}, Predicted: {prediction}, Expected: {expected} ({status})")

# Optional: Visualize the error history
plt.figure(figsize=(8, 4))
plt.plot(perceptron.errors, marker='o')
plt.title('Perceptron Training Error Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Total Misclassifications')
plt.grid(True)
plt.show()