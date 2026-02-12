#Implementation of Simple Neural Network (McCulloh Pitts model)
import numpy as np

# --- 1. Activation Function (Threshold Logic) ---

def mcp_activation(net_input, threshold):
    """
    The MCP neuron uses a fixed threshold activation.
    Output = 1 (Fires) if net_input >= threshold, else 0 (Does not fire).
    """
    return 1 if net_input >= threshold else 0

# --- 2. The MCP Neuron Function ---

def mcp_neuron(inputs, weights, threshold):
    """
    Simulates the McCulloch-Pitts neuron calculation.
    net_input = Sum(input_i * weight_i)
    Output = mcp_activation(net_input, threshold)
    """
    # Convert inputs and weights to NumPy arrays for easy dot product
    inputs = np.array(inputs)
    weights = np.array(weights)
    
    # Calculate the net input (Weighted Sum)
    net_input = np.dot(inputs, weights)
    
    # Determine the output based on the fixed threshold
    output = mcp_activation(net_input, threshold)
    
    return net_input, output

# --- 3. Implementation of Logical Gates ---

def implement_or_gate():
    """Simulates the Logical OR gate (Output is 1 if EITHER input is 1)."""
    
    print("\n--- Implementing Logical OR Gate ---")
    
    # Fixed parameters for OR Gate:
    # We want (1*1) + (1*0) >= 1  -> 1 >= 1 (True)
    # We want (0*0) + (0*0) >= 1  -> 0 >= 1 (False)
    weights = [1, 1]  # Equal importance for both inputs
    threshold = 1     # Fire if at least one weighted input is 1
    
    test_cases = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 1)
    ]
    
    print(f"Weights: {weights}, Threshold: {threshold}")
    print("Input (A, B) | Net Input | Output | Expected")
    print("-" * 38)
    
    for inputs, expected in test_cases:
        net_input, output = mcp_neuron(inputs, weights, threshold)
        print(f"  {inputs[0]}, {inputs[1]}    |   {net_input}      |    {output}   |   {expected}")


def implement_not_gate():
    """Simulates the Logical NOT gate (Output is the inverse of the input)."""
    
    print("\n--- Implementing Logical NOT Gate ---")
    
    # Fixed parameters for NOT Gate (Single Input):
    # We need a strong negative weight to inhibit firing when the input is 1.
    weights = [-1]    # Negative weight for the single input
    threshold = 0     # Needs to be low (non-positive)
    
    test_cases = [
        ([0], 1),
        ([1], 0)
    ]
    
    print(f"Weights: {weights}, Threshold: {threshold}")
    print("Input (A) | Net Input | Output | Expected")
    print("-" * 38)
    
    for inputs, expected in test_cases:
        net_input, output = mcp_neuron(inputs, weights, threshold)
        print(f"    {inputs[0]}   |    {net_input}      |    {output}   |   {expected}")

# --- 4. Run the Implementations ---

implement_or_gate()
implement_not_gate()