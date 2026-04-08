# Implement Hopfield Network to store and recall binary patterns; demonstrate retrieval from noisy inputs.

import numpy as np
import matplotlib.pyplot as plt

# Convert 0/1 to -1/1
def to_bipolar(pattern):
    return np.where(pattern == 0, -1, 1)

# Hopfield Network class
class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))

    def train(self, patterns):
        for p in patterns:
            self.weights += np.outer(p, p)
        np.fill_diagonal(self.weights, 0)
        self.weights /= len(patterns)

    def recall(self, pattern, steps=10):
        s = pattern.copy()
        for _ in range(steps):
            for i in range(self.num_neurons):
                net_input = np.dot(self.weights[i], s)
                s[i] = 1 if net_input >= 0 else -1
        return s

# Define binary patterns (5x5)
patterns = [
    np.array([
        1,0,0,0,1,
        0,1,0,1,0,
        0,0,1,0,0,
        0,1,0,1,0,
        1,0,0,0,1
    ]),
    np.array([
        0,1,1,1,0,
        1,0,0,0,1,
        1,0,0,0,1,
        1,0,0,0,1,
        0,1,1,1,0
    ])
]

# Convert patterns to bipolar
patterns = [to_bipolar(p) for p in patterns]

# Create and train network
hopfield = HopfieldNetwork(num_neurons=25)
hopfield.train(patterns)

# Create noisy version of first pattern
noisy_pattern = patterns[0].copy()
noise_indices = np.random.choice(25, 6, replace=False)
noisy_pattern[noise_indices] *= -1

# Recall pattern
recalled_pattern = hopfield.recall(noisy_pattern)

# Plot patterns
def plot_pattern(pattern, title, pos):
    plt.subplot(1, 3, pos)
    plt.imshow(pattern.reshape(5,5), cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.figure(figsize=(9,3))
plot_pattern(patterns[0], "Original Pattern", 1)
plot_pattern(noisy_pattern, "Noisy Input", 2)
plot_pattern(recalled_pattern, "Recalled Pattern", 3)
plt.show()
