# Implement Bidirectional Associative Memory (BAM) to associate input patterns A with output patterns B, and recall outputs from noisy inputs.

import numpy as np

# -----------------------------
# Step 1: Define input patterns (Letters)
# Using bipolar values: 1 and -1
# -----------------------------

A = np.array([
    [ 1, -1,  1, -1],   # Pattern A1 (Letter A)
    [-1,  1, -1,  1],   # Pattern A2 (Letter B)
    [ 1,  1, -1, -1]    # Pattern A3 (Letter C)
])

# -----------------------------
# Step 2: Define output patterns (Digits)
# -----------------------------

B = np.array([
    [ 1, -1, -1],   # Digit 1
    [-1,  1, -1],   # Digit 2
    [-1, -1,  1]    # Digit 3
])

# -----------------------------
# Step 3: Train BAM (Weight matrix)
# W = sum of outer products
# -----------------------------

W = np.zeros((A.shape[1], B.shape[1]))

for i in range(len(A)):
    W += np.outer(A[i], B[i])

print("Weight Matrix W:\n", W)

# -----------------------------
# Step 4: Recall with noisy input
# -----------------------------

# Noisy version of A1
A_noisy = np.array([1, -1, -1, -1])

# Forward recall (A → B)
B_recalled = np.sign(A_noisy @ W)

# Handle zero values
B_recalled[B_recalled == 0] = 1

print("\nNoisy Input A:", A_noisy)
print("Recalled Output B:", B_recalled)

# -----------------------------
# Step 5: Backward recall (B → A)
# -----------------------------

A_recalled = np.sign(B_recalled @ W.T)
A_recalled[A_recalled == 0] = 1

print("Recalled Input A:", A_recalled)
