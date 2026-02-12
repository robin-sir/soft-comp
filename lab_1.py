import numpy as np

def fuzzy_union_or(A, B, operator='max'):
    
    if len(A) != len(B):
        raise ValueError("Fuzzy sets must have the same length (same universe of discourse).")
    
    if operator == 'max':
        
        return np.maximum(A, B)
    
    else:
        raise NotImplementedError(f"Operator '{operator}' not supported for Fuzzy Union.")

def fuzzy_intersection_and(A, B, operator='min'):
    
    if len(A) != len(B):
        raise ValueError("Fuzzy sets must have the same length (same universe of discourse).")
        
    if operator == 'min':
        
        return np.minimum(A, B)
  
    else:
        raise NotImplementedError(f"Operator '{operator}' not supported for Fuzzy Intersection.")

def fuzzy_complement_not(A):
    
    
    return 1 - A

U = np.array([1, 2, 3, 4, 5]) 
print(f"Universe of Discourse (U): {U}\n")


A = np.array([1.0, 0.8, 0.4, 0.1, 0.0])
B = np.array([0.0, 0.1, 0.3, 0.7, 1.0])

print("--- Original Sets ---")
print(f"Fuzzy Set A: {A}")
print(f"Fuzzy Set B: {B}\n")


# a. Fuzzy UNION (OR)
A_OR_B = fuzzy_union_or(A, B)
print("--- Fuzzy UNION (A OR B) ---")
print(f"Operation: max(mu_A(x), mu_B(x))")
# Example: max(A[0], B[0]) = max(1.0, 0.0) = 1.0
print(f"Result (A OR B): {A_OR_B}\n")

# b. Fuzzy INTERSECTION (AND)
A_AND_B = fuzzy_intersection_and(A, B)
print("--- Fuzzy INTERSECTION (A AND B) ---")
print(f"Operation: min(mu_A(x), mu_B(x))")
# Example: min(A[1], B[1]) = min(0.8, 0.1) = 0.1
print(f"Result (A AND B): {A_AND_B}\n")

# c. Fuzzy COMPLEMENT (NOT)
NOT_A = fuzzy_complement_not(A)
print("--- Fuzzy COMPLEMENT (NOT A) ---")
print(f"Operation: 1 - mu_A(x)")
# Example: 1 - A[2] = 1 - 0.4 = 0.6
print(f"Result (NOT A): {NOT_A}\n")

NOT_B = fuzzy_complement_not(B)
print("--- Fuzzy COMPLEMENT (NOT B) ---")
print(f"Operation: 1 - mu_B(x)")
print(f"Result (NOT B): {NOT_B}")