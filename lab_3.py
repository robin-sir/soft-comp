import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

def demonstrate_defuzzification(universe, aggregated_mf):
    """
    Applies and compares five common defuzzification techniques to a given 
    aggregated membership function (MF).
    """
    
    # 1. Centroid (CoG/CoM) - Most common and robust
    # Calculates the center of gravity of the area under the curve.
    cog = fuzz.defuzz(universe, aggregated_mf, 'centroid')
    
    # 2. Bisector (BoA)
    # Finds the vertical line that divides the area under the curve into two equal halves.
    boa = fuzz.defuzz(universe, aggregated_mf, 'bisector')
    
    # 3. Mean of Maximum (MoM)
    # Calculates the average of all points in the universe that have the maximum membership value (height).
    mom = fuzz.defuzz(universe, aggregated_mf, 'mom')
    
    # 4. Smallest of Maximum (SoM)
    # Calculates the smallest value in the universe that has the maximum membership value.
    som = fuzz.defuzz(universe, aggregated_mf, 'som')
    
    # 5. Largest of Maximum (LoM)
    # Calculates the largest value in the universe that has the maximum membership value.
    lom = fuzz.defuzz(universe, aggregated_mf, 'lom')
    
    # --- Print Results ---
    print("--- Defuzzification Results ---")
    print(f"Centroid (CoG):        {cog:.4f}")
    print(f"Bisector (BoA):        {boa:.4f}")
    print(f"Mean of Maximum (MoM): {mom:.4f}")
    print(f"Smallest of Max (SoM): {som:.4f}")
    print(f"Largest of Max (LoM):  {lom:.4f}")

    # --- Plot Visualization ---
    plt.figure(figsize=(10, 6))
    plt.plot(universe, aggregated_mf, 'b', linewidth=2.5, label='Aggregated Fuzzy Set')
    
    # Plot the results of each defuzzification method
    plt.axvline(cog, color='r', linestyle='--', label=f'Centroid ({cog:.2f})')
    plt.axvline(boa, color='g', linestyle='-.', label=f'Bisector ({boa:.2f})')
    plt.plot([mom, mom], [0, 1.0], 'k:', label=f'MoM ({mom:.2f})') # Using a vertical line for MoM
    plt.plot([som, som], [0, 1.0], 'c:', label=f'SoM ({som:.2f})')
    plt.plot([lom, lom], [0, 1.0], 'm:', label=f'LoM ({lom:.2f})')
    
    plt.title('Comparison of Defuzzification Techniques')
    plt.ylabel('Membership Degree')
    plt.xlabel('Output Universe')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True)
    plt.show()
    
# --- 1. Define the Universe of Discourse and Fuzzy Set ---

# The universe (range of possible output values, e.g., tip percentage 0 to 25)
X = np.arange(0, 26, 0.1)

# An example of an aggregated output fuzzy set (a trapezoid + a triangle)
# In a real FIS, this is the result of applying all rules (Inference) and 
# combining the outputs (Aggregation).
# We simulate a skewed/complex aggregated set to show differences in methods.
mf_1 = fuzz.trapmf(X, [0, 5, 8, 11])
mf_2 = fuzz.trimf(X, [9, 15, 25])

# The final aggregated set is usually the maximum (union) of all rule consequences
# Note: Since the output is the result of aggregation, it is an array of membership values.
aggregated_mf = np.fmax(mf_1 * 0.7, mf_2 * 1.0) # Assume mf_1 was scaled down by rule strength 0.7

# Run the demonstration
demonstrate_defuzzification(X, aggregated_mf)