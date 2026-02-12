import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# --- 1. Define the Antecedent (Input) and Consequent (Output) Variables ---

# New Antecedent/Consequent objects hold universe variables and membership functions
# The universe (range of possible values) for each variable is defined.
# Universe for 'service' quality, ranging from 0 to 10
service = ctrl.Antecedent(np.arange(0, 11, 1), 'service')
# Universe for 'food' quality, ranging from 0 to 10
food = ctrl.Antecedent(np.arange(0, 11, 1), 'food')
# Universe for 'tip' percentage, ranging from 0 to 25%
tip = ctrl.Consequent(np.arange(0, 26, 1), 'tip')

# --- 2. Define Fuzzy Membership Functions (Fuzzification) ---

# We use simple triangular and trapezoidal membership functions (trimf, trapmf)

# Service quality fuzzy sets
service['poor'] = fuzz.trimf(service.universe, [0, 0, 5])
service['acceptable'] = fuzz.trimf(service.universe, [0, 5, 10])
service['excellent'] = fuzz.trimf(service.universe, [5, 10, 10])

# Food quality fuzzy sets
food['bad'] = fuzz.trapmf(food.universe, [0, 0, 1, 3])
food['decent'] = fuzz.trimf(food.universe, [1, 5, 9])
food['great'] = fuzz.trapmf(food.universe, [7, 9, 10, 10])

# Tip percentage fuzzy sets
tip['low'] = fuzz.trimf(tip.universe, [0, 0, 13])
tip['medium'] = fuzz.trimf(tip.universe, [0, 13, 25])
tip['high'] = fuzz.trimf(tip.universe, [13, 25, 25])

# Optional: Visualize the membership functions
# service.view()
# food.view()
# tip.view()

# --- 3. Define the Fuzzy Rules (Rule Base) ---

# Rules are defined in a natural language-like IF-THEN format
rule1 = ctrl.Rule(service['poor'] | food['bad'], tip['low'])
rule2 = ctrl.Rule(service['acceptable'], tip['medium'])
rule3 = ctrl.Rule(service['excellent'] & food['great'], tip['high'])
rule4 = ctrl.Rule(food['decent'] & service['poor'], tip['medium'])

# --- 4. Create the Control System and Simulation ---

# The ControlSystem object holds the fuzzy rules
tip_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4])

# The ControlSystemSimulation object allows us to pass inputs and get outputs
tipping_simulation = ctrl.ControlSystemSimulation(tip_control)

# --- 5. Fuzzify, Infer, Aggregate, and Defuzzify (The Process) ---

# Pass the input values to the simulation (Crisp Inputs)
# Example: Service is 6.5/10, Food is 9.8/10
tipping_simulation.input['service'] = 6.5
tipping_simulation.input['food'] = 9.8

# Compute the result (runs the entire FIS process: Fuzzification -> Inference -> Defuzzification)
tipping_simulation.compute()

# Get the crisp output value
tip_amount = tipping_simulation.output['tip']

# --- 6. Display Results ---

print(f"Service Rating: 6.5/10")
print(f"Food Rating: 9.8/10")
print(f"*** Recommended Tip: {tip_amount:.2f}% ***")

# Optional: Visualize the final result
# The `tip.view` method shows the aggregated output fuzzy set
# and the resulting crisp value (vertical line)
tip.view(sim=tipping_simulation)
plt.show()

# --- Example 2: Testing different inputs ---
print("\n--- Example 2: Poor Service/Bad Food ---")
tipping_simulation.input['service'] = 2
tipping_simulation.input['food'] = 3
tipping_simulation.compute()
print(f"Recommended Tip: {tipping_simulation.output['tip']:.2f}%")