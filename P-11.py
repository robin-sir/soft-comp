# Implement Particle Swarm Optimization (PSO) to find the global minimum of the Rastrigin function.
import numpy as np

# Rastrigin function
def rastrigin(X):
    A = 10
    return A * len(X) + np.sum(X**2 - A * np.cos(2 * np.pi * X))

# PSO parameters
num_particles = 30
dimensions = 2
iterations = 100
w = 0.7       # inertia
c1 = 1.5      # cognitive
c2 = 1.5      # social

# Initialize particles
X = np.random.uniform(-5.12, 5.12, (num_particles, dimensions))
V = np.random.uniform(-1, 1, (num_particles, dimensions))

pbest = X.copy()
pbest_val = np.array([rastrigin(x) for x in X])

gbest = pbest[np.argmin(pbest_val)]
gbest_val = min(pbest_val)

# PSO loop
for _ in range(iterations):
    for i in range(num_particles):
        r1, r2 = np.random.rand(), np.random.rand()
        V[i] = (w * V[i] +
                c1 * r1 * (pbest[i] - X[i]) +
                c2 * r2 * (gbest - X[i]))
        X[i] = X[i] + V[i]

        fitness = rastrigin(X[i])
        if fitness < pbest_val[i]:
            pbest[i] = X[i]
            pbest_val[i] = fitness

    gbest = pbest[np.argmin(pbest_val)]
    gbest_val = min(pbest_val)

print("Global Minimum Position:", gbest)
print("Minimum Value:", gbest_val)
