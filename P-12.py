# Use PSO to solve load balancing: assign N independent tasks to M machines to minimize makespan

import numpy as np
import random

# -------------------------------
# Problem Parameters
# -------------------------------
N = 6                      # number of tasks
M = 3                      # number of machines
task_time = np.array([2, 4, 6, 3, 5, 7])

num_particles = 20
iterations = 50
w, c1, c2 = 0.7, 1.5, 1.5

# -------------------------------
# Fitness Function (Makespan)
# -------------------------------
def fitness(position):
    loads = np.zeros(M)
    for i in range(N):
        machine = int(round(position[i])) % M
        loads[machine] += task_time[i]
    return max(loads)

# -------------------------------
# PSO Initialization
# -------------------------------
particles = np.random.randint(0, M, (num_particles, N))
velocities = np.zeros((num_particles, N))

pbest = particles.copy()
pbest_fitness = np.array([fitness(p) for p in particles])

gbest = pbest[np.argmin(pbest_fitness)]
gbest_fitness = min(pbest_fitness)

# -------------------------------
# PSO Main Loop
# -------------------------------
for _ in range(iterations):
    for i in range(num_particles):
        r1, r2 = random.random(), random.random()
        velocities[i] = (w * velocities[i] +
                          c1 * r1 * (pbest[i] - particles[i]) +
                          c2 * r2 * (gbest - particles[i]))
        particles[i] = particles[i] + velocities[i]

        fit = fitness(particles[i])
        if fit < pbest_fitness[i]:
            pbest[i] = particles[i]
            pbest_fitness[i] = fit

    gbest = pbest[np.argmin(pbest_fitness)]
    gbest_fitness = min(pbest_fitness)

# -------------------------------
# Result
# -------------------------------
print("Best Task Allocation (task → machine):", np.round(gbest).astype(int))
print("Minimum Makespan:", gbest_fitness)
