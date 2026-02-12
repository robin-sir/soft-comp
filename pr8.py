import random
import numpy as np

# Distance matrix
distance = np.array([
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
])

NUM_CITIES = 4
POP_SIZE = 10
GENERATIONS = 100
MUTATION_RATE = 0.2

# Calculate total distance
def total_distance(tour):
    return sum(distance[tour[i], tour[(i + 1) % NUM_CITIES]]
               for i in range(NUM_CITIES))

# Fitness function
def fitness(tour):
    return 1 / total_distance(tour)

# Initial population
def create_population():
    return [random.sample(range(NUM_CITIES), NUM_CITIES)
            for _ in range(POP_SIZE)]

# Tournament selection
def selection(population):
    a, b = random.sample(population, 2)
    return a if fitness(a) > fitness(b) else b

# Ordered crossover (OX)
def crossover(p1, p2):
    start, end = sorted(random.sample(range(NUM_CITIES), 2))
    child = [-1] * NUM_CITIES
    child[start:end] = p1[start:end]
    pointer = end

    for city in p2:
        if city not in child:
            if pointer >= NUM_CITIES:
                pointer = 0
            child[pointer] = city
            pointer += 1
    return child

# Swap mutation
def mutation(tour):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(NUM_CITIES), 2)
        tour[i], tour[j] = tour[j], tour[i]
    return tour

# Genetic Algorithm
population = create_population()

for _ in range(GENERATIONS):
    new_population = []
    for _ in range(POP_SIZE):
        parent1 = selection(population)
        parent2 = selection(population)
        child = crossover(parent1, parent2)
        child = mutation(child)
        new_population.append(child)
    population = new_population

# Best solution
best_tour = min(population, key=total_distance)
print("Best tour:", best_tour)
print("Minimum distance:", total_distance(best_tour))
