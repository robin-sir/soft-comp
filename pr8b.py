import numpy as np
import random
import matplotlib.pyplot as plt

# Distance matrix (5 cities)
dist = np.array([
    [0, 2, 9, 10, 7],
    [2, 0, 6, 4, 3],
    [9, 6, 0, 8, 5],
    [10, 4, 8, 0, 6],
    [7, 3, 5, 6, 0]
])

NUM_CITIES = len(dist)
POP_SIZE = 50
GENERATIONS = 200
MUTATION_RATE = 0.2

# Fitness function
def total_distance(route):
    return sum(dist[route[i], route[i+1]] for i in range(NUM_CITIES-1)) + dist[route[-1], route[0]]

def fitness(route):
    return 1 / total_distance(route)

# Initial population
def init_population():
    return [random.sample(range(NUM_CITIES), NUM_CITIES) for _ in range(POP_SIZE)]

# Tournament selection
def selection(pop):
    k = 3
    selected = random.sample(pop, k)
    return max(selected, key=fitness)

# Order Crossover (OX)
def crossover(p1, p2):
    a, b = sorted(random.sample(range(NUM_CITIES), 2))
    child = [-1]*NUM_CITIES
    child[a:b] = p1[a:b]

    ptr = b
    for city in p2:
        if city not in child:
            if ptr == NUM_CITIES:
                ptr = 0
            child[ptr] = city
            ptr += 1
    return child

# Swap Mutation
def mutate(route):
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(NUM_CITIES), 2)
        route[i], route[j] = route[j], route[i]
    return route

# GA Main Loop
population = init_population()
best_dist = []

for _ in range(GENERATIONS):
    new_pop = []
    for _ in range(POP_SIZE):
        p1 = selection(population)
        p2 = selection(population)
        child = crossover(p1, p2)
        child = mutate(child)
        new_pop.append(child)
    population = new_pop
    best_route = min(population, key=total_distance)
    best_dist.append(total_distance(best_route))

print("Best Route:", best_route)
print("Minimum Distance:", total_distance(best_route))

plt.plot(best_dist)
plt.xlabel("Generations")
plt.ylabel("Distance")
plt.title("GA for TSP")
plt.show()
