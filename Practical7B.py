import random

# Objective function
def fitness(x):
    return x * x

# Generate initial population
def initialize_population(size):
    return [random.uniform(-10, 10) for _ in range(size)]

# Selection (Tournament Selection)
def selection(population):
    a, b = random.sample(population, 2)
    return a if fitness(a) < fitness(b) else b

# Crossover
def crossover(parent1, parent2):
    alpha = random.random()
    return alpha * parent1 + (1 - alpha) * parent2

# Mutation
def mutation(child, rate=0.1):
    if random.random() < rate:
        child += random.uniform(-1, 1)
    return child

# Genetic Algorithm
def genetic_algorithm():
    population_size = 20
    generations = 50

    population = initialize_population(population_size)

    for _ in range(generations):
        new_population = []
        for _ in range(population_size):
            p1 = selection(population)
            p2 = selection(population)
            child = crossover(p1, p2)
            child = mutation(child)
            new_population.append(child)
        population = new_population

    best = min(population, key=fitness)
    print("Best solution:", best)
    print("Minimum value:", fitness(best))

genetic_algorithm()
