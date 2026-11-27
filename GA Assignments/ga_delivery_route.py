"""
Genetic Algorithm for Delivery Route Optimisation
Artificial Intelligence - GA Assignment
Student: Muhamad Azfar Ukail Bin Mohd Azuan (24005439)

The van starts at depot A, visits B, C, D, E, F and G exactly once each,
and returns to A. The GA searches for the visiting order with the shortest
total distance.

Chromosome: a permutation of the six delivery locations, e.g.
    ['B', 'E', 'F', 'C', 'D', 'G']  ->  route A-B-E-F-C-D-G-A
Depot A is fixed at the start and end, so it is not part of the chromosome.
This guarantees every route starts and finishes at A.
"""

import random

# ----------------------------------------------------------------------
# 1. Dataset: symmetric distance matrix (kilometres)
# ----------------------------------------------------------------------
LOCATIONS = ["A", "B", "C", "D", "E", "F", "G"]
DISTANCE_MATRIX = [
    #  A  B  C  D  E  F  G
    [0, 4, 8, 6, 7, 9, 5],  # A
    [4, 0, 5, 7, 3, 8, 6],  # B
    [8, 5, 0, 4, 6, 3, 7],  # C
    [6, 7, 4, 0, 5, 6, 8],  # D
    [7, 3, 6, 5, 0, 4, 6],  # E
    [9, 8, 3, 6, 4, 0, 5],  # F
    [5, 6, 7, 8, 6, 5, 0],  # G
]
INDEX = {name: i for i, name in enumerate(LOCATIONS)}

DEPOT = "A"
DELIVERY_STOPS = ["B", "C", "D", "E", "F", "G"]

# ----------------------------------------------------------------------
# 2. GA settings
# ----------------------------------------------------------------------
RANDOM_SEED = 42        # fixed seed -> same result every run
POPULATION_SIZE = 30
GENERATIONS = 100       # assignment requires at least 50
TOURNAMENT_SIZE = 3
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.2
ELITE_COUNT = 2         # best routes copied unchanged each generation


def distance(a, b):
    """Distance in km between two named locations."""
    return DISTANCE_MATRIX[INDEX[a]][INDEX[b]]


def full_route(chromosome):
    """Add the depot at the start and end: A + chromosome + A."""
    return [DEPOT] + list(chromosome) + [DEPOT]


def route_distance(chromosome):
    """Total km of the route, INCLUDING the return journey to depot A."""
    route = full_route(chromosome)
    return sum(distance(route[i], route[i + 1]) for i in range(len(route) - 1))


def fitness(chromosome):
    """Higher fitness = shorter route (we minimise distance)."""
    return 1.0 / route_distance(chromosome)


def is_valid(chromosome):
    """Each delivery stop appears exactly once."""
    return sorted(chromosome) == sorted(DELIVERY_STOPS)


# ----------------------------------------------------------------------
# 3. Initial population: random valid routes (shuffled permutations)
# ----------------------------------------------------------------------
def create_initial_population(size):
    population = []
    for _ in range(size):
        chromosome = DELIVERY_STOPS[:]
        random.shuffle(chromosome)
        population.append(chromosome)
    return population


# ----------------------------------------------------------------------
# 4. Selection: tournament selection
#    Pick a few random routes and keep the one with the best fitness.
# ----------------------------------------------------------------------
def tournament_selection(population):
    contestants = random.sample(population, TOURNAMENT_SIZE)
    return max(contestants, key=fitness)


# ----------------------------------------------------------------------
# 5. Crossover: Order Crossover (OX)
#    Copy a slice from parent 1, then fill the remaining positions with the
#    missing stops in the order they appear in parent 2.
#    -> child is always a valid permutation (no repeats, nothing missing).
# ----------------------------------------------------------------------
def order_crossover(parent1, parent2):
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end + 1] = parent1[start:end + 1]          # keep slice from parent 1

    fill_values = [stop for stop in parent2 if stop not in child]  # parent 2 order
    fill_positions = [i for i in range(size) if child[i] is None]
    for pos, stop in zip(fill_positions, fill_values):
        child[pos] = stop
    return child


# ----------------------------------------------------------------------
# 6. Mutation: swap mutation
#    Swap two delivery stops -> still a valid route, just a different order.
# ----------------------------------------------------------------------
def swap_mutation(chromosome):
    mutated = chromosome[:]
    if random.random() < MUTATION_RATE:
        i, j = random.sample(range(len(mutated)), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]
    return mutated


# ----------------------------------------------------------------------
# 7. Main GA loop
# ----------------------------------------------------------------------
def run_ga(seed=RANDOM_SEED, verbose=True):
    random.seed(seed)
    population = create_initial_population(POPULATION_SIZE)
    best_per_generation = []

    for generation in range(1, GENERATIONS + 1):
        # Sort routes from shortest to longest
        population.sort(key=route_distance)
        best_per_generation.append(route_distance(population[0]))

        if verbose and (generation == 1 or generation % 10 == 0):
            avg = sum(route_distance(c) for c in population) / len(population)
            print(f"Generation {generation:3d} | best = {route_distance(population[0]):2d} km"
                  f" | average = {avg:5.2f} km | best route = {'-'.join(full_route(population[0]))}")

        # Elitism: copy the best routes unchanged into the next generation
        next_population = [c[:] for c in population[:ELITE_COUNT]]

        # Fill the rest with children: selection -> crossover -> mutation
        while len(next_population) < POPULATION_SIZE:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            if random.random() < CROSSOVER_RATE:
                child = order_crossover(parent1, parent2)
            else:
                child = parent1[:]
            child = swap_mutation(child)
            assert is_valid(child)  # safety check: never repeats or misses a stop
            next_population.append(child)

        population = next_population

    best = min(population, key=route_distance)
    return best, route_distance(best), best_per_generation


if __name__ == "__main__":
    print("Genetic Algorithm - Delivery Route Optimisation")
    print(f"Seed = {RANDOM_SEED}, population = {POPULATION_SIZE}, generations = {GENERATIONS}, "
          f"tournament = {TOURNAMENT_SIZE}, crossover = {CROSSOVER_RATE}, "
          f"mutation = {MUTATION_RATE}, elites = {ELITE_COUNT}\n")

    best_route, best_km, history = run_ga()
    first_best_gen = history.index(best_km) + 1

    print("\n========== RESULT ==========")
    print(f"Best route     : {' -> '.join(full_route(best_route))}")
    print(f"Total distance : {best_km} km")
    print(f"First reached  : generation {first_best_gen}")
