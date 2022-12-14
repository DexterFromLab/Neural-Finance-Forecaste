from random import randint, random

from deap import base
from deap import creator
from deap import tools
from itertools import starmap

# Utworzenie klas indywidualnego i populacji
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Utworzenie narzędzi selekcji
toolbox = base.Toolbox()
toolbox.register("attr_bool", randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=100)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# Utworzenie funkcji celu i metody selekcji
def evalOneMax(individual):
    return sum(individual),


def mutFlipBitCustom(individual, indpb):
    for i in range(len(individual)):
        if random() < indpb:
            individual[i] = type(individual[i])(not individual[i])
            return individual

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutFlipBitCustom, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Utworzenie początkowej populacji i ewaluacja
population = toolbox.population(n=300)
fitnesses = list(map(toolbox.evaluate, population))
for ind, fit in zip(population, fitnesses):
    ind.fitness.values = fit

# Uruchomienie algorytmu genetycznego
NGEN = 40
for gen in range(NGEN):
    # Selekcja osobników do krzyżowania
    offspring = toolbox.select(population, len(population))
    # Krzyżowanie osobników
    pairs = list(zip(offspring[::2], offspring[1::2]))
    offspring = list(map(toolbox.mate, pairs[0], pairs[1]))
    # Mutacja osobników
    offspring = starmap(toolbox.mutate, zip(offspring, [(0.05, 0.05)] * len(offspring)))
    # Ewaluacja nowej populacji
    for ind in offspring:
        fitness = toolbox.evaluate(ind)
        ind.fitness.values = fitness

for ind, fit in zip(offspring, fitnesses):
    ind.fitness.values = fit

# Zastąpienie starej populacji nową
population = offspring

# Pobranie najlepszego osobnika
best_ind = tools.selBest(population, 1)[0]
print("Najlepszy osobnik: %s, ocena: %s" % (best_ind, best_ind.fitness.values))


def main():
    return 0


if __name__ == "__main__":
    main()
