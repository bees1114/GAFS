from functools import partial

from deap import base, tools, creator, gp
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import random
import numpy as np


class GeneticFeatureSelection:
    def _init_deap(self):
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        self._toolbox = base.Toolbox()
        self._toolbox.register("individual", tools.initIterate, creator.Individual, self._get_individual)
        self._toolbox.register("population", tools.initRepeat, list, self._toolbox.individual)

        self._toolbox.register("crossover", tools.cxOnePoint)
        # self._toolbox.register("mutate", tools.mutation)
        self._toolbox.register("select", tools.selNSGA2)

    def __init__(self, generations, population_size, crossover_rate=0.5, mutation_rate=0.1):
        self.column_name_list = None
        self.generations = generations
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def _evaluate(self, individual, x_train, x_test, y_train, y_test):
        clf = SVC(gamma='auto')
        print('--evaluate--')
        print(individual)
        print(x_train.columns)
        print(x_test.columns)
        x_train = x_train[individual]
        x_test = x_test[individual]
        clf.fit(x_train, y_train)
        y_hat = clf.predict(x_test)
        return np.mean(y_hat == y_test)

    def _evaluate_individuals(self, population, X, y):
        individuals = [individual for individual in population if not individual.fitness.valid]
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
        fitnesses = map(partial(self._evaluate, x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test),
                        individuals)
        for individual, fit in zip(individuals, fitnesses):
            individual.fitness.values = fit

    def _get_individual(self):
        column_length = len(self.column_name_list)
        individual_length = column_length // 5
        print('len = ' + str(individual_length))
        index = random.sample(range(column_length), individual_length)
        print(self.column_name_list[index])

        return self.column_name_list[index]

    def _get_columns(self, data_frame):
        return data_frame.columns

    def fit(self, X, y):
        self.column_name_list = self._get_columns(X)
        self._init_deap()
        self._pop = self._toolbox.population(n=self.population_size)
        self._toolbox.register("evaluate", self._evaluate_individuals, self._pop, X, y)
        self._toolbox.evaluate()



    def get_best_features(self):
        pass