import copy

import gc
import inspect

from deap import tools
from deap.tools import HallOfFame

# noinspection PyUnresolvedReferences
from pbil import generation
from pbil.generation import *
# noinspection PyUnresolvedReferences
from pbil.integration import *
from pbil.evaluations import EDAEvaluator
from pbil.individuals import Skeleton, Individual
from pbil.ptypes import process_update
from pbil.registry import PBILLogger
from utils import *


class EarlyStop(object):
    def __init__(self):
        self.n_early_stop = 10
        self.tolerance = 0.005
        self.last_bests = np.linspace(0, 1, num=self.n_early_stop)

    def is_stopping(self):
        return abs(self.last_bests.max() - self.last_bests.min()) < self.tolerance

    def update(self, halloffame, gen):
        self.last_bests[gen % self.n_early_stop] = halloffame[0].fitness


class PBIL(object):
    train_scalars = {
        'fitness_mean': lambda population, last, overall: np.mean([x.fitness for x in population]),
        'fitness_last': lambda population, last, overall: last.fitness,
        'fitness_overall': lambda population, last, overall: overall.fitness
    }

    def __init__(self,
                 resources_path, train_data, lr=0.7, selection_share=0.5, n_generations=200, n_individuals=75,
                 log_path=None
                 ):
        """
        Initializes a new instance of PBIL ensemble learning classifier.
        All PBIL hyper-parameters default to the values presented in the paper

        Cagnini, Henry E.L., Freitas, Alex A., Barros, Rodrigo C.
        An Evolutionary Algorithm for Learning Interpretable Ensembles of Classifiers.
        Brazilian Conference on Intelligent Systems. 2020.

        :param resources_path: Path to folder where at least two files must exist: classifiers.json and variables.json
        :type resources_path: str
        :param train_data: Training data as an object from the python weka wrapper library
        :type train_data: weka.core.dataset.Instances
        :param lr: Learning rate
        :type lr: float
        :param selection_share: How many individuals from general population with best fitness will update
            graphical models' probabilities.
        :type selection_share: float
        :param n_generations: Number of generations to run PBIL
        :type n_generations: int
        :param n_individuals: Number of individuals (solutions) to use
        :type n_individuals: int
        :param log_path: Optional: path to where metadata from this run should be stored.
        :type log_path: str
        """

        self.lr = lr  # type: float
        self.selection_share = selection_share  # type: float
        self.n_generations = n_generations  # type: int
        self.n_individuals = n_individuals  # type: int

        clfs = [x[0] for x in inspect.getmembers(generation, inspect.isclass)]
        classifier_names = [x for x in clfs if ClassifierWrapper in eval('generation.%s' % x).__bases__]

        self.classifier_names = classifier_names  # type: list
        self.variables = json.load(open(os.path.join(resources_path, 'variables.json'), 'r'))  # type: dict
        self.classifier_data = json.load(open(os.path.join(resources_path, 'classifiers.json'), 'r'))  # type: dict
        self.train_data = train_data  # type: Instances
        self.n_classes = len(self.train_data.class_attribute.values)

        self.evaluator = EDAEvaluator(n_folds=5, train_data=self.train_data)

        self.n_generation = 0

        self._hof = None

        scalars = copy.deepcopy(self.train_scalars)

        if log_path is not None:
            self.logger = PBILLogger(logdir_path=log_path, histogram_names=['fitness'],
                                     scalars=scalars, text_names=['last', 'overall']
                                     )
            self.logger.log_probabilities(variables=self.variables)  # register first probabilities
        else:
            self.logger = None

    def sample_and_evaluate(self, seed, n_individuals):
        """
        Samples new individuals from graphical model.

        :param seed: seed used to partition training set at every generation. The (sub)sets will be constant throughout
        all the evolutionary process, allowing a direct comparison between individuals from different generations.
        :type seed: int
        :param n_individuals: Number of individuals to sample.
        :type n_individuals: int
        :return: the recently sampled population
        :rtype: list
        """

        len_hall = len(self._hof)

        if len_hall == 0:
            parameters = {k: [] for k in self.classifier_names}
            parameters['Aggregator'] = []
            ilogs = []
        else:
            parameters = {k: [x.options[k] for x in self._hof] for k in self.classifier_names}
            parameters['Aggregator'] = [x.options['Aggregator'] for x in self._hof]
            ilogs = [x.log for x in self._hof]
            self._hof.clear()

        for j in range(n_individuals):
            ilog = dict()

            for classifier_name in self.classifier_names:
                ilog[classifier_name] = np.random.choice(
                    a=self.variables[classifier_name]['params']['a'],
                    p=self.variables[classifier_name]['params']['p']
                )
                if ilog[classifier_name]:  # whether to include this classifier in the ensemble
                    options, cclog = eval(classifier_name).sample_options(
                        variables=self.variables, classifiers=self.classifier_data
                    )

                    ilog.update(cclog)
                    parameters[classifier_name] += [options]
                else:
                    parameters[classifier_name].append([])

            ilog['Aggregator'] = np.random.choice(
                a=self.variables['Aggregator']['params']['a'], p=self.variables['Aggregator']['params']['p']
            )
            agg_options, alog = eval(ilog['Aggregator']).sample_options(variables=self.variables)
            ilog.update(alog)

            parameters['Aggregator'] += [[ilog['Aggregator']] + agg_options]

            ilogs += [ilog]

        train_aucs = self.evaluator.get_unweighted_aucs(seed=seed, parameters=parameters)

        # hall of fame is put in the front
        for i in range(0, len_hall):
            local_options = {k: parameters[k][i] for k in self.classifier_names}
            local_options['Aggregator'] = parameters['Aggregator'][i]
            self._hof.insert(Skeleton(
                seed=seed,
                log=ilogs[i],
                options=local_options,
                fitness=train_aucs[i]
            ))
        population = []
        for i in range(len_hall, n_individuals + len_hall):
            local_options = {k: parameters[k][i] for k in self.classifier_names}
            local_options['Aggregator'] = parameters['Aggregator'][i]
            population += [Skeleton(
                seed=seed,
                log=ilogs[i],
                options=local_options,
                fitness=train_aucs[i]
            )]

        return population

    def update(self, population):
        """
        Updates graphical model probabilities based on the fittest population.

        :param population: All population from a given generation.
        :type population: list
        """

        if self.logger is not None:
            self.logger.log_probabilities(variables=self.variables)
            self.logger.log_population(population=population, halloffame=self._hof)

        # selects fittest individuals
        _sorted = sorted(zip(population, [ind.fitness for ind in population]), key=lambda x: x[1], reverse=True)
        population, fitnesses = zip(*_sorted)
        fittest = population[:int(len(population) * self.selection_share)]
        observations = pd.DataFrame([fit.log for fit in fittest])

        # update classifiers probabilities
        for variable_name, variable_data in self.variables.items():
            self.variables[variable_name] = process_update(
                ptype=variable_data['ptype'], variable_name=variable_name, variable_data=variable_data,
                observations=observations, lr=self.lr, n_generations=self.n_generations
            )

        self.n_generation += 1

    def run(self, seed):
        """
        Trains this classifier.

        :param seed: seed used to partition training set at every generation. The (sub)sets will be constant throughout
        all the evolutionary process, allowing a direct comparison between individuals from different generations.
        :type seed: int
        :rtype: tuple
        :return: a tuple containing two individuals.Individual objects, where the first individual is the best solution
        (according to fitness) found throughout all the evolutionary process, and the second individual the best solution
        from the last generation.
        """

        # Statistics computation
        stats = tools.Statistics(lambda ind: ind.fitness)
        for stat_name, stat_func in PBILLogger.population_operators:
            stats.register(stat_name, stat_func)

        # An object that keeps track of the best individual found so far.
        self._hof = HallOfFame(maxsize=1)  # type: HallOfFame

        best_last, logbook = self.__run__(seed=seed, ngen=self.n_generations, stats=stats, verbose=True)

        best_overall = self._hof[0]  # type: Individual
        self._hof = None

        gc.collect()

        return best_overall, best_last

    def __run__(self, seed, ngen, stats=None, verbose=__debug__):
        """
        Do not use this method.
        """

        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

        early = EarlyStop()

        population = []
        for gen in range(ngen):
            # early stop
            if early.is_stopping():
                break

            # Generate a new population, already evaluated; re-evaluates halloffame with new seed
            population = self.sample_and_evaluate(
                seed=seed, n_individuals=self.n_individuals
            )

            self._hof.update(population)

            # Update the strategy with the evaluated individuals
            self.update(population=population)

            record = stats.compile(population) if stats is not None else {}
            logbook.record(gen=gen, nevals=len(population), **record)
            if verbose:
                print(logbook.stream)

            early.update(halloffame=self._hof, gen=gen)

        fitnesses = [ind.fitness for ind in population]

        best_skeleton = population[int(np.argmax(fitnesses))]  # type: Skeleton
        best_last = Individual(
            seed=seed, log=best_skeleton.log,
            options=best_skeleton.options, train_data=self.train_data
        )

        skts = [self._hof[i] for i in range(len(self._hof))]
        self._hof.clear()
        for i in range(len(skts)):
            ind = Individual(
                seed=seed, log=skts[i].log,
                options=skts[i].options, train_data=self.train_data
            )
            self._hof.insert(ind)

        return best_last, logbook
