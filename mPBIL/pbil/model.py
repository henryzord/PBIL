import copy

import gc
import inspect

from deap import tools
from deap.tools import HallOfFame

import mPBIL
from mPBIL.pbil.generation import *
from mPBIL.pbil.integration import *
from weka.core.dataset import Instances
from mPBIL.pbil.evaluations import EDAEvaluator
from mPBIL.pbil.individuals import Skeleton, Individual
from mPBIL.pbil.ptypes import process_update
from mPBIL.pbil.registry import PBILLogger
from mPBIL.utils import *
from datetime import datetime as dt
import json
import os


class EarlyStop(object):
    def __init__(self, timeout):
        self.n_early_stop = 10
        self.tolerance = 0.005
        self.last_bests = np.linspace(0, 1, num=self.n_early_stop)
        self.start = dt.now()
        self.timeout = timeout

    def is_overtime(self):
        """
        if a timeout parameter is set, and the evolutionary process is taking longer than this parameter
        """
        if (self.timeout > 0) and ((dt.now() - self.start).total_seconds() > self.timeout):
            return True
        return False

    def is_stopping(self):
        if self.is_overtime():
            return True

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
                 log_path=None, timeout=3600, timeout_individual=60, n_folds=5, fitness_metric='unweighted_auc'
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
        :param timeout: Optional: maximum allowed time for algorithm to run. Defaults to 3600 seconds (one hour)
        :type timeout: int
        :param timeout_individual: Optional: maximum allowed time for an individual to be trained. Defaults to 60 seconds
        :type timeout: int
        :param n_folds: Optional: number of folds to use when evaluating individual fitness. Use 0 for 80-20 holdout,
        1 for leave-one-out, or >=2 for internal cross validation. Defaults to 5 folds
        :type n_folds: int
        :param fitness_metric: Optional: fitness metric to be used. Supports 'unweighted_auc' and 'balanced_accuracy'.
        Defaults to 'unweighted_auc'
        :type fitness_metric: str
        """

        self.lr = lr  # type: float
        self.selection_share = selection_share  # type: float
        self.n_generations = n_generations  # type: int
        self.n_individuals = n_individuals  # type: int
        self.timeout = timeout  # type: int
        self.timeout_individual = timeout_individual  # type: int

        clfs = [x[0] for x in inspect.getmembers(mPBIL.pbil.generation, inspect.isclass)]
        classifier_names = [x for x in clfs if ClassifierWrapper in eval('mPBIL.pbil.generation.%s' % x).__bases__]

        self.classifier_names = classifier_names  # type: list
        self.variables = json.load(open(os.path.join(resources_path, 'variables.json'), 'r'))  # type: dict
        self.classifier_data = json.load(open(os.path.join(resources_path, 'classifiers.json'), 'r'))  # type: dict
        self.train_data = train_data  # type: Instances
        self.n_classes = len(self.train_data.class_attribute.values)

        self.evaluator = EDAEvaluator(n_folds=n_folds, train_data=self.train_data, fitness_metric=fitness_metric)

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

    def sample_and_evaluate(self, seed, n_individuals, early_stop):
        """
        Samples new individuals from graphical model.

        :param seed: seed used to partition training set at every generation. The (sub)sets will be constant throughout
        all the evolutionary process, allowing a direct comparison between individuals from different generations.
        :type seed: int
        :param n_individuals: Number of individuals to sample.
        :type n_individuals: int
        :param early_stop: A EarlyStop instance, to check for time
        :type early_stop: EarlyStop
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

        counter_individuals = 0
        while counter_individuals < n_individuals:
            if early_stop.is_overtime():
                break

            discard = False

            start = dt.now()

            ilog = dict()

            for classifier_name in self.classifier_names:
                # if a timeout_individual parameter is set and
                # the induction time of this individual takes longer than it
                if (self.timeout_individual > 0) and ((dt.now() - start).total_seconds() > self.timeout_individual):
                    discard = True
                    break

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

            if discard:
                continue

            ilog['Aggregator'] = np.random.choice(
                a=self.variables['Aggregator']['params']['a'], p=self.variables['Aggregator']['params']['p']
            )
            agg_options, alog = eval(ilog['Aggregator']).sample_options(variables=self.variables)
            ilog.update(alog)

            parameters['Aggregator'] += [[ilog['Aggregator']] + agg_options]

            ilogs += [ilog]
            counter_individuals += 1

        train_fitness = self.evaluator.get_fitness_scores(
            seed=seed, parameters=parameters,
            timeout=early_stop.timeout, start_time=early_stop.start.strftime('%Y-%m-%d-%H-%M-%S')
        )

        # hall of fame is put in the front
        for i in range(0, len_hall):
            local_options = {k: parameters[k][i] for k in self.classifier_names}
            local_options['Aggregator'] = parameters['Aggregator'][i]
            self._hof.insert(Skeleton(
                seed=seed,
                log=ilogs[i],
                options=local_options,
                fitness=train_fitness[i]
            ))
        population = []
        for i in range(len_hall, n_individuals + len_hall):
            local_options = {k: parameters[k][i] for k in self.classifier_names}
            local_options['Aggregator'] = parameters['Aggregator'][i]
            population += [Skeleton(
                seed=seed,
                log=ilogs[i],
                options=local_options,
                fitness=train_fitness[i]
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

        t1 = dt.now()

        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals', 'elapsed_time'] + (stats.fields if stats else [])

        early = EarlyStop(self.timeout)

        population = []
        for gen in range(ngen):
            # early stop
            if early.is_stopping():
                break

            # Generate a new population, already evaluated; re-evaluates halloffame with new seed
            population = self.sample_and_evaluate(
                seed=seed, n_individuals=self.n_individuals, early_stop=early
            )

            self._hof.update(population)

            # Update the strategy with the evaluated individuals
            self.update(population=population)

            record = stats.compile(population) if stats is not None else {}
            t2 = dt.now()
            logbook.record(gen=gen, nevals=len(population), elapsed_time=round((t2 - t1).total_seconds()), **record)
            t1 = dt.now()

            if verbose:
                print(logbook.stream)

            early.update(halloffame=self._hof, gen=gen)

        fitnesses = [ind.fitness for ind in population]

        best_skeleton = population[int(np.argmax(fitnesses))]  # type: Skeleton
        best_last = Individual(
            seed=seed, log=best_skeleton.log,
            options=best_skeleton.options, train_data=self.train_data, skip_evaluation=self.evaluator.n_folds == 0
        )

        skts = [self._hof[i] for i in range(len(self._hof))]
        self._hof.clear()
        for i in range(len(skts)):
            ind = Individual(
                seed=seed, log=skts[i].log,
                options=skts[i].options, train_data=self.train_data, skip_evaluation=self.evaluator.n_folds == 0
            )
            self._hof.insert(ind)

        return best_last, logbook
