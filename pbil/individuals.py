import copy

from pbil.evaluations import EDAEvaluation, internal_5fcv
from pbil.generation import *
from pbil.integration import baseline_aggregator_options


class Skeleton(object):
    def __init__(self, seed, log, options, fitness):
        """

        :param log: Hyper-parameters used for generating this individual
        :type log: dict
        """

        self.seed = seed
        self.log = log
        self.options = options
        self.fitness = fitness

    def __deepcopy__(self, memodict={}):
        ind = Skeleton(
            seed=copy.deepcopy(self.seed),
            log=copy.deepcopy(self.log),
            options=copy.deepcopy(self.options),
            fitness=copy.deepcopy(self.fitness)
        )
        return ind

    @staticmethod
    def from_sets(seed, jobject_ensemble, train_data):

        train_evaluation_obj = internal_5fcv(seed=seed, jobject=jobject_ensemble, train_data=train_data)
        train_pevaluation = EDAEvaluation.from_jobject(train_evaluation_obj, data=train_data, seed=seed)

        return train_pevaluation

    def __str__(self):
        return str(self.fitness)


class Individual(Skeleton):
    def __init__(self, seed, log, options, train_data):
        """

        :param log:
        :param options:
        :param train_data:
        """

        _jobject_ensemble = Individual.__set_jobject_ensemble__(options=options, train_data=train_data)

        train_evaluation = Skeleton.from_sets(
            jobject_ensemble=_jobject_ensemble, train_data=train_data, seed=seed
        )

        super(Individual, self).__init__(
            seed=seed, log=log, options=options,
            fitness=train_evaluation.unweighted_area_under_roc
        )

        self.train_evaluation = train_evaluation
        self._jobject_ensemble = _jobject_ensemble
        self._train_data = train_data

        self.classifiers = self.__initialize_classifiers__()

    def __deepcopy__(self, memodict={}):
        ind = Individual(
            seed=copy.deepcopy(self.seed),
            log=copy.deepcopy(self.log),
            options=copy.deepcopy(self.options),
            train_data=self._train_data
        )
        return ind

    def __initialize_classifiers__(self):
        env = javabridge.get_env()  # type: javabridge.JB_Env
        clf_objs_names = env.get_object_array_elements(javabridge.call(self._jobject_ensemble, 'getClassifiersNames', '()[[Ljava/lang/String;'))

        clfs = []
        for t in clf_objs_names:
            obj_name, obj_class, obj_sig = list(map(javabridge.to_string, env.get_object_array_elements(t)))
            if len(self.options[obj_class]) > 0:
                obj = javabridge.get_field(self._jobject_ensemble, obj_name, obj_sig)

                clf = eval(obj_class).from_jobject(obj)
                clfs += [clf]

        return clfs

    @staticmethod
    def __set_jobject_ensemble__(options, train_data):
        opts = []
        for flag, listoptions in options.items():
            opts.extend(['-' + flag, ' '.join(listoptions)])

        eda_ensemble = javabridge.make_instance(
            'Leda/EDAEnsemble;', '([Ljava/lang/String;Lweka/core/Instances;)V', opts, train_data.jobject
        )
        return eda_ensemble

    def predict(self, data):
        """
        Predicts on new data.

        :param data: Data to make predictions.
        :type data: weka.core.datasets.Instances
        :return: a numpy.ndarray array where each entry is the prediction for that instance. Returned values are the
        same as the ones present in training set (i.e. if class has strings instead of integers, then strings will be
        returned, instead of the index of the class value).
        :rtype: numpy.ndarray
        """

        predictions = self.predict_proba(data)
        predicted = predictions.argmax(axis=1)
        return np.array(self._train_data.class_attribute.values)[predicted]

    def predict_proba(self, data):
        """
        Makes probabilistic predictions.

        :param data: Data to make predictions.
        :type data: weka.core.datasets.Instances
        :return: a numpy.ndarray matrix where each row is a instance and each column a class.
        :rtype: numpy.ndarray
        """
        env = javabridge.get_env()  # type: javabridge.JB_Env

        if len(data) == 1:
            dist = javabridge.call(
                self._jobject_ensemble, 'distributionForInstance', '(Lweka/core/Instance;)[D', data.jobject)
            final_dist = env.get_double_array_elements(dist)
        else:
            dist = javabridge.call(
                self._jobject_ensemble, 'distributionsForInstances', '(Lweka/core/Instances;)[[D', data.jobject
            )
            final_dist = np.array([env.get_double_array_elements(x) for x in env.get_object_array_elements(dist)])

        return final_dist

    def __str__(self):
        clf_texts = ''
        for clf in self.classifiers:
            clf_texts += '%s\n\n' % str(clf)
        return clf_texts

    @classmethod
    def from_baseline(cls, seed, classifiers, train_data):
        options, ilog = baseline_classifiers_options(classifiers)
        aggoptions, agglog = baseline_aggregator_options(None)
        options.update(aggoptions)
        ilog.update(agglog)

        ind = Individual(seed=seed, log=ilog, options=options, train_data=train_data)
        return ind
