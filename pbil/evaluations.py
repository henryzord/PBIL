import itertools as it
import os
from functools import reduce

import javabridge
import numpy as np
import pandas as pd
from weka.classifiers import Evaluation

from pbil.registry import generate_probability_averager_script
from utils import from_python_stringlist_to_java_stringlist


class EDAEvaluator(object):
    def __init__(self, n_folds, train_data):
        self.n_classes = len(train_data.attribute(train_data.class_index).values)
        self.train_data = train_data
        self.n_folds = n_folds

        self.evaluator = javabridge.make_instance(
            "Leda/EDAEvaluator;", "(ILweka/core/Instances;)V",
            self.n_folds,
            self.train_data.jobject
        )

    def get_unweighted_aucs(self, seed, parameters):
        res = javabridge.call(
            self.evaluator, "evaluateEnsembles", "(I" + 7 * "[[Ljava/lang/String;" + ")[D", seed,
            from_python_stringlist_to_java_stringlist(parameters['J48']),
            from_python_stringlist_to_java_stringlist(parameters['SimpleCart']),
            from_python_stringlist_to_java_stringlist(parameters['REPTree']),
            from_python_stringlist_to_java_stringlist(parameters['PART']),
            from_python_stringlist_to_java_stringlist(parameters['JRip']),
            from_python_stringlist_to_java_stringlist(parameters['DecisionTable']),
            from_python_stringlist_to_java_stringlist(parameters['Aggregator'])
        )

        env = javabridge.get_env()  # type: javabridge.JB_Env
        train_aucs = env.get_double_array_elements(res)

        return train_aucs


class EDAEvaluation(Evaluation):
    metrics = [
        ("avg_cost", np.mean),
        ("class_priors", np.sum),
        ("confusion_matrix", np.sum),
        ("correct", np.sum),
        ("error_rate", np.mean),
        ("incorrect", np.sum),
        ("kappa", np.mean),
        ("kb_information", np.mean),
        ("kb_mean_information", np.mean),
        ("kb_relative_information", np.mean),
        ("mean_absolute_error", np.mean),
        ("mean_prior_absolute_error", np.mean),
        # ("n_classes", np.mean),
        ("num_instances", np.sum),
        ("percent_correct", np.mean),
        ("percent_incorrect", np.mean),
        ("percent_unclassified", np.mean),
        ("relative_absolute_error", np.mean),
        ("root_mean_prior_squared_error", np.mean),
        ("root_mean_squared_error", np.mean),
        ("root_relative_squared_error", np.mean),
        ("sf_entropy_gain", np.mean),
        ("sf_mean_entropy_gain", np.mean),
        ("sf_mean_prior_entropy", np.mean),
        ("sf_mean_scheme_entropy", np.mean),
        ("sf_prior_entropy", np.mean),
        ("size_of_predicted_regions", np.mean),
        ("total_cost", np.sum),
        ("unclassified", np.sum),
        ("unweighted_area_under_roc", np.mean),
        ("unweighted_macro_f_measure", np.mean),
        ("unweighted_micro_f_measure", np.mean),
        ("weighted_area_under_prc", np.mean),
        ("weighted_area_under_roc", np.mean),
        ("weighted_f_measure", np.mean),
        ("weighted_false_negative_rate", np.mean),
        ("weighted_false_positive_rate", np.mean),
        ("weighted_matthews_correlation", np.mean),
        ("weighted_precision", np.mean),
        ("weighted_recall", np.mean),
        ("weighted_true_negative_rate", np.mean),
        ("weighted_true_positive_rate", np.mean),
    ]

    # converts names of metrics from Java Weka to Python Wrapper Weka
    metrics_dict = {
        "avg_cost": "avgCost",
        "class_priors": "getClassPriors",
        "confusion_matrix": "confusionMatrix",
        "correct": "correct",
        "error_rate": "errorRate",
        "incorrect": "incorrect",
        "kappa": "kappa",
        "kb_information": "KBInformation",
        "kb_mean_information": "KBMeanInformation",
        "kb_relative_information": "KBRelativeInformation",
        "mean_absolute_error": "meanAbsoluteError",
        "mean_prior_absolute_error": "meanPriorAbsoluteError",
        # "n_classes": "n_classes",
        "num_instances": "numInstances",
        "percent_correct": "pctCorrect",
        "percent_incorrect": "pctIncorrect",
        "percent_unclassified": "pctUnclassified",
        "unclassified": "unclassified",
        "relative_absolute_error": "relativeAbsoluteError",
        "root_mean_prior_squared_error": "rootMeanPriorSquaredError",
        "root_mean_squared_error": "rootMeanSquaredError",
        "root_relative_squared_error": "rootRelativeSquaredError",
        "sf_entropy_gain": "SFEntropyGain",
        "sf_mean_entropy_gain": "SFMeanEntropyGain",
        "sf_mean_prior_entropy": "SFMeanPriorEntropy",
        "sf_mean_scheme_entropy": "SFMeanSchemeEntropy",
        "sf_prior_entropy": "SFPriorEntropy",
        "size_of_predicted_regions": "sizeOfPredictedRegions",
        "total_cost": "totalCost",
        "weighted_area_under_prc": "weightedAreaUnderPRC",
        "weighted_area_under_roc": "weightedAreaUnderROC",
        "unweighted_area_under_roc": "unweightedAreaUnderRoc",
        "weighted_f_measure": "weightedFMeasure",
        "weighted_false_negative_rate": "weightedFalseNegativeRate",
        "weighted_false_positive_rate": "weightedFalsePositiveRate",
        "weighted_matthews_correlation": "weightedMatthewsCorrelation",
        "weighted_precision": "weightedPrecision",
        "weighted_recall": "weightedRecall",
        "weighted_true_negative_rate": "weightedTrueNegativeRate",
        "weighted_true_positive_rate": "weightedTruePositiveRate",
        "unweighted_macro_f_measure": "unweightedMacroFmeasure",
        "unweighted_micro_f_measure": "unweightedMicroFmeasure"
    }

    def __init__(self, data, seed):
        super().__init__(data)
        self.n_classes = len(data.attribute(data.class_index).values)
        self.seed = seed

    @property
    def unweighted_area_under_roc(self):
        auc_areas = 0.
        for j in range(self.n_classes):
            auc_areas += self.area_under_roc(j)
        return auc_areas / self.n_classes

    @classmethod
    def from_jobject(cls, jobject, data, seed):
        """

        :param jobject: Java object encapsulating Evaluation instance.
        :type jobject: javabridge.JB_Object
        :param data: Training instances.
        :type data: weka.core.Instances
        :param seed:
        :return:
        """
        inst = cls(data, seed=seed)
        inst.jobject = jobject
        return inst


def __get_relation__(path):
    """
    Get experiment results as a dataframe.

    :param path: Path pointing to folder with experiment results as a series of csv files.
    :return: A dataframe with the relation of experiment results as a dataframe.
    """
    files = os.listdir(path)
    df = pd.DataFrame(list(map(lambda x: x[:-len('.txt')].split('_'), files)),
                      columns=['dataset', 'sample', 'fold']).dropna()
    return df


def internal_5fcv(seed, jobject, train_data):
    env = javabridge.get_env()  # type: javabridge.JB_Env

    random_jobject = javabridge.make_instance('Ljava/util/Random;', '(J)V', seed)

    train_evaluation_obj = javabridge.make_instance(
        'Lweka/classifiers/Evaluation;', '(Lweka/core/Instances;)V', train_data.jobject
    )

    javabridge.call(
        train_evaluation_obj, 'crossValidateModel',
        '(Lweka/classifiers/Classifier;Lweka/core/Instances;ILjava/util/Random;)V',
        jobject, train_data.jobject, 5, random_jobject
    )

    return train_evaluation_obj


def collapse_metrics(metadata_path, n_samples, n_folds, write=True, only_baselines=False):
    missing = check_missing_experiments(
        path=metadata_path,
        n_samples=n_samples, n_folds=n_folds
    )
    if len(missing) > 0:
        raise ValueError('Could not collapse metrics for dataset. Some runs are missing.')

    samples_dicts = dict()

    relation = __get_relation__(metadata_path)
    samples = relation['sample'].unique()

    for sample in samples:
        this_sample_relation = relation.loc[relation['sample'] == sample]

        rels = list(map(
            lambda z: pd.read_csv('%s/%s.csv' % (metadata_path, '_'.join(z)), index_col=0),
            this_sample_relation.values
        ))
        ens_names = rels[0].index

        condensed = pd.DataFrame(
            index=ens_names,
            columns=pd.MultiIndex.from_product([list(zip(*EDAEvaluation.metrics))[0], ['mean', 'std']],
                                               names=['metric', 'statistics']),
            dtype=np.float64
        )
        for metric_name, metric_operation in EDAEvaluation.metrics:
            if (metric_name == 'confusion_matrix') or (metric_name == 'class_priors'):
                dict_ens = {}
                for ens_name in ens_names:
                    for rel in rels:
                        try:
                            dict_ens[ens_name] += eval(rel.loc[ens_name, metric_name])
                        except KeyError:
                            dict_ens[ens_name] = eval(rel.loc[ens_name, metric_name])

                condensed[(metric_name, 'mean')] = pd.Series(dict_ens)
                condensed[(metric_name, 'std')] = np.repeat(np.nan, len(dict_ens))

            else:
                for ens_name in ens_names:
                    values = [rel.loc[ens_name, metric_name] for rel in rels]
                    condensed.loc[ens_name, (metric_name, 'mean')] = np.mean(values)
                    condensed.loc[ens_name, (metric_name, 'std')] = np.std(values)

        condensed['sample'] = np.repeat(sample, len(condensed))
        samples_dicts[sample] = condensed

    # adds mean of means
    summary = reduce(lambda x, y: x.append(y), samples_dicts.values())

    pre_agg = summary.drop('std', axis=1, level=1)
    pre_agg.columns = pre_agg.columns.droplevel(1)

    agg = pre_agg.groupby(level=0).agg([np.mean, np.std])

    summary.index = ['-'.join([x, y]) for x, y in zip(summary.index, summary['sample'])]
    del summary['sample']
    agg.index = ['-'.join([x, y]) for x, y in zip(agg.index, np.repeat('mean-of-means', len(agg)))]
    summary = summary.append(agg)

    if write:
        summary.to_csv(os.path.join(metadata_path, 'summary.csv'))

    # tides up for display
    to_return_index = list(map(lambda x: '-'.join(x), it.product(ens_names, ['mean-of-means'])))
    to_return_columns = pd.MultiIndex.from_product([['unweighted_area_under_roc', 'percent_correct'], ['mean', 'std']])
    to_return = summary.loc[to_return_index, to_return_columns]  # type: pd.DataFrame
    to_return.columns = pd.MultiIndex.from_product([['AUC', 'Accuracy'], ['mean', 'std']])
    to_return.index = [x.split('-')[0] for x in to_return.index]
    to_return.loc[:, (slice(None), 'mean')] = to_return.loc[:, (slice(None), 'mean')].applymap('{:,.4f}'.format)
    to_return.loc[:, (slice(None), 'std')] = to_return.loc[:, (slice(None), 'std')].applymap('{:,.2f}'.format)

    if not only_baselines:
        generate_probability_averager_script(dataset_path=os.sep.join(metadata_path.split(os.sep)[:-1]))
    return to_return


def check_missing_experiments(path, n_samples, n_folds):
    """
    Checks missing results files in the "overall" folder.

    Please point to a folder that contains the .csv results of the experiments.

    :param path: Path to folder with csv results.
    :param n_samples: Number of trial sampples in the experiment.
    :param n_folds: Number of folds in the cross-validation.
    :rtype: list
    :return: A list of tuples where the first item in each tuple is the trial, and the second the number of the fold,
        that are missing in the results folder pointed by path.
    """

    assert path is not None, ValueError('path must point to the folder with csv results.')

    folds = range(1, n_folds + 1)
    samples = range(1, n_samples + 1)

    files = os.listdir(path)

    missing = []
    combs = it.product(samples, folds)
    for sample, fold in combs:
        if 'test_sample-%02d_fold-%02d.csv' % (sample, fold) not in files:
            missing += [(sample, fold)]

    return missing


def evaluate_on_test(jobject, test_data):
    """
    Evaluates a classifier (as a Java Object) on a test set (as a Java Instances object).
    """

    test_evaluation_obj = javabridge.make_instance(
        'Lweka/classifiers/Evaluation;', '(Lweka/core/Instances;)V', test_data.jobject
    )
    javabridge.call(
        test_evaluation_obj, "evaluateModel",
        "(Lweka/classifiers/Classifier;Lweka/core/Instances;[Ljava/lang/Object;)[D",
        jobject, test_data.jobject, []
    )
    return test_evaluation_obj
