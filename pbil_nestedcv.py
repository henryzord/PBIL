from multiprocessing import set_start_method
try:
    set_start_method("spawn")
except RuntimeError:
    pass  # is in child process, trying to set context to spawn but failing because is already set

import multiprocessing as mp
import time

import argparse
import json
import os
import sys

import javabridge
from mPBIL.pbil.model import PBIL
from weka.core import jvm
from weka.core.classes import Random
from weka.core.converters import Loader
from weka.core.dataset import Instances
from sklearn.metrics import roc_auc_score
import numpy as np
from copy import deepcopy
from datetime import datetime as dt


def get_pbil_combinations():
    combinations = []

    learning_rate_values = [0.13, 0.26, 0.52]
    selection_share_values = [0.3, 0.5]
    n_individuals = 50
    n_generations = 100
    timeout = 3600  # one hour
    timeout_individual = 60

    n_folds = 0  # holdout

    for learning_rate in learning_rate_values:
        for selection_share in selection_share_values:
            comb = {
                "learning_rate": learning_rate,
                "selection_share": selection_share,
                "n_individuals": n_individuals,
                "n_generations": n_generations,
                "timeout": timeout,
                "timeout_individual": timeout_individual,
                "n_folds": n_folds
            }
            combinations += [comb]

    return combinations


def read_dataset(path: str) -> Instances:
    loader = Loader("weka.core.converters.ArffLoader")  # type: weka.core.converters.Loader

    data = loader.load_file(path)
    data.class_is_last()

    filter_obj = javabridge.make_instance('Lweka/filters/unsupervised/instance/Randomize;', '()V')
    javabridge.call(filter_obj, 'setRandomSeed', '(I)V', 1)
    javabridge.call(filter_obj, 'setInputFormat', '(Lweka/core/Instances;)Z', data.jobject)
    jtrain_data = javabridge.static_call(
        'Lweka/filters/Filter;', 'useFilter',
        '(Lweka/core/Instances;Lweka/filters/Filter;)Lweka/core/Instances;',
        data.jobject, filter_obj
    )
    data = Instances(jtrain_data)
    return data


def get_params(args: argparse.Namespace) -> dict:
    """
    Get parameters of script that is running. Makes a copy.

    :param args: The parameters as passed to this script
    :type args: argparse.Namespace
    :return: the parameters, as a dictionary
    :rtype: dict
    """
    return deepcopy(args.__dict__)


def run_external_fold(
        n_external_fold: int, n_internal_folds: int,
        dataset_name: str, datasets_path: str,
        metadata_path: str, experiment_folder: str
):
    some_exception = None  # type: Exception

    try:
        seed = Random(1)

        external_train_data = read_dataset(
            os.path.join(
                datasets_path,
                dataset_name,
                '%s-10-%dtra.arff' % (dataset_name, n_external_fold)
            )
        )  # type: Instances

        external_test_data = read_dataset(
            os.path.join(
                datasets_path,
                dataset_name,
                '%s-10-%dtst.arff' % (dataset_name, n_external_fold)
            )
        )  # type: Instances

        external_train_data.stratify(n_internal_folds)

        class_unique_values = np.array(external_train_data.attribute(external_train_data.class_index).values)

        combinations = get_pbil_combinations()  # type: list

        overall_aucs = []  # type: list
        last_aucs = []  # type: list

        for comb in combinations:
            internal_actual_classes = []
            overall_preds = []
            last_preds = []

            for n_internal_fold in range(n_internal_folds):
                internal_train_data = external_train_data.train_cv(n_internal_folds, n_internal_fold, seed)
                internal_test_data = external_train_data.test_cv(n_internal_folds, n_internal_fold)

                internal_actual_classes.extend(list(internal_test_data.values(internal_test_data.class_index)))

                pbil = PBIL(
                    resources_path=os.path.join(sys.modules['mPBIL'].__path__[0], 'resources'),
                    train_data=internal_train_data,
                    lr=comb['learning_rate'], selection_share=comb['selection_share'],
                    n_generations=comb['n_generations'], n_individuals=comb['n_individuals'],
                    timeout=comb['timeout'], timeout_individual=comb['timeout_individual'],
                    n_folds=comb['n_folds']
                )

                overall, last = pbil.run(1)

                overall_preds.extend(list(map(list, overall.predict_proba(internal_test_data))))
                last_preds.extend(list(map(list, last.predict_proba(internal_test_data))))

            internal_actual_classes = np.array(internal_actual_classes, dtype=np.int)
            overall_preds = np.array(overall_preds)
            last_preds = np.array(last_preds)

            overall_auc = 0.
            last_auc = 0.
            for i, c in enumerate(class_unique_values):
                actual_binary_class = (internal_actual_classes == i).astype(np.int)
                overall_auc += roc_auc_score(y_true=actual_binary_class, y_score=overall_preds[:, i])
                last_auc += roc_auc_score(y_true=actual_binary_class, y_score=last_preds[:, i])

            overall_aucs += [overall_auc / len(class_unique_values)]
            last_aucs += [last_auc / len(class_unique_values)]

        best_overall = int(np.argmax(overall_aucs))  # type: int
        best_last = int(np.argmax(last_aucs))  # type: int

        uses_overall = overall_aucs[best_overall] > last_aucs[best_last]
        best_index = best_overall if uses_overall else best_last  # type: int

        os.mkdir(os.path.join(metadata_path, experiment_folder, dataset_name, 'sample_01_fold_%02d' % n_external_fold))

        pbil = PBIL(
            resources_path=os.path.join(sys.modules['mPBIL'].__path__[0], 'resources'),
            train_data=external_train_data,
            lr=combinations[best_index]['learning_rate'], selection_share=combinations[best_index]['selection_share'],
            n_generations=combinations[best_index]['n_generations'],
            n_individuals=combinations[best_index]['n_individuals'],
            timeout=combinations[best_index]['timeout'], timeout_individual=combinations[best_index]['timeout_individual'],
            n_folds=combinations[best_index]['n_folds'],
            log_path=os.path.join(metadata_path, experiment_folder, dataset_name, 'sample_01_fold_%02d' % n_external_fold)
        )

        overall, last = pbil.run(1)

        clf = overall if uses_overall else last

        pbil.logger.individual_to_file(individual=clf, individual_name='last' if not uses_overall else 'overall', step=pbil.n_generation)
        pbil.logger.probabilities_to_file()

        external_preds = list(map(list, clf.predict_proba(external_test_data)))
        external_actual_classes = list(external_test_data.values(external_test_data.class_index).astype(np.int))

        with open(
                os.path.join(metadata_path, experiment_folder, dataset_name,
                             'test_sample-01_fold-%02d_parameters.json' % n_external_fold),
                'w'
        ) as write_file:
            dict_best_params = deepcopy(combinations[best_index])
            dict_best_params['individual'] = 'overall' if uses_overall else 'last'
            for k in dict_best_params.keys():
                dict_best_params[k] = str(dict_best_params[k])

            json.dump(dict_best_params, write_file, indent=2)

        with open(
                os.path.join(metadata_path, experiment_folder, dataset_name, 'overall',
                             'test_sample-01_fold-%02d_overall.preds' % n_external_fold)
                , 'w') as write_file:
            write_file.write('classValue;Individual\n')
            for i in range(len(external_actual_classes)):
                write_file.write('%r;%s\n' % (external_actual_classes[i], ','.join(map(str, external_preds[i]))))

    except Exception as e:
        some_exception = e
    finally:
        if some_exception is not None:
            raise some_exception


def create_metadata_folder(some_args: argparse.Namespace, metadata_path: str, dataset_name: str) -> str:
    experiment_folder = dt.now().strftime('%Y-%m-%d-%H-%M-%S')

    os.mkdir(os.path.join(metadata_path, experiment_folder))
    os.mkdir(os.path.join(metadata_path, experiment_folder, dataset_name))
    os.mkdir(os.path.join(metadata_path, experiment_folder, dataset_name, 'overall'))

    with open(os.path.join(metadata_path, experiment_folder, 'parameters.json'), 'w') as write_file:
        dict_params = get_params(some_args)
        json.dump(dict_params, write_file, indent=2)

    return experiment_folder


def start_jvms(heap_size):
    if not jvm.started:
        jvm.start(max_heap_size=heap_size)


def stop_jvms(_):
    if jvm.started:
        jvm.stop()


def main(args):
    n_jobs = args.n_jobs
    n_external_folds = 10  # TODO do not change this
    n_internal_folds = args.n_internal_folds

    experiment_folder = create_metadata_folder(args, args.metadata_path, args.dataset_name)

    if n_jobs == 1:
        print('WARNING: using single-thread.')
        time.sleep(5)

        some_exception = None
        jvm.start(max_heap_size=args.heap_size)
        try:
            for i in range(1, n_external_folds + 1):
                run_external_fold(
                    i, n_internal_folds,
                    args.dataset_name, args.datasets_path,
                    args.metadata_path, experiment_folder
                )
        except Exception as e:
            some_exception = e
        finally:
            jvm.stop()
            if some_exception is not None:
                raise some_exception
    else:
        print('Using %d processes' % n_jobs)
        time.sleep(5)

        with mp.Pool(processes=n_jobs) as pool:
            iterable_params = [
                (x, n_internal_folds,
                 args.dataset_name, args.datasets_path,
                 args.metadata_path, experiment_folder
                 ) for x in range(1, n_external_folds + 1)]

            pool.map(start_jvms, iterable=[args.heap_size for x in range(1, n_external_folds + 1)])
            pool.starmap(func=run_external_fold, iterable=iterable_params)
            pool.map(stop_jvms, iterable=range(1, n_external_folds + 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Runs a nested cross validation for PBIL.'
    )

    parser.add_argument(
        '--heap-size', action='store', required=False, default='2G',
        help='string that specifies the maximum size, in bytes, of the memory allocation pool. '
             'This value must be a multiple of 1024 greater than 2MB. Append the letter k or K to indicate kilobytes, '
             'or m or M to indicate megabytes. Defaults to 2G'
    )

    parser.add_argument(
        '--metadata-path', action='store', required=True,
        help='Path to where all datasets are stored'
    )

    parser.add_argument(
        '--datasets-path', action='store', required=True,
        help='Path to where all datasets are stored'
    )

    parser.add_argument(
        '--dataset-name', action='store', required=True,
        help='Name of dataset to run nested cross validation'
    )

    parser.add_argument(
        '--n-internal-folds', action='store', required=True,
        help='Number of folds to use to perform an internal cross-validation for each combination of hyper-parameters',
        type=int,
        choices=set(range(1, 6))
    )

    parser.add_argument(
        '--n-jobs', action='store', required=False,
        help='Number of parallel threads to use when running this script',
        type=int, choices=set(range(1, 11)), default=1
    )

    _some_args = parser.parse_args()

    main(args=_some_args)
