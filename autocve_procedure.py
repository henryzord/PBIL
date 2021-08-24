"""
Procedure to experiment based on AUTOCVE EVOSTAR paper: 10 trials of a holdout with a 70/30 split, using balanced
accuracy as fitness function
"""

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

def get_pbil_combination():
    learning_rate = 0.13  # ok, checked
    selection_share = 0.3  # ok, checked
    n_individuals = 50  # ok, checked
    n_generations = 100  # ok, checked
    timeout = 5400  # ok, checked
    timeout_individual = 60  # ok, checked

    n_folds = 5  # 5-fcv # ok, checked

    comb = {
        "learning_rate": learning_rate,
        "selection_share": selection_share,
        "n_individuals": n_individuals,
        "n_generations": n_generations,
        "timeout": timeout,
        "timeout_individual": timeout_individual,
        "n_folds": n_folds,
        "fitness_metric": 'balanced_accuracy'
    }
    return comb


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


def run_holdout(
        n_trial: int, dataset_name: str, datasets_path: str, metadata_path: str, experiment_folder: str
):
    some_exception = None  # type: Exception

    try:
        train_data = read_dataset(
            os.path.join(
                datasets_path,
                'dataset_%s_trial_%02d_train.arff' % (dataset_name, n_trial)
            )
        )  # type: Instances

        test_data = read_dataset(
            os.path.join(
                datasets_path,
                'dataset_%s_trial_%02d_test.arff' % (dataset_name, n_trial)
            )
        )  # type: Instances

        # class_unique_values = np.array(train_data.attribute(train_data.class_index).values)

        combination = get_pbil_combination()  # type: dict

        good = True
        try:
            os.mkdir(os.path.join(metadata_path, experiment_folder, dataset_name, 'sample_%02d_fold_00' % n_trial))
        except:
            good = False


        pbil = PBIL(
            resources_path=os.path.join(sys.modules['mPBIL'].__path__[0], 'resources'),
            train_data=train_data,
            lr=combination['learning_rate'], selection_share=combination['selection_share'],
            n_generations=combination['n_generations'],
            n_individuals=combination['n_individuals'],
            timeout=combination['timeout'], timeout_individual=combination['timeout_individual'],
            n_folds=combination['n_folds'], fitness_metric=combination['fitness_metric'],
            log_path=os.path.join(metadata_path, experiment_folder, dataset_name, 'sample_%02d_fold_00' % n_trial) if good else None
        )

        _, clf = pbil.run(1)

        if good:
            try:
                pbil.logger.individual_to_file(individual=clf, individual_name='last', step=pbil.n_generation)
                pbil.logger.probabilities_to_file()
            except:
                pass

        external_preds = list(map(list, clf.predict_proba(test_data)))
        external_actual_classes = list(test_data.values(test_data.class_index).astype(np.int))

        with open(
                os.path.join(metadata_path, experiment_folder, dataset_name,
                             'test_sample-%02d_fold-00_parameters.json' % n_trial),
                'w'
        ) as write_file:
            dict_best_params = deepcopy(combination)
            dict_best_params['individual'] = 'last'
            for k in dict_best_params.keys():
                dict_best_params[k] = str(dict_best_params[k])

            json.dump(dict_best_params, write_file, indent=2)

        with open(
                os.path.join(metadata_path, experiment_folder, dataset_name, 'overall',
                             'test_sample-%02d_fold-00_last.preds' % n_trial)
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
    N_TRIALS = 10  # 10 runs of holdout

    experiment_folder = create_metadata_folder(args, args.metadata_path, args.dataset_name)

    if n_jobs == 1:
        print('WARNING: using single-thread.')
        time.sleep(5)

        some_exception = None
        jvm.start(max_heap_size=args.heap_size)
        try:
            for i in range(1, N_TRIALS + 1):
                run_holdout(
                    i,
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
                (x,
                 args.dataset_name, args.datasets_path,
                 args.metadata_path, experiment_folder
                 ) for x in range(1, N_TRIALS + 1)]

            pool.map(start_jvms, iterable=[args.heap_size for x in range(1, N_TRIALS + 1)])
            pool.starmap(func=run_holdout, iterable=iterable_params)
            pool.map(stop_jvms, iterable=range(1, N_TRIALS + 1))


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
        '--n-jobs', action='store', required=False,
        help='Number of parallel threads to use when running this script',
        type=int, choices=set(range(1, 11)), default=1
    )

    _some_args = parser.parse_args()

    main(args=_some_args)
