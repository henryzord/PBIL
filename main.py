import argparse
import cProfile
import itertools as it
import multiprocessing as mp
import shutil
from datetime import datetime as dt
from functools import reduce

from pbil.evaluations import EDAEvaluation, collapse_metrics, check_missing_experiments
from pbil.evaluations import evaluate_on_test
from pbil.individuals import Individual
from pbil.model import PBIL
from utils import *

pd.set_option('display.max_rows', 30)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 300)


def profile_this(*args, **kwargs):
    if kwargs['n_fold'] == 1 and kwargs['n_sample'] == 1:
        str_call = 'process_fold_sample(%s)' % (
            ','.join([str(k) + '=' + ('\'%s\'' % v if isinstance(v, str) else str(v)) for k, v in kwargs.items()])
        )
        cProfile.runctx(str_call, globals(), locals(), os.path.join(kwargs['this_path'], 'first_run.pstat'))


def process_fold_sample(
        n_sample, n_fold, n_individuals, n_generations, learning_rate, selection_share, datasets_path,
        resources_path, this_path, seed, heap_size='2G', only_baselines=False
    ):

    some_exception = None

    try:
        jvm.start(max_heap_size=heap_size)

        train_data, test_data = read_datasets(datasets_path, n_fold)

        subfolder_path = os.path.join(this_path, 'sample_%02.d_fold_%02.d' % (n_sample, n_fold))

        if os.path.exists(subfolder_path):
            shutil.rmtree(subfolder_path)
            os.mkdir(subfolder_path)

        ens_names = ['baseline']
        jobjects = []
        if not only_baselines:
            pbil = PBIL(lr=learning_rate, selection_share=selection_share,
                        n_generations=n_generations, n_individuals=n_individuals,
                        resources_path=resources_path,
                        train_data=train_data, log_path=subfolder_path
                        )

            overall, last = pbil.run(seed)

            ens_names += ['overall', 'last']
            jobjects += [overall._jobject_ensemble, last._jobject_ensemble]

            pbil.logger.individual_to_file(individual=overall, individual_name='overall', step=pbil.n_generation)
            pbil.logger.individual_to_file(individual=last, individual_name='last', step=pbil.n_generation)
            pbil.logger.probabilities_to_file()

            # noinspection PyUnusedLocal
            baseline = Individual.from_baseline(
                seed=seed, classifiers=[clf for clf in pbil.classifier_names if overall.log[clf]],
                train_data=train_data
            )
        else:
            baseline = Individual.from_baseline(
                seed=seed, classifiers=['J48', 'SimpleCart', 'PART', 'JRip', 'DecisionTable'],
                train_data=train_data
            )

        jobjects += [baseline._jobject_ensemble]

        dict_models = dict()
        for ens_name, jobject in zip(ens_names, jobjects):
            test_evaluation = EDAEvaluation.from_jobject(
                jobject=evaluate_on_test(jobject=jobject, test_data=test_data),
                data=test_data,
                seed=seed
            )

            dict_metrics = dict()
            for metric_name, metric_aggregator in EDAEvaluation.metrics:
                value = getattr(test_evaluation, metric_name)
                if isinstance(value, np.ndarray):
                    new_value = np.array2string(value.ravel().astype(np.int32), separator=',')
                    new_value_a = 'np.array(%s, dtype=np.%s).reshape(%s)' % (new_value, value.dtype, value.shape)
                    value = new_value_a

                dict_metrics[metric_name] = value

            df_metrics = pd.DataFrame(dict_metrics, index=[ens_name])
            dict_models[ens_name] = df_metrics

        collapsed = reduce(lambda x, y: x.append(y), dict_models.values())
        collapsed.to_csv(os.path.join(
                this_path, 'overall',
                'test_sample-%02.d_fold-%02.d.csv' % (n_sample, n_fold)), index=True
            )

    except Exception as e:
        some_exception = e
    finally:
        jvm.stop()
        if some_exception is not None:
            raise some_exception


def __get_running_processes__(jobs, datasets_status, n_samples, n_folds, only_baselines):
    if len(jobs) > 0:
        jobs[0].join()

        for job in jobs:  # type: mp.Process
            if not job.is_alive():
                jobs.remove(job)

    for dataset_name in datasets_status.keys():
        if not datasets_status[dataset_name]['finished']:
            missing = check_missing_experiments(
                path=os.path.join(datasets_status[dataset_name]['path'], 'overall'),
                n_samples=n_samples,
                n_folds=n_folds
            )
            if len(missing) == 0:
                try:
                    summary = collapse_metrics(
                        metadata_path=os.path.join(datasets_status[dataset_name]['path'], 'overall'),
                        n_samples=n_samples,
                        n_folds=n_folds,
                        only_baselines=only_baselines
                    )
                    datasets_status[dataset_name] = True
                    print('summary for dataset %s:' % dataset_name)
                    print(summary)
                except Exception as e:
                    print(
                        'could not collapse metrics for dataset %s. Reason: %s' % (dataset_name, str(e))
                    )

    return jobs, datasets_status


def main(args):
    assert os.path.isdir(args.datasets_path), ValueError('%s does not point to a datasets folder!' % args.datasets_path)

    datasets_names = args.datasets_names.split(',')

    now = dt.now()

    mp.set_start_method('spawn')
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    p = mp.Process(
        target=metadata_path_start, kwargs=dict(
            now=now, args=args, datasets_names=datasets_names, queue=queue
        )
    )
    p.start()
    p.join()

    these_paths = queue.get()

    combs = list(it.product(range(1, args.n_samples + 1), range(1, 11)))

    datasets_status = {k.split(os.sep)[-1]: dict(path=k, finished=False) for k in these_paths}

    jobs = []
    for (this_path, dataset_name), (n_sample, n_fold) in it.product(zip(these_paths, datasets_names), combs):
        tensorboard_start(this_path=this_path, launch_tensorboard=args.launch_tensorboard)

        if len(jobs) >= args.n_jobs:
            jobs, datasets_status = __get_running_processes__(
                jobs=jobs,
                datasets_status=datasets_status,
                n_samples=args.n_samples,
                n_folds=10,
                only_baselines=args.only_baselines
            )

        job = mp.Process(
            target=process_fold_sample,
            kwargs=dict(
                n_sample=n_sample, n_fold=n_fold,
                seed=args.seed,
                n_generations=args.n_generations,
                n_individuals=args.n_individuals,
                learning_rate=args.learning_rate,
                selection_share=args.selection_share,
                datasets_path=os.path.join(args.datasets_path, dataset_name),
                resources_path=args.resources_path,
                this_path=this_path,
                heap_size=args.heap_size,
                only_baselines=args.only_baselines
            )
        )
        time.sleep(60)  # avoids processes getting clogged and causing a bottleneck

        job.start()
        jobs += [job]

    # blocks everything
    for job in jobs:
        job.join()

    # last time for collapsing metrics
    __get_running_processes__(
        jobs=jobs,
        datasets_status=datasets_status,
        n_samples=args.n_samples,
        n_folds=10,
        only_baselines=args.only_baselines
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Main script for running Estimation of Distribution Algorithms for ensemble learning.'
    )

    parser.add_argument(
        '--datasets-path', action='store', required=True,
        help='Must lead to a path that contains several subpaths, one for each dataset. Each subpath, in turn, must '
             'have the arff files.'
    )

    parser.add_argument(
        '--datasets-names', action='store', required=True,
        help='Name of the datasets to run experiments. Must be a list separated by a comma '
             'Example:\n'
             'python script.py --datasets-names iris,mushroom,adult'
    )

    parser.add_argument(
        '--metadata-path', action='store', required=True,
        help='Path to folder where runs results will be stored.'
    )

    parser.add_argument(
        '--resources-path', action='store', required=True,
        help='Path to a folder where at least these files must exist: classifiers.json and variables.json'
    )

    parser.add_argument(
        '--seed', action='store', required=False, default=np.random.randint(np.iinfo(np.int32).max),
        help='Seed used to initialize base classifiers (i.e. Weka-related). It is not used to bias PBIL.', type=int
    )

    parser.add_argument(
        '--n-jobs', action='store', required=False, default=1,
        help='Number of jobs to use. Will use one job per sample per fold. '
             'If unspecified or set to 1, will run in a single core.',
        type=int
    )
    parser.add_argument(
        '--heap-size', action='store', required=False, default='2G',
        help='string that specifies the maximum size, in bytes, of the memory allocation pool. '
             'This value must be a multiple of 1024 greater than 2MB. Append the letter k or K to indicate kilobytes, '
             'or m or M to indicate megabytes. Defaults to 2G'
    )

    parser.add_argument(
        '--n-generations', action='store', required=True,
        help='Maximum number of generations to run the algorithm', type=int
    )

    parser.add_argument(
        '--n-individuals', action='store', required=True,
        help='Number of individuals in the population', type=int
    )

    parser.add_argument(
        '--n-samples', action='store', required=True,
        help='Number of times to run the algorithm', type=int
    )

    parser.add_argument(
        '--learning-rate', action='store', required=True,
        help='Learning rate of PBIL', type=float
    )

    parser.add_argument(
        '--selection-share', action='store', required=True,
        help='Fraction of fittest population to use to update graphical model', type=float
    )

    parser.add_argument(
        '--launch-tensorboard', action='store', required=False, default=False,
        help='Whether to launch tensorboard.'
    )

    parser.add_argument(
        '--only-baselines', action='store_true', required=False, default=False,
        help='Only run baseline algorithms.'
    )

    some_args = parser.parse_args()

    main(args=some_args)
