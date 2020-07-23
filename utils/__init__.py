import json
import os
import signal
import subprocess
import sys
import time
import webbrowser

import javabridge
import numpy as np
import pandas as pd
import psutil as psutil
from scipy.io import arff
from weka.core import jvm
from weka.core.converters import Loader
from weka.core.dataset import Instances


def path_to_dataframe(dataset_path):
    """
    Reads dataframes from an .arff file, casts categorical attributes to categorical type of pandas.

    :param dataset_path:
    :return:
    """

    value, metadata = path_to_arff(dataset_path)

    df = pd.DataFrame(value, columns=metadata._attrnames)

    attributes = metadata._attributes
    for attr_name, (attr_type, rang_vals) in attributes.items():
        if attr_type in ('nominal', 'string'):
            df[attr_name] = df[attr_name].apply(lambda x: x.decode('utf-8'))

            df[attr_name] = df[attr_name].astype('category')
        elif attr_type == 'date':
            raise TypeError('unsupported attribute type!')
        else:
            df[attr_name] = df[attr_name].astype(np.float32)

    return df


def path_to_arff(dataset_path):
    """
    Given a path to a dataset, reads and returns a dictionary which comprises an arff file.

    :type dataset_path: str
    :param dataset_path: Path to the dataset. Must contain the .arff file extension (i.e., "my_dataset.arff")
    :rtype: dict
    :return: a dictionary with the arff dataset.
    """

    dataset_type = dataset_path.split('.')[-1].strip()
    assert dataset_type == 'arff', TypeError('Invalid type for dataset! Must be an \'arff\' file!')
    af = arff.loadarff(dataset_path)
    return af


def from_python_stringlist_to_java_stringlist(matrix):
    env = javabridge.get_env()  # type: javabridge.JB_Env
    # finding array's length

    # creating an empty array of arrays
    jarr = env.make_object_array(len(matrix), env.find_class('[Ljava/lang/String;'))
    # setting each item as an array of int row by row
    for i in range(len(matrix)):
        arrayobj = env.make_object_array(len(matrix[i]), env.find_class('Ljava/lang/String;'))
        for j in range(len(matrix[i])):
            env.set_object_array_element(
                arrayobj, j,
                javabridge.make_instance('Ljava/lang/String;', '(Ljava/lang/String;)V', matrix[i][j])
            )
        env.set_object_array_element(jarr, i, arrayobj)

    return jarr


def read_datasets(dataset_path, n_fold=None):
    """
    Read a dataset in .arff format.

    :param dataset_path: Can be one of two: either a path to a folder where several .arff files are setored, or a path
    to an actual .arff file.

        If the former, then n_fold must be provided. This method returns a tuple where the first
        item is the training data for the n_fold fold, and the second item is the test fold for n_fold fold.

        If the later, then this function reads the dataset and returns a python weka wrapper pointer to it.
    :type dataset_path: str
    :param n_fold: optional - must be provided if dataset_path is a folder.
    :type n_fold: int
    :rtype: tuple | weka.core.dataset.Instances
    :return: a tuple with training and test set if dataset_path is a folder and n_fold is provided; an instance of
    weka.core.dataset.Instances if dataset_path is a file.
    """

    loader = Loader("weka.core.converters.ArffLoader")

    if os.path.isdir(dataset_path):
        dataset_name = dataset_path.split(os.sep)[-1]

        train_path = os.path.join(dataset_path, '-'.join([dataset_name, '10', '%dtra.arff' % n_fold]))
        test_path = os.path.join(dataset_path, '-'.join([dataset_name, '10', '%dtst.arff' % n_fold]))

        train_data = loader.load_file(train_path)
        train_data.class_is_last()

        test_data = loader.load_file(test_path)
        test_data.class_is_last()

        filter_obj = javabridge.make_instance('Lweka/filters/unsupervised/instance/Randomize;', '()V')
        javabridge.call(filter_obj, 'setRandomSeed', '(I)V', 1)
        javabridge.call(filter_obj, 'setInputFormat', '(Lweka/core/Instances;)Z', train_data.jobject)
        jtrain_data = javabridge.static_call(
            'Lweka/filters/Filter;', 'useFilter',
            '(Lweka/core/Instances;Lweka/filters/Filter;)Lweka/core/Instances;',
            train_data.jobject, filter_obj
        )
        jtest_data = javabridge.static_call(
            'Lweka/filters/Filter;', 'useFilter',
            '(Lweka/core/Instances;Lweka/filters/Filter;)Lweka/core/Instances;',
            test_data.jobject, filter_obj
        )

        train_data = Instances(jtrain_data)
        test_data = Instances(jtest_data)

        return train_data, test_data
    else:
        train_data = loader.load_file(dataset_path)
        train_data.class_is_last()

        filter_obj = javabridge.make_instance('Lweka/filters/unsupervised/instance/Randomize;', '()V')
        javabridge.call(filter_obj, 'setRandomSeed', '(I)V', 1)
        javabridge.call(filter_obj, 'setInputFormat', '(Lweka/core/Instances;)Z', train_data.jobject)
        jtrain_data = javabridge.static_call(
            'Lweka/filters/Filter;', 'useFilter',
            '(Lweka/core/Instances;Lweka/filters/Filter;)Lweka/core/Instances;',
            train_data.jobject, filter_obj
        )

        train_data = Instances(jtrain_data)
        return train_data


def metadata_path_start(now, args, datasets_names, queue=None):
    """
    Creates a metadata path for this run.

    :param now: current time, as a datetime.datetime object.
    :param args: args object, as generated by argparse library
    :param datasets_names: names of datasets, as a list
    :param queue: queue of processes
    """

    jvm.start()

    str_time = now.strftime('%d-%m-%Y-%H-%M-%S')

    joined = os.getcwd() if not os.path.isabs(args.metadata_path) else ''
    to_process = [args.metadata_path, str_time]

    for path in to_process:
        joined = os.path.join(joined, path)
        if not os.path.exists(joined):
            os.mkdir(joined)

    with open(os.path.join(joined, 'parameters.json'), 'w') as f:
        json.dump({k: getattr(args, k) for k in args.__dict__}, f, indent=2)

    these_paths = []
    for dataset_name in datasets_names:
        local_joined = os.path.join(joined, dataset_name)
        these_paths += [local_joined]

        if not os.path.exists(local_joined):
            os.mkdir(local_joined)
            os.mkdir(os.path.join(local_joined, 'overall'))

        y_tests = []
        class_name = None
        for n_fold in range(1, 11):
            train_data, test_data = read_datasets(os.path.join(args.datasets_path, dataset_name), n_fold)
            y_tests += [test_data.values(test_data.class_attribute.index)]
            class_name = train_data.class_attribute.name

        # concatenates array of y's
        pd.DataFrame(
            np.concatenate(y_tests),
            columns=[class_name]
        ).to_csv(os.path.join(local_joined, 'overall', 'y_test.txt'), index=False)

    jvm.stop()

    if queue is not None:
        queue.put(these_paths)

    return joined


def tensorboard_start(this_path, launch_tensorboard):
    if (not is_debugging()) and launch_tensorboard:
        pid = find_process_pid_by_name('tensorboard')
        if pid is not None:
            os.kill(pid, signal.SIGKILL)

        p = subprocess.Popen(
            ["tensorboard", "--logdir", this_path, "--port", "default"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        time.sleep(1)
        webbrowser.open_new_tab("http://localhost:6006")


def find_process_pid_by_name(process_name):
    """
    Returns PID of process if it is alive and running, otherwise returns None.
    Adapted from https://thispointer.com/python-check-if-a-process-is-running-by-name-and-find-its-process-id-pid/
    """

    # Iterate over the all the running process
    for proc in psutil.process_iter():
        try:
            pinfo = proc.as_dict(attrs=['pid', 'name'])
            # Check if process name contains the given name string.
            if process_name.lower() == pinfo['name'].lower():
                return pinfo['pid']
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    return None


def is_debugging() -> bool:
    """
    Is this code running under debug mode?

    :rtype: bool
    :return: True if the program is in debug mode, False otherwise
    """
    gettrace = getattr(sys, 'gettrace', None)

    if gettrace is None:
        return False
    elif gettrace():  # in debug mode
        return True
    return False
