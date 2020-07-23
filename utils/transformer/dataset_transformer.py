"""
This script reads a path with several datasets in .dat format (the KEEL databaset preferred format) and converts to
arff. Performs all conversions required (e.g. removing lines that are not supported in arff, changing attribute types
from real/integer to numeric, etc).

It also has a function for summarizing information on datasets from a folder.
"""

import re

import numpy as np
import pandas as pd
import argparse
import os
from scipy.io import arff
import subprocess


def is_weka_compatible(path_dataset):
    """
    Checks whether a dataset is in the correct format for being used by Weka. Throws an exception if not.

    :param path_dataset: Path to dataset. Must be in .arff format.
    """

    ff = arff.loadarff(open(path_dataset, 'r'))  # checks if file opens in scipy

    p = subprocess.Popen(
        ["java", "-classpath", "/home/henry/weka-3-8-3/weka.jar", "weka.classifiers.rules.ZeroR", "-t", path_dataset],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    output, err = p.communicate(b"input data that is passed to subprocess' stdin")
    rc = p.returncode

    if rc != 0:
        raise Exception('Weka could not process correctly the dataset.')


def convert_to_arff(dataset_path):
    list_datasets = os.listdir(dataset_path)

    for dataset in list_datasets:
        list_files = os.listdir(os.path.join(dataset_path, dataset))
        for filename in list_files:
            raw = filename.split('.')[0].split('-')
            file_ext = filename.split('.')[-1]
            if file_ext == 'arff':
                continue
            fold_mode = raw[-1]
            n_folds = raw[-2]
            name = '-'.join(raw[:-2])
            n_fold, mode = int(fold_mode[:-3]), fold_mode[-3:]

            old_out_path = os.path.join(dataset_path, dataset, filename)
            new_out_path = os.path.join(dataset_path, dataset, filename.split('.')[0] + '.arff')

            with open(old_out_path) as read_file, \
                    open(new_out_path, 'w') as write_file:

                for line in read_file:
                    line = line.lower()

                    if ('@input' not in line) and ('@output' not in line):
                        line = line.replace('<null>', '?')
                        line = line.replace('{', ' {')

                        _matches = re.findall("integer\\[.*\\]", line) + re.findall("real\\[.*\\]", line)
                        for _match in _matches:
                            line = line.replace(_match, 'numeric')

                        write_file.write(line)

            try:
                print('checking %s' % new_out_path)
                is_weka_compatible(new_out_path)
                os.remove(old_out_path)

            except Exception as e:
                raise e

        print('finished dataset %s' % dataset)


def get_metadata(dataset_path):

    list_datasets = os.listdir(dataset_path)

    df = pd.DataFrame(
        index=list_datasets,
        columns=['n_instances', 'n_attributes', 'n_categorical', 'n_numeric', 'n_classes', 'n_missing'], dtype=np.float32
    )

    df[['n_instances', 'n_missing']] = 0

    for dataset in list_datasets:
        list_files = os.listdir(os.path.join(dataset_path, dataset))
        for filename in list_files:
            print('processing %s' % filename)

            raw = filename.split('.')[0].split('-')
            file_ext = filename.split('.')[-1]
            if file_ext == '.dat':
                continue
            fold_mode = raw[-1]
            n_folds = raw[-2]
            name = '-'.join(raw[:-2])
            n_fold, mode = int(fold_mode[:-3]), fold_mode[-3:]

            old_out_path = os.path.join(dataset_path, dataset, filename)

            with open(old_out_path) as read_file:
                n_attributes = 0
                reached_data = False
                count_lines = 0
                n_missing = 0
                last_line = None
                n_categorical = 0
                n_numeric = 0
                for line in read_file:
                    line = line.lower()

                    if reached_data:
                        count_lines += 1

                        if '?' in line:
                            n_missing += 1

                    elif '@data' in line:
                        reached_data = True
                        # collects class labels
                        n_classes = len(re.findall('\\{.*\\}', last_line)[0][1:-2].split(','))

                    elif '@attribute' in line:
                        if 'numeric' in line or 'real' in line or 'integer' in line:
                            n_numeric += 1
                        else:
                            n_categorical += 1

                        n_attributes += 1

                    last_line = line

                if mode == 'tst':
                    df.loc[name, 'n_instances'] += count_lines
                    df.loc[name, 'n_missing'] += n_missing

                    n_attributes -= 1  # removes class attribute from the count
                    n_categorical = max(0, n_categorical - 1)  # removes class attribute from the class

                    pairs = [('n_classes', n_classes), ('n_attributes', n_attributes),
                             ('n_categorical', n_categorical), ('n_numeric', n_numeric)]

                    for k, v in pairs:
                        if np.isnan(df.loc[name, k]):
                            df.loc[name, k] = v
                        elif df.loc[name, k] != v:
                            raise ValueError('Found different number of %s among folds!' % k)

        df.loc[name, 'n_missing'] = df.loc[name, 'n_missing'] / df.loc[name, 'n_instances']

    print(df)
    df.to_csv('keel-datasets_10fcv_metatadata.csv', index=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script for checking if datasets are OK.'
    )
    parser.add_argument(
        '-datasets-path', action='store', required=True,
        help='Path to datasets folder. Datasets must be in .arff format.'
    )
    args = parser.parse_args()
    get_metadata(dataset_path=args.datasets_path)
