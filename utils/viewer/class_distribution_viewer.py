"""
Script for generating distribution of instances among classes for datasets.
This script requires Java and a valid Weka installation.
"""

import os
import argparse
import numpy as np
from weka.core import jvm
from collections import Counter
from matplotlib import pyplot as plt

from utils import read_datasets


def main(read_path, write_path, fileformat='png'):

    some_exception = None
    try:
        jvm.start()
        dataset_names = os.listdir(read_path)
        for dataset in dataset_names:
            print(dataset)

            train_data, test_data = read_datasets(os.path.join(read_path, dataset), n_fold=1)

            for inst in test_data:
                train_data.add_instance(inst)

            y = train_data.values(train_data.class_attribute.index)

            fig, ax = plt.subplots(figsize=(1, 1))  # type: (plt.Figure, plt.Axes)

            classes = sorted(np.unique(y))
            xticks = np.arange(len(classes))

            counts = Counter(y)

            ax.bar(xticks, height=[counts[c] for c in classes])

            ax.set_xticks(xticks)
            ax.set_xticklabels(classes)

            plt.axis('off')

            plt.savefig(os.path.join(write_path, '.'.join([dataset, fileformat])), format=fileformat, transparent=True)

            plt.clf()
            plt.close()

    except Exception as e:
        some_exception = e
    finally:
        jvm.stop()
        if some_exception is not None:
            raise some_exception


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script for generating distribution of instances among classes for datasets. This script requires '
                    'Java and a valid Weka installation.'
    )
    parser.add_argument(
        '-datasets-path', action='store', required=True,
        help='Path to datasets folder. Datasets must be in .arff format.'
    )

    parser.add_argument(
        '-write-path', action='store', required=True,
        help='Path where to write figures.'
    )

    args = parser.parse_args()
    main(read_path=args.datasets_path, write_path=args.write_path)
