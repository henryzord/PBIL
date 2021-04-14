import argparse
import subprocess
import os


def main(datasets_path, weka_path, n_folds):
    seed = 0

    default_params = ['java', '-classpath', weka_path, 'weka.filters.supervised.instance.StratifiedRemoveFolds',
                      '-c', "last", '-S', str(seed), '-N', str(n_folds)]

    folders = [x for x in os.listdir(datasets_path) if os.path.isdir(os.path.join(datasets_path, x))]
    for folder in folders:
        input = os.path.join(datasets_path, folder, folder + ".arff")
        for i in range(1, 10 + 1):
            params = default_params + ['-i', input, '-F', str(i),
                                       '-o', os.path.join(datasets_path, folder, folder + '-10-%dtst.arff' % i)]
            subprocess.call(params)
            params = default_params + ['-V', '-i', input, '-F', str(i),
                                       '-o', os.path.join(datasets_path, folder, folder + '-10-%dtra.arff' % i)]
            subprocess.call(params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Given a dataset in .arff format, generates 10 folds, '
                    'in the format available at KEEL dataset repository. This script requires Java and a valid Weka'
                    'installation to properly run.'
    )

    parser.add_argument(
        '--weka-path', action='store', required=True,
        help='Path to weka .jar'
    )

    parser.add_argument(
        '--datasets-path', action='store', required=True,
        help='Path to datasets that must have their folds generated'
    )

    parser.add_argument(
        '--n-folds', action='store', required=True,
        help='Number of folds to generate.'
    )

    some_args = parser.parse_args()

    main(datasets_path=some_args.datasets_path, weka_path=some_args.weka_path, n_folds=args.n_folds)
