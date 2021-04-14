import argparse
import operator as op
import subprocess
from functools import reduce

import pandas as pd
import numpy as np

import os

import matplotlib as mpl
mpl.use('agg')
import tensorflow as tf
from matplotlib import pyplot as plt
import inspect
import graphviz

_probabilities_path_name = '--probabilities-path'


class Logger(object):
    cast_functions = {
        'scalar': lambda value, placeholder: np.dtype(placeholder.dtype.as_numpy_dtype).type(value),
        'hist': lambda value, placeholder: value.astype(placeholder.dtype.as_numpy_dtype),
        'text': lambda value, placeholder: np.dtype(placeholder.dtype.as_numpy_dtype).type(value)
    }

    def __init__(self, logdir_path, histogram_names=None, scalars=None, text_names=None):
        if PBILLogger.tf_session is not None:
            PBILLogger.tf_session.close()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        PBILLogger.tf_session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        PBILLogger.tf_session.as_default()

        self.logdir_path = logdir_path
        if not os.path.exists(self.logdir_path):
            os.mkdir(self.logdir_path)

        # initializes summary writer
        self.summary_writer = tf.summary.FileWriter(self.logdir_path)

        self.scalars = scalars
        scalar_names = list(self.scalars.keys())

        if scalar_names is not None:
            for scalar_name in scalar_names:
                if getattr(Logger, '%s_scalar_placeholder' % scalar_name, None) is None:
                    placeholder = tf.placeholder(tf.float32)

                    setattr(Logger, '%s_scalar_placeholder' % scalar_name, placeholder)
                    setattr(
                        Logger, '%s_scalar' % scalar_name,
                        tf.summary.scalar(scalar_name, placeholder)
                    )

        if histogram_names is not None:
            for histogram_name in histogram_names:
                if getattr(Logger, '%s_hist_placeholder' % histogram_name, None) is None:
                    placeholder = tf.placeholder(tf.float32, shape=[None])
                    setattr(Logger, '%s_hist_placeholder' % histogram_name, placeholder)
                    setattr(
                        Logger, '%s_hist' % histogram_name,
                        tf.summary.histogram(histogram_name, placeholder)
                    )

        if text_names is not None:
            for text_name in text_names:
                if getattr(Logger, '%s_text_placeholder' % text_name, None) is None:
                    placeholder = tf.placeholder(tf.string)

                    setattr(Logger, '%s_text_placeholder' % text_name, placeholder)
                    setattr(
                        Logger, '%s_text' % text_name,
                        tf.summary.text(text_name, placeholder)
                    )

        self.scalar_names = scalar_names
        self.histogram_names = histogram_names
        self.text_names = text_names

        self.prob_record = []
        self.pop_record = []

    def __core_log__(self, value, step, name, tf_type):
        placeholder = getattr(self, '%s_%s_placeholder' % (name, tf_type))  # type: tf.Tensor
        histogram = getattr(self, '%s_%s' % (name, tf_type))  # type: tf.Tensor

        summary = PBILLogger.tf_session.run(
            histogram, feed_dict={placeholder: Logger.cast_functions[tf_type](value=value, placeholder=placeholder)}
        )
        self.summary_writer.add_summary(summary, step)
        # self.summary_writer.flush()  # forces writing to tensorboard

    def log_histogram(self, values, step, histogram_name):
        """

        :param values: Array of values to put into the histogram.
        :type values: numpy.ndarray
        :param step: The number of the histogram that is being logged.
        :param histogram_name: Name of the histogram to log.
        :type histogram_name: str
        :return:
        """
        self.__core_log__(value=values, step=step, name=histogram_name, tf_type='hist')

    def log_scalar(self, value, step, scalar_name):
        self.__core_log__(value=value, step=step, name=scalar_name, tf_type='scalar')

    def log_text(self, value, step, text_name):
        self.__core_log__(value=value, step=step, name=text_name, tf_type='text')


class PBILLogger(Logger):
    tf_session = None  # type: tf.Session

    classifiers_filename = 'classifiers'
    graph_filename = 'graph'

    population_operators = [
        ('min', np.min),
        ('max', np.max),
        ('mean', np.mean),
        ('std', np.std)
    ]

    def log_probabilities(self, variables):
        data = {}
        for name, info in variables.items():
            if info['ptype'] == 'discrete':
                try:
                    data.update({
                        name + '_' + str(a): p for a, p in zip(info['params']['a'], info['params']['p'])
                    })
                except TypeError:  # variable has non-learnable probabilities
                    pass
            elif info['ptype'] == 'continuous':
                data.update({
                    name + '_loc': info['params']['loc'], name + '_scale': info['params']['scale'],
                })
            else:
                raise TypeError('ptype for variable %s not understood: %s' % (name, info['ptype']))

        self.prob_record += [data]

    def log_population(self, population, halloffame):
        """

        :param population: current (general) population (i.e. not only fittest)
        :type population: list
        """

        fitnesses = [x.fitness for x in population]
        last = population[np.argmax(fitnesses)]
        overall = halloffame[0]

        n_generation = len(self.pop_record)

        self.pop_record += [dict()]
        for k, v in self.scalars.items():
            this_metric = v(population=population, last=last, overall=overall)
            self.log_scalar(value=this_metric, step=n_generation, scalar_name=k)
            self.pop_record[-1][v] = this_metric

        self.log_histogram(values=np.array(fitnesses), step=n_generation, histogram_name='fitness')

    @staticmethod
    def __convert_img_and_save__(logger, matrix, name, step):
        """
        Converts an image to the format accepted by tensorflow image tab.

        :param logger:
        :param matrix:
        :param name:
        :param step:
        :return:
        """
        matrix = (matrix * 255).round().astype(np.uint8)
        matrix.shape = (1,) + matrix.shape
        _img = tf.summary.image(name=name, tensor=matrix)
        _summary = PBILLogger.tf_session.run(_img)
        logger.summary_writer.add_summary(_summary, step)

    def probabilities_to_file(self):
        df = pd.DataFrame(self.prob_record)

        # columns that encode the probability that a binary variable assumes False; can be safely dropped
        dfa = df.loc[:, ['false' not in x.lower() for x in df.columns]]  # type: pd.DataFrame
        dfb = dfa.join(pd.DataFrame(self.pop_record, index=pd.RangeIndex(1, len(self.pop_record) + 1)))
        dfb.to_csv(os.path.join(self.logdir_path, 'probabilities.csv'), index=False)

        write_probability_scale(path=os.path.join(
            self.logdir_path, 'probability_scale.csv'),
            probabilities=dfb
        )
        write_compiling_script(self.logdir_path)

    def individual_to_file(self, individual, individual_name, step):
        tf_clf_texts = ''
        file_clf_texts = ''
        for clf in individual.classifiers:
            if clf.graph is not None:
                clf_name = clf.classname.split('.')[-1]
                dot = graphviz.Source(clf.graph)  # type: graphviz.Source

                _clf_path = os.path.join(
                    self.logdir_path, '_'.join([individual_name, clf_name, PBILLogger.graph_filename])
                )

                # otherwise will not export png
                subprocess.call(['dot', '-c'])

                dot.render(
                    filename=_clf_path,
                    format='png',
                    cleanup=True
                )
                tfimg = plt.imread(_clf_path + '.png')

                PBILLogger.__convert_img_and_save__(
                    logger=self,
                    matrix=tfimg,
                    name=individual_name + '_' + clf_name,
                    step=step
                )

                tf_clf_texts += '%s\n\n' % str(clf)
                file_clf_texts += '# %s\n\n![](%s_%s_graph.png)\n\n' % (clf_name, individual_name, clf_name)

            else:
                _to_add = '%s\n\n' % str(clf)
                file_clf_texts += _to_add
                tf_clf_texts += _to_add

        # writes classifiers' texts to tensorboard
        self.log_text(
            value=tf_clf_texts,
            step=step,
            text_name=individual_name
        )

        with open(os.path.join(self.logdir_path, '_'.join([individual_name, PBILLogger.classifiers_filename]) + '.md'), 'w') as classifiers_md:
            classifiers_md.write(file_clf_texts)

        self.summary_writer.flush()


def write_compiling_script(logdir_path):
    with open(os.path.join(logdir_path, 'compile_results.py'), 'w') as some_file:
        some_file.write(inspect.getsource(log_graphs))
        some_file.write('\n\nif __name__ == \'__main__\':\n\tlog_graphs(\'%s\')' % logdir_path)


# noinspection PyUnresolvedReferences
def log_graphs(probabilities_path):
    """
    Plots probability graphs to files, one file per graph
    :param probabilities_path:
    :return:
    """
    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib.cm import viridis
    from matplotlib.colors import to_hex
    import os

    dfb = pd.read_csv(os.path.join(probabilities_path, 'probabilities.csv'))
    variables = np.unique(['_'.join(y.split('_')[:-1]) for y in [x for x in dfb.columns if 'fitness' not in x]])

    _fitness = dfb['fitness_mean']

    ''' Creates one plot per variable '''
    for variable_name in variables:
        plt.clf()
        ax = plt.subplot()

        try:
            if (variable_name + '_scale' in dfb.columns) and (variable_name + '_loc' in dfb.columns):
                raw = [variable_name + '_' + x for x in ['loc', 'scale']]
            else:  # is continuous
                raw = [x for x in dfb.columns if variable_name in x]
        except TypeError as te:
            continue

        values = dfb.loc[:, raw]

        colors = [to_hex(x) for x in viridis(np.linspace(0, 1, len(values.columns) + 1))]

        ax.set_title(variable_name)
        last_bottom = np.zeros(len(values))
        x_axis = np.arange(len(values))
        width = 0.98
        bars = []
        labels = []
        for i, val_name in enumerate(values.columns):
            bars += [plt.bar(x_axis, values[val_name], bottom=last_bottom, width=width, color=colors[i])]
            labels += [val_name.split('_')[-1]]
            last_bottom += values[val_name]

        bars += [ax.plot(_fitness, label='avg fitness', c=colors[-1])[0]]
        labels += ['avg fitness']

        ax.set_xlabel('generation')
        ax.set_ylabel('probability')

        # puts legend outside plot
        box = ax.get_position()
        ax.set_position([box.x0, box.y0 + box.height * 0.2, box.width, box.height * 0.8])

        ax.legend(
            tuple(bars), tuple(labels), loc='upper center',
            bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=3
        )

        plt.savefig(
            os.path.join(probabilities_path, 'probs_%s.pdf' % variable_name)
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Main script for running Estimation of Distribution Algorithms for ensemble learning.'
    )

    parser.add_argument(
        _probabilities_path_name, action='store', required=True,
        help='Path where probabilities are store.'
    )

    args = parser.parse_args()

    log_graphs(probabilities_path=args.probabilities_path)


def write_probability_scale(path, probabilities):
    abs(probabilities.iloc[0] - probabilities.iloc[-1]).sort_values(ascending=False).to_csv(path)


def generate_probability_averager_script(dataset_path):

    subfolders = set(os.listdir(dataset_path)) - {'overall'}
    probss = []
    lens = []
    for subfolder in subfolders:
        prob_file = os.path.join(dataset_path, subfolder, 'probabilities.csv')
        probs = pd.read_csv(prob_file)
        probss += [probs]
        lens += [len(probs)]

    max_len = max(lens)
    nx = np.arange(max_len)
    for i, prob in enumerate(probss):
        nox = np.linspace(0, max_len, num=len(prob))
        ny = {}
        for column in prob.columns:
            ny[column] = np.interp(nx, nox, prob[column])
        probss[i] = pd.DataFrame(ny)

    probss = reduce(op.add, probss) / len(subfolders)
    probss.to_csv(os.path.join(dataset_path, 'overall', 'probabilities.csv'), index=False)

    write_probability_scale(path=os.path.join(dataset_path, 'overall', 'probability_scale.csv'), probabilities=probss)

    write_compiling_script(os.path.join(dataset_path, 'overall'))
