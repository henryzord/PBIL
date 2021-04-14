import numpy as np
from collections import Counter


def process_sample(ptype, **kwargs):
    """

    :param ptype:
    :type ptype: str
    :param kwargs:
    :return:
    """
    return eval(ptype.capitalize() + '.sample')(**kwargs)


def process_update(ptype, **kwargs):
    return eval(ptype.capitalize() + '.update')(**kwargs)


class AttributeType(object):
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def sample(algorithm_name, param_name, properties, variables):
        pass

    @staticmethod
    def update(variable_name, variable_data, observations, lr, n_generations):
        pass


class Discrete(AttributeType):
    @staticmethod
    def sample(algorithm_name, param_name, properties, variables):
        composite_name = algorithm_name + '_' + param_name
        
        sampled = np.random.choice(
            a=variables[composite_name]['params']['a'],
            p=variables[composite_name]['params']['p']
        )

        register = ('learnable' not in variables[composite_name]) or (variables[composite_name]['learnable'])

        if variables[composite_name]['dtype'] != 'np.bool':
            return [properties[composite_name]['optionName'], str(sampled)], {composite_name: sampled} if register else {}
        else:
            if sampled == True:
                if properties[composite_name]['presenceMeans'] == True:
                    return [properties[composite_name]['optionName']], {composite_name: True} if register else {}
                else:
                    return [], {composite_name: True} if register else {}
            else:
                if properties[composite_name]['presenceMeans'] == False:
                    return [properties[composite_name]['optionName']], {composite_name: False} if register else {}
                else:
                    return [], {composite_name: False} if register else {}

    @staticmethod
    def update(variable_name, variable_data, observations, lr, n_generations):
        if variable_name not in observations.columns:  # untracked variable
            return variable_data

        occurrences = Counter(observations[variable_name].dropna())
        n_observations = sum(occurrences.values())

        for i, k in enumerate(variable_data['params']['a']):
            variable_data['params']['p'][i] = (1. - lr) * variable_data['params']['p'][i] + \
                                              lr * occurrences[k] / float(n_observations)

        # prevents that really small fluctuations in the probability distribution affect the computation of the rest
        rest = np.clip(a=1. - sum(variable_data['params']['p']), a_min=0, a_max=1)
        variable_data['params']['p'][np.random.choice(len(variable_data['params']['p']))] += rest

        return variable_data


class Continuous(AttributeType):
    @staticmethod
    def sample(algorithm_name, param_name, properties, variables):
        composite_name = algorithm_name + '_' + param_name
        register = ('learnable' not in variables[composite_name]) or (variables[composite_name]['learnable'])

        sampled = np.clip(
            a=np.random.normal(
                loc=variables[composite_name]['params']['loc'],
                scale=variables[composite_name]['params']['scale']
            ),
            a_min=variables[composite_name]['params']['a_min'],
            a_max=variables[composite_name]['params']['a_max']
        )
        return [properties[composite_name]['optionName'], str(sampled)], {composite_name: sampled} if register else {}

    @staticmethod
    def update(variable_name, variable_data, observations, lr, n_generations):
        if variable_name not in observations.columns:  # untracked variable
            return variable_data

        observed = observations[variable_name].astype(np.float64).dropna()
        diff = np.mean(observed) - variable_data['params']['loc']

        variable_data['params']['loc'] += lr * diff
        variable_data['params']['scale'] = max(0,
                                               variable_data['params']['scale'] - (variable_data['params']['scale_init']
                                                                                   / float(n_generations))
                                               )

        return variable_data
