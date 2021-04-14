import re

import javabridge
import numpy as np
import pandas as pd
from weka.classifiers import Classifier

from mPBIL.pbil.ptypes import process_sample


def baseline_classifiers_options(classifiers):
    """
    Generates classifiers with default hyper-parameters.

    :rtype: tuple
    return:
    """
    log = dict()
    dict_options = dict()

    if 'SimpleCart' in classifiers:
        dict_options['SimpleCart'] = ['-M', '2', '-N', '5', '-C', '1', '-S', '1']
        log.update({
            'SimpleCart_minNumObj': 2, 'SimpleCart_heuristic': True, 'SimpleCart_useOneSE': False,
            'SimpleCart_usePrune': True, 'SimpleCart_sizePer': 1, 'SimpleCart_seed': 1, 'SimpleCart_numFoldsPruning': 5
        })
    else:
        dict_options['SimpleCart'] = []

    if 'REPTree' in classifiers:
        dict_options['REPTree'] = ['-M', '2', '-V', '0.001', '-N', '3', '-S', '1', '-L', '-1', '-I', '0.0']
        log.update({
            "REPTree_seed": 1, "REPTree_minNum": 2, "REPTree_numFolds": 3, "REPTree_maxDepth": -1,
            "REPTree_noPruning": False
        })
    else:
        dict_options['REPTree'] = []

    if 'J48' in classifiers:
        dict_options['J48'] = ['-C', '0.25', '-M', '2']
        log.update({
            'J48_pruning': 'confidenceFactor', 'J48_confidenceFactorValue': 0.25, 'J48_minNumObj': 2,
            'J48_confidenceFactorSubtreeRaising': True, 'J48_useMDLcorrection': True, 'J48_collapseTree': True,
            'J48_binarySplits': False, 'J48_useLaplace': False, 'J48_doNotMakeSplitPointActualValue': False

        })
    else:
        dict_options['J48'] = []

    if 'PART' in classifiers:
        dict_options['PART'] = ['-M', '2', '-C', '0.25', '-Q', '1']
        log.update({
            "PART_minNumObj": 2, "PART_pruning": "confidenceFactor", "PART_confidenceFactorValue": 0.25,
            "PART_seed": 1, "PART_binarySplits": False, "PART_useMDLcorrection": True,
            "PART_doNotMakeSplitPointActualValue": False
        })
    else:
        dict_options['PART'] = []

    if 'JRip' in classifiers:
        dict_options['JRip'] = ['-F', '3', '-N', '2.0', '-O', '2', '-S', '1']
        log.update({
            "JRip_folds": 3, "JRip_minNo": 2, "JRip_optimizations": 2, "JRip_seed": 1, "JRip_usePruning": True,
            "JRip_checkErrorRate": True
        })
    else:
        dict_options['JRip'] = []

    if 'DecisionTable' in classifiers:
        dict_options['DecisionTable'] = ['-R', '-X', '1', '-S', 'weka.attributeSelection.BestFirst -D 1 -N 5']
        log.update({
            "DecisionTable_crossVal": 1, "DecisionTable_search": "BestFirst", "BestFirst_direction": 1,
            "BestFirst_searchTermination": 5
        })
    else:
        dict_options['DecisionTable'] = []

    return dict_options, log


def __sample_rest__(to_sample, algorithm_name, options, clog, variables, classifiers):
    # samples other variables
    for variable_name in to_sample:
        option, log = process_sample(
            ptype=variables[algorithm_name + '_' + variable_name]['ptype'], algorithm_name=algorithm_name,
            param_name=variable_name, properties=classifiers, variables=variables
        )
        options.extend(option)
        clog.update(log)

    return options, clog


class ClassifierWrapper(Classifier):
    weka_name = None

    def __init__(self, jobject=None, options=None):
        super().__init__(classname=self.weka_name, options=options, jobject=jobject)
        self.log = None

    @staticmethod
    def sample_options(variables, classifiers):
        return [], {}

    @classmethod
    def from_jobject(cls, obj):
        env = javabridge.get_env()  # type: javabridge.JB_Env

        inst = cls(jobject=obj, options=None)
        return inst


class J48(ClassifierWrapper):
    weka_name = 'J48'

    @staticmethod
    def sample_options(variables, classifiers):
        to_sample = [
            'useLaplace', 'binarySplits', 'minNumObj', 'collapseTree', 'doNotMakeSplitPointActualValue',
            'useMDLcorrection'
        ]

        options = []
        clog = dict()

        # chooses pruning method
        pruning_method = np.random.choice(
            a=variables['J48_pruning']['params']['a'], p=variables['J48_pruning']['params']['p']
        )
        clog['J48_pruning'] = pruning_method

        if pruning_method == 'reducedErrorPruning':
            options += [classifiers['J48_reducedErrorPruning']['optionName']]
            to_sample.extend(['seed', 'numFolds', 'reducedErrorPruningSubtreeRaising'])

        elif pruning_method == 'confidenceFactor':
            to_sample.extend(['confidenceFactorValue', 'confidenceFactorSubtreeRaising'])
        elif pruning_method == 'unpruned':
            options += [classifiers['J48_unpruned']['optionName']]
        else:
            raise ValueError('Pruning method not understood: %s' % pruning_method)

        options, clog = __sample_rest__(to_sample=to_sample, algorithm_name=J48.weka_name,
                                        options=options, clog=clog, variables=variables, classifiers=classifiers)

        return options, clog

    def __str__(self):
        txt = super(J48, self).__str__()

        try:
            header = '# J48 Decision Tree'
            body = '\n\n'.join(map(lambda x: x.strip(), txt.split('------------------')[-1].split('Number of Leaves')[0].strip().replace('|   ', '  * ').split('\n')))

            return '%s\n\n%s' % (header, body)
        except:
            return txt


class PART(ClassifierWrapper):
    weka_name = 'PART'

    @staticmethod
    def sample_options(variables, classifiers):
        to_sample = ['binarySplits', 'minNumObj', 'doNotMakeSplitPointActualValue', 'useMDLcorrection']
        options = []
        clog = dict()

        # chooses pruning method
        pruning_method = np.random.choice(
            a=variables['PART_pruning']['params']['a'], p=variables['PART_pruning']['params']['p']
        )

        clog['PART_pruning'] = pruning_method

        if pruning_method == 'reducedErrorPruning':
            options += [classifiers['PART_reducedErrorPruning']['optionName']]
            to_sample.extend(['seed', 'numFolds'])
        elif pruning_method == 'confidenceFactor':
            to_sample.extend(['confidenceFactorValue'])
        elif pruning_method == 'unpruned':
            options += [classifiers['PART_unpruned']['optionName']]
        else:
            raise ValueError('Pruning method not understood: %s' % pruning_method)

        options, clog = __sample_rest__(to_sample=to_sample, algorithm_name=PART.weka_name,
                                        options=options, clog=clog, variables=variables, classifiers=classifiers)

        return options, clog

    def __str__(self):
        txt = super(PART, self).__str__()

        try:
            rules = txt.split('------------------\n\n')[-1]

            rules = list(map(lambda x: x.replace('\n', ' ').split(':'), rules.split('\n\n')[:-1]))
            df = pd.DataFrame(rules, columns=['conditions', 'predicted class'])

            fmt = ['---' for i in range(len(df.columns))]
            df_fmt = pd.DataFrame([fmt], columns=df.columns)
            df_formatted = pd.concat([df_fmt, df])
            rules_str = df_formatted.to_csv(sep="|", index=False)

            r_str = '# PART\n\nDecision list:\n\n%s' % rules_str
            return r_str
        except:
            return txt


class SimpleCart(ClassifierWrapper):
    weka_name = 'SimpleCart'

    @staticmethod
    def sample_options(variables, classifiers):
        to_sample = ['minNumObj', 'heuristic']

        options = []
        clog = dict()

        perform_pruning = np.random.choice(
            a=variables['SimpleCart_usePrune']['params']['a'], p=variables['SimpleCart_usePrune']['params']['p']
        )
        clog['SimpleCart_usePrune'] = perform_pruning

        if perform_pruning:
            options += [classifiers['SimpleCart_usePrune']['optionName']]
            to_sample.extend(['seed', 'sizePer', 'numFoldsPruning', 'useOneSE'])

        options, clog = __sample_rest__(to_sample=to_sample, algorithm_name=SimpleCart.weka_name,
                                        options=options, clog=clog, variables=variables, classifiers=classifiers)

        return options, clog

    def __str__(self):
        txt = super(SimpleCart, self).__str__()

        try:
            header = '# SimpleCart Decision Tree'
            body = '\n\n'.join(
                map(
                    lambda x: x.strip(),
                    txt.split('CART Decision Tree')[-1].split('Number of Leaf Nodes')[0].strip().replace(
                        '|  ', '  * ').split('\n')
                )
            )

            return '%s\n\n%s' % (header, body)
        except:
            return txt


class REPTree(ClassifierWrapper):
    weka_name = 'REPTree'

    @staticmethod
    def sample_options(variables, classifiers):
        to_sample = ['minNum', 'maxDepth', 'noPruning', 'numFolds', 'seed']

        options = []
        clog = dict()

        options, clog = __sample_rest__(to_sample=to_sample, algorithm_name=REPTree.weka_name,
                                        options=options, clog=clog, variables=variables, classifiers=classifiers)

        return options, clog

    def __str__(self):
        txt = super(REPTree, self).__str__()

        try:
            header = '# REPTree Decision Tree'
            body = '\n\n'.join(map(lambda x: x.strip(), txt.split('============')[-1].split('Size of the tree')[0].strip().replace('|   ', '  * ').split('\n')))
            return '%s\n\n%s' % (header, body)
        except:
            return txt


class JRip(ClassifierWrapper):
    weka_name = 'JRip'

    @staticmethod
    def sample_options(variables, classifiers):
        to_sample = ['checkErrorRate', 'minNo', 'seed']

        options = []
        clog = dict()

        option, log = process_sample(
            ptype=variables[JRip.weka_name + '_usePruning']['ptype'], algorithm_name=JRip.weka_name,
            param_name='usePruning', properties=classifiers, variables=variables
        )
        clog.update(log)
        options.extend(option)
        if clog['JRip_usePruning']:
            to_sample.extend(['folds', 'optimizations'])

        options, clog = __sample_rest__(to_sample=to_sample, algorithm_name=JRip.weka_name,
                                        options=options, clog=clog, variables=variables, classifiers=classifiers)

        return options, clog

    def __str__(self):
        txt = super(JRip, self).__str__()

        try:
            rules = txt.split('===========')[-1].split('Number of Rules')[0].strip()

            class_attr_name = np.unique(re.findall('\) => (.*)=', rules))[0]
            rules = list(map(lambda x: x.split(' => %s=' % class_attr_name), rules.split('\n')))

            df = pd.DataFrame(rules, columns=['conditions', 'predicted class'])

            fmt = ['---' for i in range(len(df.columns))]
            df_fmt = pd.DataFrame([fmt], columns=df.columns)
            df_formatted = pd.concat([df_fmt, df])
            rules_str = df_formatted.to_csv(sep="|", index=False)

            r_str = '# JRip\n\nDecision list:\n\n%s' % rules_str
            return r_str
        except:
            return txt


class DecisionTable(ClassifierWrapper):
    weka_name = 'DecisionTable'

    @staticmethod
    def sample_options(variables, classifiers):
        # -R is for always displaying rules, -S for search string
        options = ['-R']
        clog = dict()

        search_method = np.random.choice(
            a=variables['DecisionTable_search']['params']['a'], p=variables['DecisionTable_search']['params']['p']
        )
        clog['DecisionTable_search'] = search_method

        search_options = []
        if search_method == 'GreedyStepwise':
            search_options += ['-num-slots', '1']  # number of cores to run the search onto

            search_backwards, log = process_sample(
                ptype=variables[search_method + '_' + 'searchBackwards']['ptype'],
                algorithm_name=search_method, param_name='searchBackwards',
                properties=classifiers, variables=variables
            )
            search_options.extend(search_backwards)
            clog.update(log)

            if log[search_method + '_' + 'searchBackwards'] != True:
                conservative_forward, log = process_sample(
                    ptype=variables[search_method + '_' + 'conservativeForwardSelection']['ptype'],
                    algorithm_name=search_method, param_name='conservativeForwardSelection',
                    properties=classifiers, variables=variables
                )
                search_options.extend(conservative_forward)
                clog.update(log)

        elif search_method == 'BestFirst':
            surrogate_variables = ['direction', 'searchTermination']
            for sur_var in surrogate_variables:
                search_option, log = process_sample(ptype=variables[search_method + '_' + sur_var]['ptype'],
                                                    algorithm_name=search_method, param_name=sur_var,
                                                    properties=classifiers, variables=variables
                                                    )
                search_options.extend(search_option)
                clog.update(log)

        else:
            raise ValueError('Search method not understood: %s' % search_method)

        # samples other variables
        to_sample = ['evaluationMeasure', 'useIBk', 'crossVal']
        options, clog = __sample_rest__(to_sample=to_sample, algorithm_name=DecisionTable.weka_name,
                                    options=options, clog=clog, variables=variables, classifiers=classifiers)

        options.extend(['-S', ('weka.attributeSelection.%s ' % search_method) + ' '.join(search_options)])

        return options, clog

    def __str__(self):
        txt = super(DecisionTable, self).__str__()

        try:
            usesIbk = javabridge.call(self.jobject, 'getUseIBk', '()Z')

            default = 'Non matches covered by ' + ('Majority class' if usesIbk == False else 'IB1')

            lines = (txt.lower().replace('\'', '').split('rules:')[-1]).split('\n')

            sanitized_lines = []
            for line in lines:
                if (len(line) > 0) and ('=' not in line):
                    columns = line.strip().split(' ')
                    sanitized_columns = []
                    for column in columns:
                        if len(column) > 0:
                            sanitized_columns += [column]

                    sanitized_lines += [sanitized_columns]

            df = pd.DataFrame(sanitized_lines[1:], columns=sanitized_lines[0])

            fmt = ['---' for i in range(len(df.columns))]
            df_fmt = pd.DataFrame([fmt], columns=df.columns)
            df_formatted = pd.concat([df_fmt, df])
            table_str = df_formatted.to_csv(sep="|", index=False)

            r_str = '# Decision Table\n\n%s\n\n%s' % (default, table_str)
            return r_str
        except:
            return txt
