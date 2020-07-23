
class Aggregator(object):
    def __init__(self, variables, **kwargs):
        self.hyperparameters = variables
        self.options = []

    @staticmethod
    def sample_options(variables):
        return [], {}

    def aggregate(self, predictions, **kwargs):
        raise NotImplementedError('not implemented yet!')

    def aggregate_proba(self, predictions, **kwargs):
        raise NotImplementedError('not implemented yet!')


class CompetenceBasedAggregator(Aggregator):
    pass


class MajorityVotingAggregator(Aggregator):
    pass


def baseline_aggregator_options(variables):
    options, _ = MajorityVotingAggregator.sample_options(variables=variables)
    log = dict(Aggregator='MajorityVotingAggregator')
    return {'Aggregator': ['MajorityVotingAggregator'] + options}, log


def configure_aggregator(name, **kwargs):
    aggregator = eval(name)(**kwargs)
    options = aggregator.options
    return aggregator, options
