AGGREGATOR_CANDIDATES = ['dca', 'mom', 'simple', 'median']


class Exp6FeatureExplain:
    """Placeholder for feature-level explanation across aggregation strategies."""

    def __init__(self, args=None):
        self.args = args
        self.aggregators = list(AGGREGATOR_CANDIDATES)

    def run(self):
        raise NotImplementedError('Exp6 feature explanation is a placeholder and will be implemented later.')
