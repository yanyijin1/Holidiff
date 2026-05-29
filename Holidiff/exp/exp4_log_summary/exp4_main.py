BASELINE_METHODS = [
    'TimeGrad',
    'CSDI',
    'TSDiff',
    'Diffusion-TS',
    'MG-TSD',
    'SimDiff',
    'HoliDiff',
]


class Exp4LogSummary:
    """Placeholder for multi-metric log summarization and baseline tables."""

    def __init__(self, args=None):
        self.args = args
        self.baselines = list(BASELINE_METHODS)

    def run(self):
        raise NotImplementedError('Exp4 log summary is a placeholder and will be implemented later.')
