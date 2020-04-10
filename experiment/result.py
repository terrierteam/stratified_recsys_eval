import numpy as np
from collections import OrderedDict
from cornac.experiment.result import _table_format, Result


NUM_FMT = '{:.4f}'


class STResult(list):
    """
    Stratified Result Class for a single model
    """

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def __str__(self):
        return '[{}]\n{}'.format(self.model_name, self.table)

    def organize(self):
        headers = list(self[0].metric_avg_results.keys())
        data, index, sizes = [], [], []
        for f, r in enumerate(self):
            data.append([r.metric_avg_results[m] for m in headers])
            index.append('Q%d' % f)
            sizes.append(r.metric_avg_results['SIZE'])

        # add mean and std rows (total accumulative)
        data = np.asarray(data)
        mean, std = data.mean(axis=0), data.std(axis=0)

        # add unbiased stratified evaluation
        weights = np.asarray(sizes) / sizes[0]
        unbiased = np.average(
            data[1:], axis=0, weights=weights[1:]) * sum(weights[1:])

        # update the size
        unbiased[-1] = sizes[0]

        # update the table
        data = np.vstack([data, unbiased])
        data = [[NUM_FMT.format(v) for v in row] for row in data]
        index.extend(['Unbiased'])

        self.table = _table_format(
            data, headers, index, h_bars=[1, 2, len(data)])

        # add unbiased to the list
        self.append(Result(model_name=self[0].model_name,
                           metric_avg_results=OrderedDict(
                               zip(headers, unbiased)),
                           metric_user_results=None))
