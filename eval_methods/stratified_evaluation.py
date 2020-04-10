import time
import powerlaw

import numpy as np
import pandas as pd

from collections import defaultdict

from cornac.utils import get_rng
from cornac.utils.common import safe_indexing
from cornac.data import Dataset
from cornac.eval_methods.base_method import BaseMethod
from cornac.eval_methods.ratio_split import RatioSplit

from experiment.result import STResult


class StratifiedEvaluation(BaseMethod):
    """Propensity-based Stratified Evaluation Method.

    Parameters
    ----------
    data: array-like, required
        Raw preference data in the triplet format [(user_id, item_id, rating_value)].

    test_size: float, optional, default: 0.2
        The proportion of the test set, 
        if > 1 then it is treated as the size of the test set.

    n_strata: int, optional, default: 5
        The number of strata for propensity-based stratification.

    rating_threshold: float, optional, default: 1.0
        Threshold used to binarize rating values into positive or negative feedback for
        model evaluation using ranking metrics (rating metrics are not affected).

    seed: int, optional, default: None
        Random seed for reproducibility.

    exclude_unknowns: bool, optional, default: True
        If `True`, unknown users and items will be ignored during model evaluation.

    verbose: bool, optional, default: False
        Output running log.
    """

    def __init__(
        self,
        data,
        test_size=0.2,
        val_size=0.0,
        n_strata=5,
        rating_threshold=1.0,
        seed=None,
        exclude_unknowns=True,
        verbose=False,
        **kwargs
    ):
        BaseMethod.__init__(
            self,
            data=data,
            rating_threshold=rating_threshold,
            seed=seed,
            exclude_unknowns=exclude_unknowns,
            verbose=verbose,
            **kwargs
        )

        self.n_strata = n_strata

        # estimate propensities
        self.props = self._estimate_propensities()

        # split the data into train/valid/test sets
        self.train_size, self.val_size, self.test_size = RatioSplit.validate_size(
            val_size, test_size, len(self._data))
        self._split()

    def _split(self):
        data_idx = self.rng.permutation(len(self._data))
        train_idx = data_idx[:self.train_size]
        test_idx = data_idx[-self.test_size:]
        val_idx = data_idx[self.train_size:-self.test_size]

        train_data = safe_indexing(self._data, train_idx)
        test_data = safe_indexing(self._data, test_idx)
        val_data = safe_indexing(self._data, val_idx) if len(
            val_idx) > 0 else None

        self._build_stratified_datasets(train_data=train_data,
                                        test_data=test_data,
                                        val_data=val_data)

    def _estimate_propensities(self):

        # find the item's frequencies
        item_freq = defaultdict(int)
        for u, i, r in self._data:
            item_freq[i] += 1

        # fit the exponential param
        data = np.array([e for e in item_freq.values()], dtype=np.float)
        results = powerlaw.Fit(data, discrete=True,
                               fit_method='Likelihood')
        alpha = results.power_law.alpha
        fmin = results.power_law.xmin

        if self.verbose:
            print('Powerlaw exponential estimates: %f, min=%d' % (alpha, fmin))

        # replace raw frequencies with the estimated propensities
        for k, v in item_freq.items():
            if v > fmin:
                item_freq[k] = pow(v, alpha)

        return item_freq  # user-independent propensity estimations

    def _build_stratified_datasets(self, train_data, test_data, val_data):

        if train_data is None or len(train_data) == 0:
            raise ValueError("train_data is required but None or empty!")
        if test_data is None or len(test_data) == 0:
            raise ValueError("test_data is required but None or empty!")

        self.global_uid_map.clear()
        self.global_iid_map.clear()

        # build training set
        self.train_set = Dataset.build(
            data=train_data,
            fmt=self.fmt,
            global_uid_map=self.global_uid_map,
            global_iid_map=self.global_iid_map,
            seed=self.seed,
            exclude_unknowns=False,
        )
        if self.verbose:
            print("---")
            print("Training data:")
            print("Number of users = {}".format(self.train_set.num_users))
            print("Number of items = {}".format(self.train_set.num_items))
            print("Number of ratings = {}".format(self.train_set.num_ratings))
            print("Max rating = {:.1f}".format(self.train_set.max_rating))
            print("Min rating = {:.1f}".format(self.train_set.min_rating))
            print("Global mean = {:.1f}".format(self.train_set.global_mean))

        # build test set
        self.test_set = Dataset.build(
            data=test_data,
            fmt=self.fmt,
            global_uid_map=self.global_uid_map,
            global_iid_map=self.global_iid_map,
            seed=self.seed,
            exclude_unknowns=self.exclude_unknowns,
        )
        if self.verbose:
            print("---")
            print("Test data (Q0):")
            print("Number of users = {}".format(len(self.test_set.uid_map)))
            print("Number of items = {}".format(len(self.test_set.iid_map)))
            print("Number of ratings = {}".format(self.test_set.num_ratings))
            print("Max rating = {:.1f}".format(self.test_set.max_rating))
            print("Min rating = {:.1f}".format(self.test_set.min_rating))
            print("Global mean = {:.1f}".format(self.test_set.global_mean))
            print(
                "Number of unknown users = {}".format(
                    self.test_set.num_users - self.train_set.num_users
                )
            )
            print(
                "Number of unknown items = {}".format(
                    self.test_set.num_items - self.train_set.num_items
                )
            )

        # build stratified datasets
        self.stratified_sets = {}

        # match the corresponding propensity score for each feedback
        test_props = np.array([self.props[i]
                               for u, i, r in test_data], dtype=np.float64)

        # stratify
        strata, bins = pd.cut(x=test_props,
                              bins=self.n_strata,
                              labels=['Q%d' %
                                      i for i in range(1, self.n_strata+1)],
                              retbins=True)

        for stratum in sorted(np.unique(strata)):

            # sample the corresponding sub-population
            qtest_data = []
            for (u, i, r), q in zip(test_data, strata):
                if q == stratum:
                    qtest_data.append((u, i, r))

            # build a dataset
            qtest_set = Dataset.build(
                data=qtest_data,
                fmt=self.fmt,
                global_uid_map=self.global_uid_map,
                global_iid_map=self.global_iid_map,
                seed=self.seed,
                exclude_unknowns=self.exclude_unknowns,
            )
            if self.verbose:
                print("---")
                print("Test data ({}):".format(stratum))
                print("Number of users = {}".format(
                    len(qtest_set.uid_map)))
                print("Number of items = {}".format(
                    len(qtest_set.iid_map)))
                print("Number of ratings = {}".format(
                    qtest_set.num_ratings))
                print("Max rating = {:.1f}".format(qtest_set.max_rating))
                print("Min rating = {:.1f}".format(qtest_set.min_rating))
                print("Global mean = {:.1f}".format(qtest_set.global_mean))
                print(
                    "Number of unknown users = {}".format(
                        qtest_set.num_users - self.train_set.num_users
                    )
                )
                print(
                    "Number of unknown items = {}".format(
                        self.test_set.num_items - self.train_set.num_items
                    )
                )

            self.stratified_sets[stratum] = qtest_set

        if val_data is not None and len(val_data) > 0:
            self.val_set = Dataset.build(
                data=val_data,
                fmt=self.fmt,
                global_uid_map=self.global_uid_map,
                global_iid_map=self.global_iid_map,
                seed=self.seed,
                exclude_unknowns=self.exclude_unknowns,
            )
            if self.verbose:
                print("---")
                print("Validation data:")
                print("Number of users = {}".format(len(self.val_set.uid_map)))
                print("Number of items = {}".format(len(self.val_set.iid_map)))
                print("Number of ratings = {}".format(self.val_set.num_ratings))

        if self.verbose:
            print("---")
            print("Total users = {}".format(self.total_users))
            print("Total items = {}".format(self.total_items))

        self.train_set.total_users = self.total_users
        self.train_set.total_items = self.total_items

        self._build_modalities()

        return self

    def evaluate(self, model, metrics, user_based, show_validation):

        result = STResult(model.name)

        if self.train_set is None:
            raise ValueError("train_set is required but None!")
        if self.test_set is None:
            raise ValueError("test_set is required but None!")

        self._reset()
        self._organize_metrics(metrics)

        ###########
        # FITTING #
        ###########
        if self.verbose:
            print("\n[{}] Training started!".format(model.name))

        start = time.time()
        model.fit(self.train_set, self.val_set)
        train_time = time.time() - start

        ##############
        # EVALUATION #
        ##############

        if self.verbose:
            print("\n[{}] Evaluation started!".format(model.name))

        # evaluate on the sampled test set
        start = time.time()
        test_result = self._eval(
            model=model,
            test_set=self.test_set,
            val_set=self.val_set,
            user_based=user_based,
        )
        test_time = time.time() - start
        test_result.metric_avg_results["SIZE"] = self.test_set.num_ratings
        result.append(test_result)

        if self.verbose:
            print("\n[{}] Stratified Evaluation started!".format(model.name))

        # evaluate on different strata
        start = time.time()

        for stratum, qtest_set in self.stratified_sets.items():

            qtest_result = self._eval(
                model=model,
                test_set=qtest_set,
                val_set=self.val_set,
                user_based=user_based,
            )

            test_time = time.time() - start
            qtest_result.metric_avg_results["SIZE"] = qtest_set.num_ratings

            result.append(qtest_result)

        result.organize()

        val_result = None
        if show_validation and self.val_set is not None:
            start = time.time()
            val_result = self._eval(
                model=model, test_set=self.val_set, val_set=None, user_based=user_based
            )
            val_time = time.time() - start

        return result, val_result
