import pickle

from cornac.datasets import movielens
from eval_methods.stratified_evaluation import StratifiedEvaluation
from experiment.experiment import STExperiment
from utils import get_models, get_metrics

import sys
sys.stdout = open('log_ml_large_1M.txt', 'w', 1)

dims = [e for e in range(10, 110, 10)]

# load the movielens dataset
ml = movielens.load_feedback(variant="1M")

# propensity-based stratified evaluation
stra_eval_method = StratifiedEvaluation(data=ml,
                                        n_strata=2,
                                        rating_threshold=4.0,
                                        verbose=True)

# run the experiment
exp_stra = STExperiment(eval_method=stra_eval_method,
                        models=get_models(variant='large', dims=dims),
                        metrics=get_metrics(variant='small'),
                        verbose=True)

exp_stra.run()

with open('../data/exp_stra_ml.pkl', 'wb') as exp_file:
    pickle.dump(exp_stra.result, exp_file)
