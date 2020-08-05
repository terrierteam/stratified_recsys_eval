import pickle

from experiment.experiment import STExperiment
from eval_methods.stratified_evaluation import StratifiedEvaluation
from datasets import yahoo_music, coats
from utils import get_models, get_metrics
from cornac.experiment.experiment import Experiment
from cornac.eval_methods.base_method import BaseMethod

import sys
sys.stdout = open('log_yahoo.txt', 'w', 1)

dims = [e for e in range(10, 110, 10)]

print('-------OPEN LOOP EVALUATION-------')

# load the closed/open loop datasets
ds_closed = yahoo_music.load_feedback(variant='closed_loop')
ds_open = yahoo_music.load_feedback(variant='open_loop')


# train on closed-loop dataset and evaluate on open loop (random) dataset
eval_method = BaseMethod.from_splits(train_data=ds_closed,
                                     test_data=ds_open,
                                     rating_threshold=4.0,
                                     verbose=True)

# run the experiment
exp_open = Experiment(eval_method=eval_method,
                      models=get_models(variant='large', dims=dims),
                      metrics=get_metrics(variant='large'),
                      verbose=True)

exp_open.run()

with open('../data/exp_open_yahoo.pkl', 'wb') as exp_file:
    pickle.dump(exp_open.result, exp_file)

print('-------STRATIFIED EVALUATION-------')


stra_eval_method = StratifiedEvaluation(data=ds_closed,
                                        n_strata=2,
                                        rating_threshold=4.0,
                                        verbose=True)

# run the experiment
exp_stra = STExperiment(eval_method=stra_eval_method,
                        models=get_models(variant='large', dims=dims),
                        metrics=get_metrics(variant='large'),
                        verbose=True)

exp_stra.run()

with open('../data/exp_stra_coats.pkl', 'wb') as exp_file:
    pickle.dump(exp_stra.result, exp_file)
