from cornac.experiment.experiment import Experiment
from cornac.eval_methods.cross_validation import CrossValidation
from cornac.experiment.result import ExperimentResult
from cornac.experiment.result import CVExperimentResult

from eval_methods.stratified_evaluation import StratifiedEvaluation


class STExperiment(Experiment):

    def _create_result(self):

        if isinstance(self.eval_method, CrossValidation) or isinstance(self.eval_method, StratifiedEvaluation):
            self.result = CVExperimentResult()
        else:
            self.result = ExperimentResult()
            if self.show_validation and self.eval_method.val_set is not None:
                self.val_result = ExperimentResult()
