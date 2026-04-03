from openood.utils import Config

from .base_evaluator import BaseEvaluator
from .ood_evaluator import OODEvaluator
from .excel_evaluator import ExcelOODEvaluator



def get_evaluator(config: Config):
    evaluators = {
        'base': BaseEvaluator,
        'ood': OODEvaluator,
        'excel_ood': ExcelOODEvaluator

    }
    return evaluators[config.evaluator.name](config)
