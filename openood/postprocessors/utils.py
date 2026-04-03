from openood.utils import Config

from .base_postprocessor import BasePostprocessor
from .excel_postprocessor import ExcelPostprocessor


def get_postprocessor(config: Config):
    postprocessors = {
        'msp': BasePostprocessor,
        'excel': ExcelPostprocessor,
    }

    return postprocessors[config.postprocessor.name](config)
