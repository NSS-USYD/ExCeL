from openood.utils import Config

from .test_ood_pipeline import TestOODPipeline
from .train_pipeline import TrainPipeline


def get_pipeline(config: Config):
    pipelines = {
        'train': TrainPipeline,
        'test_ood': TestOODPipeline,
    }

    return pipelines[config.pipeline.name](config)
