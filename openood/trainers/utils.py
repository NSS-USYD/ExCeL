from torch.utils.data import DataLoader

from openood.utils import Config

from .base_trainer import BaseTrainer


def get_trainer(net, train_loader: DataLoader, val_loader: DataLoader,
                config: Config):
    trainers = {
        'base': BaseTrainer,
    }
    return trainers[config.trainer.name](net, train_loader, config)
