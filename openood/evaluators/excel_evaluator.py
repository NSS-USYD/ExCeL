import copy
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List

from openood.postprocessors import BasePostprocessor
from openood.utils import Config

from .ood_evaluator import OODEvaluator as BaseOODEvaluator
from .metrics import compute_all_metrics


class ExcelOODEvaluator(BaseOODEvaluator):
    def __init__(self, config: Config):
        super().__init__(config)
        self.id_pca_orig = None
        self.id_pca = None

    def eval_acc(self,
                 net: nn.Module,
                 data_loader: DataLoader,
                 postprocessor: BasePostprocessor = None,
                 epoch_idx: int = -1,
                 fsood: bool = False,
                 csid_data_loaders: DataLoader = None):
        if type(net) is dict:
            net['backbone'].eval()
        else:
            net.eval()

        template_loader = data_loader
        if hasattr(postprocessor, 'id_loader_dict') and 'train' in postprocessor.id_loader_dict:
            template_loader = postprocessor.id_loader_dict['train']

        self.id_pred, self.id_conf, self.id_gt, self.id_pca_orig = \
            postprocessor.inference(net, template_loader, pca_obj=None, pca_fit=1)

        metrics = {}
        metrics['acc'] = sum(self.id_pred == self.id_gt) / len(self.id_pred)
        metrics['epoch_idx'] = epoch_idx
        return metrics

    def hyperparam_search(self,
                          net: nn.Module,
                          id_data_loader,
                          ood_data_loader,
                          postprocessor: BasePostprocessor):
        print('Starting automatic parameter search...')
        aps_dict = {}
        max_auroc = 0
        hyperparam_names = []
        hyperparam_list = []

        for name in postprocessor.args_dict.keys():
            hyperparam_names.append(name)
        for name in hyperparam_names:
            hyperparam_list.append(postprocessor.args_dict[name])

        hyperparam_combination = self.recursive_generator(
            hyperparam_list, len(hyperparam_list))

        for hyperparam in hyperparam_combination:
            postprocessor.set_hyperparam(hyperparam)

            id_pca_copy = copy.deepcopy(self.id_pca_orig)
            _, _, _, pca_object = postprocessor.inference(
                net, id_data_loader, pca_obj=id_pca_copy, pca_fit=2)

            id_pred, id_conf, id_gt, _ = postprocessor.inference(
                net, id_data_loader, pca_obj=pca_object, pca_fit=0)
            ood_pred, ood_conf, ood_gt, _ = postprocessor.inference(
                net, ood_data_loader, pca_obj=pca_object, pca_fit=0)

            ood_gt = -1 * np.ones_like(ood_gt)
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])

            ood_metrics = compute_all_metrics(conf, label, pred)
            index = hyperparam_combination.index(hyperparam)
            aps_dict[index] = ood_metrics[1]
            print(f'Hyperparam:{hyperparam}, auroc:{aps_dict[index]}')

            if ood_metrics[1] > max_auroc:
                max_auroc = ood_metrics[1]
                self.id_pca = pca_object

        for key in aps_dict.keys():
            if aps_dict[key] == max_auroc:
                postprocessor.set_hyperparam(hyperparam_combination[key])

        print('Final hyperparam: {}'.format(postprocessor.get_hyperparam()))
        return max_auroc

    def eval_ood(self,
                 net: nn.Module,
                 id_data_loaders: Dict[str, DataLoader],
                 ood_data_loaders: Dict[str, Dict[str, DataLoader]],
                 postprocessor: BasePostprocessor,
                 fsood: bool = False):
        if type(net) is dict:
            for subnet in net.values():
                subnet.eval()
        else:
            net.eval()

        assert 'test' in id_data_loaders
        dataset_name = self.config.dataset.name

        if self.config.postprocessor.APS_mode:
            assert 'val' in id_data_loaders
            assert 'val' in ood_data_loaders
            self.hyperparam_search(net, id_data_loaders['val'],
                                   ood_data_loaders['val'], postprocessor)

        print(f'Performing inference on {dataset_name} dataset...', flush=True)
        id_pred, id_conf, id_gt, _ = postprocessor.inference(
            net, id_data_loaders['test'], pca_obj=self.id_pca, pca_fit=0)

        if self.config.recorder.save_scores:
            self._save_scores(id_pred, id_conf, id_gt, dataset_name)

        print(u'\u2500' * 70, flush=True)
        self._eval_ood(net, [id_pred, id_conf, id_gt],
                       ood_data_loaders, postprocessor, ood_split='nearood')

        print(u'\u2500' * 70, flush=True)
        self._eval_ood(net, [id_pred, id_conf, id_gt],
                       ood_data_loaders, postprocessor, ood_split='farood')

    def _eval_ood(self,
                  net: nn.Module,
                  id_list: List[np.ndarray],
                  ood_data_loaders: Dict[str, Dict[str, DataLoader]],
                  postprocessor: BasePostprocessor,
                  ood_split: str = 'nearood'):
        print(f'Processing {ood_split}...', flush=True)
        [id_pred, id_conf, id_gt] = id_list
        metrics_list = []

        for dataset_name, ood_dl in ood_data_loaders[ood_split].items():
            print(f'Performing inference on {dataset_name} dataset...',
                  flush=True)

            ood_pred, ood_conf, ood_gt, _ = postprocessor.inference(
                net, ood_dl, pca_obj=self.id_pca, pca_fit=0)

            ood_gt = -1 * np.ones_like(ood_gt)
            if self.config.recorder.save_scores:
                self._save_scores(ood_pred, ood_conf, ood_gt, dataset_name)

            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])

            print(f'Computing metrics on {dataset_name} dataset...')
            ood_metrics = compute_all_metrics(conf, label, pred)

            if self.config.recorder.save_csv:
                self._save_csv(ood_metrics, dataset_name=dataset_name)
            metrics_list.append(ood_metrics)

        print('Computing mean metrics...', flush=True)
        metrics_list = np.array(metrics_list)
        metrics_mean = np.mean(metrics_list, axis=0)
        if self.config.recorder.save_csv:
            self._save_csv(metrics_mean, dataset_name=ood_split)
