# import mmcv
from copy import deepcopy
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
# from mmcls.apis import init_model

import openood.utils.comm as comm


from .resnet18_32x32 import ResNet18_32x32
from .resnet18_224x224 import ResNet18_224x224



def get_network(network_config):

    num_classes = network_config.num_classes

    if network_config.name == 'resnet18_32x32':
        net = ResNet18_32x32(num_classes=num_classes)

    elif network_config.name == 'resnet18_224x224':
        net = ResNet18_224x224(num_classes=num_classes)

    else:
        raise Exception('Unexpected Network Architecture!')

    if network_config.pretrained:
        if type(net) is dict:
            if isinstance(network_config.checkpoint, list):
                for subnet, checkpoint in zip(net.values(),
                                              network_config.checkpoint):
                    if checkpoint is not None:
                        if checkpoint != 'none':
                            subnet.load_state_dict(torch.load(checkpoint),
                                                   strict=False)
            elif isinstance(network_config.checkpoint, str):
                ckpt = torch.load(network_config.checkpoint)
                subnet_ckpts = {k: {} for k in net.keys()}
                for k, v in ckpt.items():
                    for subnet_name in net.keys():
                        if k.startwith(subnet_name):
                            subnet_ckpts[subnet_name][k.replace(
                                subnet_name + '.', '')] = v
                            break

                for subnet_name, subnet in net.items():
                    subnet.load_state_dict(subnet_ckpts[subnet_name])

        elif network_config.name == 'bit' and not network_config.normal_load:
            net.load_from(np.load(network_config.checkpoint))
        elif network_config.name == 'vit':
            pass
        # else:
        #     try:
        #         net.load_state_dict(torch.load(network_config.checkpoint),
        #                             strict=False)
        #     except RuntimeError:
        #         # sometimes fc should not be loaded
        #         loaded_pth = torch.load(network_config.checkpoint)
        #         loaded_pth.pop('fc.weight')
        #         loaded_pth.pop('fc.bias')
        #         net.load_state_dict(loaded_pth, strict=False)
        # print('Model Loading {} Completed!'.format(network_config.name))

        else:
            loaded_pth = torch.load(network_config.checkpoint)
            print('Loading checkpoint from:', network_config.checkpoint,
                  flush=True)
            print('Checkpoint type:', type(loaded_pth), flush=True)

            if isinstance(loaded_pth, dict) and 'state_dict' in loaded_pth:
                loaded_pth = loaded_pth['state_dict']
            elif isinstance(loaded_pth, dict) and 'model' in loaded_pth:
                loaded_pth = loaded_pth['model']
            elif isinstance(loaded_pth, dict) and 'net' in loaded_pth:
                loaded_pth = loaded_pth['net']

            try:
                load_msg = net.load_state_dict(loaded_pth, strict=False)
                print(load_msg, flush=True)
            except RuntimeError:
                print('Checkpoint fc layer does not match, dropping fc.*',
                      flush=True)
                loaded_pth.pop('fc.weight', None)
                loaded_pth.pop('fc.bias', None)
                load_msg = net.load_state_dict(loaded_pth, strict=False)
                print(load_msg, flush=True)

        print('Model Loading {} Completed!'.format(network_config.name))


    if network_config.num_gpus > 1:
        if type(net) is dict:
            for key, subnet in zip(net.keys(), net.values()):
                net[key] = torch.nn.parallel.DistributedDataParallel(
                    subnet.cuda(),
                    device_ids=[comm.get_local_rank()],
                    broadcast_buffers=True)
        else:
            net = torch.nn.parallel.DistributedDataParallel(
                net.cuda(),
                device_ids=[comm.get_local_rank()],
                broadcast_buffers=True)

    if network_config.num_gpus > 0:
        if type(net) is dict:
            for subnet in net.values():
                subnet.cuda()
        else:
            net.cuda()

    cudnn.benchmark = True
    return net
