import torch
import torch.nn as nn
import torch.nn.functional as F 

def create_criterion(config, logger):
    loss_type = config['training_opt']['loss_type']
    loss_params = config['training_opt']['loss_params']
    if loss_type == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss(**loss_params)
    else:
        raise NotImplementedError
    return criterion