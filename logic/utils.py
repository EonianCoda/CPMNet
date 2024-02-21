from typing import Dict
import logging
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

def write_metrics(metrics: Dict[str, float], epoch: int, prefix: str, writer: SummaryWriter):
    for metric, value in metrics.items():
        writer.add_scalar(f'{prefix}/{metric}', value, global_step = epoch)
    writer.flush()

def save_states(save_path: str, model: nn.Module, optimizer: torch.optim.Optimizer = None, scheduler: torch.optim.lr_scheduler = None, **kwargs):
    save_dict = {'model_state_dict': model.state_dict(), 'model_structure': model}
    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler_state_dict'] = scheduler.state_dict()
    for key, value in kwargs.items():
        save_dict[key] = value
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(save_dict, save_path)
    
def load_states(load_path: str, device: torch.device, model: nn.Module, optimizer: torch.optim.Optimizer = None, scheduler: torch.optim.lr_scheduler = None, **kwargs):
    checkpoint = torch.load(load_path, map_location=device)
    
    if 'state_dict' not in checkpoint and 'model_state_dict' not in checkpoint:
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    for key, value in kwargs.items():
        if key not in checkpoint:
            logger.warning(f'Key {key} not found in checkpoint')
        kwargs[key].load_state_dict(checkpoint[key])
        
def load_model(load_path: str):
    checkpoint = torch.load(load_path)
    
    # Build model
    if 'model_structure' in checkpoint:
        model = checkpoint['model_structure']
    else:
        from networks.ResNet_3D_CPM import Resnet18
        model = Resnet18()
        
    # Load state dict
    if 'state_dict' not in checkpoint and 'model_state_dict' not in checkpoint:
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        
    return model