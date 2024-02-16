from typing import Dict

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


def write_metrics(metrics: Dict[str, float], epoch: int, prefix: str, writer: SummaryWriter):
    for metric, value in metrics.items():
        writer.add_scalar(f'{prefix}/{metric}', value, global_step = epoch)
    writer.flush()

def save_states(model: nn.Module, 
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler,
                save_path: str,
                **kwargs):
    
    save_dict = {'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'model_structure': model}
    for key, value in kwargs.items():
        save_dict[key] = value
    torch.save(save_dict, save_path)