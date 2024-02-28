import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999, warmup_steps: int = 0):
        self.model = model
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.last_step = 0
        self.shadow = {}
        self.backup = {}

    def _get_decay(self):
        if self.warmup_steps <= 0 or self.last_step > self.warmup_steps:
            return self.decay
        else:
            return min(max((self.last_step / self.warmup_steps) * self.decay, 0), self.decay)

    def register(self):
        """Register model parameters for EMA.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def to(self, device):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.shadow[name].to(device)

    def update(self):
        self.last_step += 1
        decay = self._get_decay()
        if decay == 0:
            return
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self) -> dict:
        return {
            'shadow': self.shadow,
            'backup': self.backup,
            'last_step': self.last_step,
            'warmup_steps': self.warmup_steps,
            'decay': self.decay
        }

    def load_state_dict(self, state_dict: dict):
        self.shadow = state_dict['shadow']
        self.backup = state_dict['backup']
        self.last_step = state_dict['last_step']
        self.warmup_steps = state_dict['warmup_steps']
        self.decay = state_dict['decay']