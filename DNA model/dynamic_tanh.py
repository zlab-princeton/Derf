import torch
import torch.nn as nn


class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        return self.weight * torch.tanh(self.alpha * x) + self.bias

    def extra_repr(self):
        return f'normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}'

def _is_rmsnorm(m: nn.Module) -> bool:
    return "rmsnorm" in m.__class__.__name__.lower()


def convert_ln_to_dyt(module):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = DynamicTanh(module.normalized_shape)
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_dyt(child))
    del module
    return module_output

def convert_rmsnorm_to_dyt(module):
    module_output = module
    if _is_rmsnorm(module):
        module_output = DynamicTanh(module.weight.shape)
    for name, child in module.named_children():
        module_output.add_module(name, convert_rmsnorm_to_dyt(child))
    del module
    return module_output