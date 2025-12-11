import torch
import torch.nn as nn


class Dynamic_erf(nn.Module):
    def __init__(self, normalized_shape, alpha_init_value=0.5, shift_init_value=0.0):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.shift_init_value = shift_init_value

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.shift = nn.Parameter(torch.ones(1) * shift_init_value)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        return self.weight * torch.erf(self.alpha * x + self.shift) + self.bias

    def extra_repr(self):
        return f'normalized_shape={self.normalized_shape}, alpha_init_value={self.alpha_init_value}'

def _is_rmsnorm(m: nn.Module) -> bool:
    return "rmsnorm" in m.__class__.__name__.lower()

def convert_ln_to_derf(module):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = Dynamic_erf(module.normalized_shape)
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_derf(child))
    del module
    return module_output

def convert_rmsnorm_to_derf(module):
    module_output = module
    if _is_rmsnorm(module):
        module_output = Dynamic_erf(module.weight.shape)
    for name, child in module.named_children():
        module_output.add_module(name, convert_rmsnorm_to_derf(child))
    del module
    return module_output
