import torch
import torch.nn as nn


class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, elementwise_affine, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.elementwise_affine = elementwise_affine
        self.alpha_init_value = alpha_init_value
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        if self.elementwise_affine:
            return self.weight * torch.tanh(self.alpha * x) + self.bias
        else:
            return torch.tanh(self.alpha * x)

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, elementwise_affine={self.elementwise_affine}, alpha_init_value={self.alpha_init_value}"


def convert_ln_to_dyt(module, alpha_init_value=0.5):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = DynamicTanh(module.normalized_shape, module.elementwise_affine, alpha_init_value=alpha_init_value)
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_dyt(child, alpha_init_value))
    del module
    return module_output