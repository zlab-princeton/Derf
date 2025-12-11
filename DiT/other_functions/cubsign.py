import torch
import torch.nn as nn
from timm.layers import LayerNorm2d


class Cubsign(nn.Module):
    def __init__(self, normalized_shape, elementwise_affine, alpha_init_value=0.5, shift_init_value=0.0):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.elementwise_affine = elementwise_affine
        self.alpha_init_value = alpha_init_value
        self.shift_init_value = shift_init_value

        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.shift = nn.Parameter(torch.ones(1) * shift_init_value)

    def forward(self, x):
        x_transformed = self.alpha * x + self.shift
        x_transformed = x_transformed**3 / (torch.abs(x_transformed)**3 + 1)
        if self.elementwise_affine:
            return x_transformed * self.weight + self.bias
        else:
            return x_transformed

    def extra_repr(self):
        return f"normalized_shape={self.normalized_shape}, elementwise_affine={self.elementwise_affine}, alpha_init_value={self.alpha_init_value}, shift_init_value={self.shift_init_value}"


def convert_ln_to_cubsign(module, alpha_init_value=0.5, shift_init_value=0.0):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = Cubsign(module.normalized_shape, module.elementwise_affine, alpha_init_value=alpha_init_value, shift_init_value=shift_init_value)
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_cubsign(child, alpha_init_value, shift_init_value))
    del module
    return module_output

