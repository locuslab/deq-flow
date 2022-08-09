import types

import torch
from torch import nn

from torch.nn import functional as F
from torch.nn.parameter import Parameter


def _norm(p, dim):
    """Computes the norm over all dimensions except dim"""
    if dim is None:
        return p.norm()
    elif dim == 0:
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        return p.contiguous().view(p.size(0), -1).norm(dim=1).view(*output_size)
    elif dim == p.dim() - 1:
        output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
        return p.contiguous().view(-1, p.size(-1)).norm(dim=0).view(*output_size)
    else:
        return _norm(p.transpose(0, dim), 0).transpose(0, dim)


def compute_weight(module, name, dim):
    g = getattr(module, name + '_g')
    v = getattr(module, name + '_v')
    return v * (g / _norm(v, dim))

    
def apply_atom_wn(module, names, dims):
    if type(names) is str:
        names = [names]

    if type(dims) is int:
        dims = [dims]
    
    assert len(names) == len(dims)

    for name, dim in zip(names, dims):
        weight = getattr(module, name)

        # remove w from parameter list
        del module._parameters[name]

        # add g and v as new parameters and express w as g/||v|| * v
        module.register_parameter(name + '_g', Parameter(_norm(weight, dim).data))
        module.register_parameter(name + '_v', Parameter(weight.data))
        setattr(module, name, compute_weight(module, name, dim))
    
    module._wn_names = names
    module._wn_dims = dims


def reset_atom_wn(module):
    # Typically, every time the module is called we need to recompute the weight. However,
    # in the case of DEQ, the same weight is shared across layers, and we can save
    # a lot of intermediate memory by just recomputing once (at the beginning of first call).

    for name, dim in zip(module._wn_names, module._wn_dims):
        setattr(module, name, compute_weight(module, name, dim))


def remove_atom_wn(module):
    for name, dim in zip(module._wn_names, module._wn_dims):
        weight = compute_weight(module, name, dim)
        delattr(module, name)
        del module._parameters[name + '_g']
        del module._parameters[name + '_v']
        module.register_parameter(name, Parameter(weight.data))
    
    del module._wn_names
    del module._wn_dims


target_modules = {
        nn.Linear: ('weight', 0), 
        nn.Conv1d: ('weight', 0), 
        nn.Conv2d: ('weight', 0), 
        nn.Conv3d: ('weight', 0)
        }


def register_wn_module(module_class, names='weight', dims=0):
    '''
    Register your self-defined module class for ``nested_weight_norm''.
    This module class will be automatically indexed for WN.

    Args: 
        module_class (type): module class to be indexed for weight norm (WN).
        names (string): attribute name of ``module_class'' for WN to be applied.
        dims (int, optional): dimension over which to compute the norm
 
    Returns:
        None
    '''
    target_modules[module_class] = (names, dims)


def _is_skip_name(name, filter_out):
    for skip_name in filter_out:
        if name.startswith(skip_name):
            return True
    
    return False


def apply_weight_norm(model, filter_out=None):
    if type(filter_out) is str:
        filter_out = [filter_out]

    for name, module in model.named_modules():
        if filter_out and _is_skip_name(name, filter_out):
            continue 

        class_type = type(module)
        if class_type in target_modules:
            apply_atom_wn(module, *target_modules[class_type])


def reset_weight_norm(model):
    for module in model.modules():
        if hasattr(module, '_wn_names'):
            reset_atom_wn(module)


def remove_weight_norm(model):
    for module in model.modules():
        if hasattr(module, '_wn_names'):
            remove_atom_wn(module)


if __name__ == '__main__':
    z = torch.randn(8, 128, 32, 32)

    net = nn.Conv2d(128, 256, 3, padding=1)
    z_orig = net(z)

    apply_weight_norm(net)
    z_wn = net(z)

    reset_weight_norm(net)
    z_wn_reset = net(z)
    
    remove_weight_norm(net)
    z_back = net(z)
    
    print((z_orig - z_wn).abs().mean().item())
    print((z_orig - z_wn_reset).abs().mean().item())
    print((z_orig - z_back).abs().mean().item())
    
    net = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(256, 128, 3, padding=1)
            )
    z_orig = net(z)
    
    apply_weight_norm(net)
    z_wn = net(z)
    
    reset_weight_norm(net)
    z_wn_reset = net(z)

    remove_weight_norm(net)
    z_back = net(z)

    print((z_orig - z_wn).abs().mean().item())
    print((z_orig - z_wn_reset).abs().mean().item())
    print((z_orig - z_back).abs().mean().item())

