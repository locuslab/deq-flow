# Modified based on the DEQ repo.

import torch
from torch import nn
import torch.nn.functional as functional
from torch.autograd import Function
import numpy as np 
import pickle
import sys
import os
from scipy.optimize import root
import time
from termcolor import colored


def batch_masked_mixing(mask, mask_var, orig_var):
    '''
    First align the axes of mask to mask_var.
    Then mix mask_var and orig_var through the aligned mask.

    Args:
        mask: a tensor of shape (B,)
        mask_var: a tensor of shape (B, ...) for the mask to select
        orig_var: a tensor of shape (B, ...) for the reversed mask to select
    '''

    if torch.is_tensor(mask_var):
        axes_to_align = len(mask_var.shape) - 1
    elif torch.is_tensor(orig_var):
        axes_to_align = len(orig_var.shape) - 1
    else:
        raise ValueError('Either mask_var or orig_var should be a Pytorch tensor!')

    aligned_mask = mask.view(mask.shape[0], *[1 for _ in range(axes_to_align)])

    return aligned_mask * mask_var + ~aligned_mask * orig_var


def init_solver_stats(x0, init_loss=1e8):
    trace_dict = {
            'abs': [torch.tensor(init_loss, device=x0.device).repeat(x0.shape[0])],
            'rel': [torch.tensor(init_loss, device=x0.device).repeat(x0.shape[0])]
            }
    lowest_dict = {
            'abs': torch.tensor(init_loss, device=x0.device).repeat(x0.shape[0]),
            'rel': torch.tensor(init_loss, device=x0.device).repeat(x0.shape[0])
            }
    lowest_step_dict = {
            'abs': torch.tensor(0, device=x0.device).repeat(x0.shape[0]),
            'rel': torch.tensor(0, device=x0.device).repeat(x0.shape[0]),
            }

    return trace_dict, lowest_dict, lowest_step_dict


def _safe_norm(v):
    if not torch.isfinite(v).all():
        return np.inf
    return torch.norm(v)


def scalar_search_armijo(phi, phi0, derphi0, c1=1e-4, alpha0=1, amin=0):
    ite = 0
    phi_a0 = phi(alpha0)    # First do an update with step size 1
    if phi_a0 <= phi0 + c1*alpha0*derphi0:
        return alpha0, phi_a0, ite

    # Otherwise, compute the minimizer of a quadratic interpolant
    alpha1 = -(derphi0) * alpha0**2 / 2.0 / (phi_a0 - phi0 - derphi0 * alpha0)
    phi_a1 = phi(alpha1)

    # Otherwise loop with cubic interpolation until we find an alpha which
    # satisfies the first Wolfe condition (since we are backtracking, we will
    # assume that the value of alpha is not too small and satisfies the second
    # condition.
    while alpha1 > amin:       # we are assuming alpha>0 is a descent direction
        factor = alpha0**2 * alpha1**2 * (alpha1-alpha0)
        a = alpha0**2 * (phi_a1 - phi0 - derphi0*alpha1) - \
            alpha1**2 * (phi_a0 - phi0 - derphi0*alpha0)
        a = a / factor
        b = -alpha0**3 * (phi_a1 - phi0 - derphi0*alpha1) + \
            alpha1**3 * (phi_a0 - phi0 - derphi0*alpha0)
        b = b / factor

        alpha2 = (-b + torch.sqrt(torch.abs(b**2 - 3 * a * derphi0))) / (3.0*a)
        phi_a2 = phi(alpha2)
        ite += 1

        if (phi_a2 <= phi0 + c1*alpha2*derphi0):
            return alpha2, phi_a2, ite

        if (alpha1 - alpha2) > alpha1 / 2.0 or (1 - alpha2/alpha1) < 0.96:
            alpha2 = alpha1 / 2.0

        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        phi_a1 = phi_a2

    # Failed to find a suitable step length
    return None, phi_a1, ite


def line_search(update, x0, g0, g, nstep=0, on=True):
    """
    `update` is the propsoed direction of update.

    Code adapted from scipy.
    """
    tmp_s = [0]
    tmp_g0 = [g0]
    tmp_phi = [torch.norm(g0)**2]
    s_norm = torch.norm(x0) / torch.norm(update)

    def phi(s, store=True):
        if s == tmp_s[0]:
            return tmp_phi[0]    # If the step size is so small... just return something
        x_est = x0 + s * update
        g0_new = g(x_est)
        phi_new = _safe_norm(g0_new)**2
        if store:
            tmp_s[0] = s
            tmp_g0[0] = g0_new
            tmp_phi[0] = phi_new
        return phi_new
    
    if on:
        s, phi1, ite = scalar_search_armijo(phi, tmp_phi[0], -tmp_phi[0], amin=1e-2)
    if (not on) or s is None:
        s = 1.0
        ite = 0

    x_est = x0 + s * update
    if s == tmp_s[0]:
        g0_new = tmp_g0[0]
    else:
        g0_new = g(x_est)
    return x_est, g0_new, x_est - x0, g0_new - g0, ite


def rmatvec(part_Us, part_VTs, x):
    # Compute x^T(-I + UV^T)
    # x: (N, D)
    # part_Us: (N, D, L_thres)
    # part_VTs: (N, L_thres, D)
    if part_Us.nelement() == 0:
        return -x
    xTU = torch.einsum('bd, bdl -> bl', x, part_Us)             # (B, L_thres)
    return -x + torch.einsum('bl, bld -> bd', xTU, part_VTs)    # (B, D)


def matvec(part_Us, part_VTs, x):
    # Compute (-I + UV^T)x
    # x: (B, D)
    # part_Us: (B, D, L_thres)
    # part_VTs: (B, L_thres, D)
    if part_Us.nelement() == 0:
        return -x
    VTx = torch.einsum('bld, bd -> bl', part_VTs, x)            # (B, L_thres)
    return -x + torch.einsum('bdl, bl -> bd', part_Us, VTx)     # (B, D)


def broyden(func, x0, 
        threshold=50, eps=1e-3, stop_mode="rel", indexing=None,
        LBFGS_thres=None, ls=False, **kwargs):
    bsz, dim = x0.flatten(start_dim=1).shape
    g = lambda y: func(y.view_as(x0)).view_as(y) - y
    
    alternative_mode = 'rel' if stop_mode == 'abs' else 'abs'
    LBFGS_thres = threshold if LBFGS_thres is None else LBFGS_thres

    x_est = x0.flatten(start_dim=1)     # (B, D)
    gx = g(x_est)                       # (B, D)
    nstep = 0
    tnstep = 0

    # For fast calculation of inv_jacobian (approximately)
    Us = torch.zeros(bsz, dim, LBFGS_thres, dtype=x0.dtype, device=x0.device)   # One can also use an L-BFGS scheme to further reduce memory
    VTs = torch.zeros(bsz, LBFGS_thres, dim, dtype=x0.dtype, device=x0.device)
    update = -matvec(Us[:,:,:nstep], VTs[:,:nstep], gx)                         # Formally should be -torch.matmul(inv_jacobian (-I), gx)
    prot_break = False
    
    new_objective = 1e8
        
    trace_dict, lowest_dict, lowest_step_dict = init_solver_stats(x0)
    nstep, lowest_xest = 0, x_est
    
    indexing_list = []

    while nstep < threshold:
        x_est, gx, delta_x, delta_gx, ite = line_search(update, x_est, gx, g, nstep=nstep, on=ls)
        nstep += 1
        tnstep += (ite+1)
        abs_diff = gx.norm(dim=1)
        rel_diff = abs_diff / ((gx + x_est).norm(dim=1) + 1e-8)

        diff_dict = {'abs': abs_diff,
                     'rel': rel_diff}
        trace_dict['abs'].append(abs_diff)
        trace_dict['rel'].append(rel_diff)
        
        for mode in ['rel', 'abs']:
            is_lowest = diff_dict[mode] < lowest_dict[mode]
            if mode == stop_mode:
                lowest_xest = batch_masked_mixing(is_lowest, x_est, lowest_xest)
                lowest_xest = lowest_xest.view_as(x0).clone().detach() 
            lowest_dict[mode] = batch_masked_mixing(is_lowest, diff_dict[mode], lowest_dict[mode])
            lowest_step_dict[mode] = batch_masked_mixing(is_lowest, nstep, lowest_step_dict[mode])

        if indexing and (nstep+1) in indexing:
            indexing_list.append(lowest_xest)

        new_objective = diff_dict[stop_mode].max()
        if new_objective < eps: break
        
        if nstep > 30:
            progress = torch.stack(trace_dict[stop_mode][-30:]).max(dim=1)[0] \
                    / torch.stack(trace_dict[stop_mode][-30:]).min(dim=1)[0]
            if new_objective < 3*eps and progress.max() < 1.3:
                # if there's hardly been any progress in the last 30 steps
                break

        part_Us, part_VTs = Us[:,:,:nstep-1], VTs[:,:nstep-1]
        vT = rmatvec(part_Us, part_VTs, delta_x)
        u = (delta_x - matvec(part_Us, part_VTs, delta_gx)) / torch.einsum('bd,bd->b', vT, delta_gx)[:,None]
        vT[vT != vT] = 0
        u[u != u] = 0
        VTs[:,(nstep-1) % LBFGS_thres] = vT
        Us[:,:,(nstep-1) % LBFGS_thres] = u
        update = -matvec(Us[:,:,:nstep], VTs[:,:nstep], gx)
    
    # Fill everything up to the threshold length
    for _ in range(threshold+1-len(trace_dict[stop_mode])):
        trace_dict[stop_mode].append(lowest_dict[stop_mode])
        trace_dict[alternative_mode].append(lowest_dict[alternative_mode])
    
    # at least return the lowest value when enabling  ``indexing''
    if indexing and not indexing_list:
        indexing_list.append(lowest_xest)
 
    info = {
            'abs_lowest': lowest_dict['abs'],
            'rel_lowest': lowest_dict['rel'],
            'abs_trace': trace_dict['abs'],
            'rel_trace': trace_dict['rel'],
            'nstep': lowest_step_dict[stop_mode], 
            }
    return lowest_xest, indexing_list, info


def anderson(func, x0, 
        threshold=50, eps=1e-3, stop_mode='rel', indexing=None,
        m=6, lam=1e-4, beta=1.0, **kwargs):
    """ Anderson acceleration for fixed point iteration. """
    bsz, dim = x0.flatten(start_dim=1).shape
    f = lambda x: func(x.view_as(x0)).view_as(x) 

    alternative_mode = 'rel' if stop_mode == 'abs' else 'abs'
    X = torch.zeros(bsz, m, dim, dtype=x0.dtype, device=x0.device)
    F = torch.zeros(bsz, m, dim, dtype=x0.dtype, device=x0.device)

    x0_flat = x0.flatten(start_dim=1)
    X[:,0], F[:,0] = x0_flat, f(x0_flat)
    X[:,1], F[:,1] = F[:,0], f(F[:,0])
    
    H = torch.zeros(bsz, m+1, m+1, dtype=x0.dtype, device=x0.device)
    H[:,0,1:] = H[:,1:,0] = 1
    y = torch.zeros(bsz, m+1, 1, dtype=x0.dtype, device=x0.device)
    y[:,0] = 1

    trace_dict, lowest_dict, lowest_step_dict = init_solver_stats(x0)
    lowest_xest = x0

    indexing_list = []

    for k in range(2, threshold):
        n = min(k, m)
        G = F[:,:n]-X[:,:n]
        H[:,1:n+1,1:n+1] = torch.bmm(G,G.transpose(1,2)) + lam*torch.eye(n, dtype=x0.dtype, device=x0.device)[None]
        alpha = torch.linalg.solve(H[:,:n+1,:n+1], y[:,:n+1])[:, 1:n+1, 0]   # (bsz x n)

        X[:,k%m] = beta * (alpha[:,None] @ F[:,:n])[:,0] + (1-beta)*(alpha[:,None] @ X[:,:n])[:,0]
        F[:,k%m] = f(X[:,k%m])
        gx = F[:,k%m] - X[:,k%m]
        abs_diff = gx.norm(dim=1)
        rel_diff = abs_diff / (F[:,k%m].norm(dim=1) + 1e-8)

        diff_dict = {'abs': abs_diff,
                     'rel': rel_diff}
        trace_dict['abs'].append(abs_diff)
        trace_dict['rel'].append(rel_diff)
        
        for mode in ['rel', 'abs']:
            is_lowest = diff_dict[mode] < lowest_dict[mode]
            if mode == stop_mode:
                lowest_xest = batch_masked_mixing(is_lowest, F[:,k%m], lowest_xest)
                lowest_xest = lowest_xest.view_as(x0).clone().detach() 
            lowest_dict[mode] = batch_masked_mixing(is_lowest, diff_dict[mode], lowest_dict[mode])
            lowest_step_dict[mode] = batch_masked_mixing(is_lowest, k+1, lowest_step_dict[mode])

        if indexing and (k+1) in indexing:
            indexing_list.append(lowest_xest)

        if trace_dict[stop_mode][-1].max() < eps:
            for _ in range(threshold-1-k):
                trace_dict[stop_mode].append(lowest_dict[stop_mode])
                trace_dict[alternative_mode].append(lowest_dict[alternative_mode])
            break
    
    # at least return the lowest value when enabling  ``indexing''
    if indexing and not indexing_list:
        indexing_list.append(lowest_xest)

    X = F = None
 
    info = {
            'abs_lowest': lowest_dict['abs'],
            'rel_lowest': lowest_dict['rel'],
            'abs_trace': trace_dict['abs'],
            'rel_trace': trace_dict['rel'],
            'nstep': lowest_step_dict[stop_mode], 
            }
    return lowest_xest, indexing_list, info


def naive_solver(f, x0, 
        threshold=50, eps=1e-3, stop_mode='rel', indexing=None, 
        return_final=True, **kwargs):
    """ Naive Unrolling for fixed point iteration. """
    alternative_mode = 'rel' if stop_mode == 'abs' else 'abs'
    
    trace_dict, lowest_dict, lowest_step_dict = init_solver_stats(x0)
    lowest_xest = x0

    indexing_list = []
    
    fx = x = x0
    for k in range(threshold):
        x = fx
        fx = f(x)
        gx = fx - x
        abs_diff = gx.flatten(start_dim=1).norm(dim=1)
        rel_diff = abs_diff / (fx.flatten(start_dim=1).norm(dim=1) + 1e-8)

        diff_dict = {'abs': abs_diff,
                     'rel': rel_diff}
        trace_dict['abs'].append(abs_diff)
        trace_dict['rel'].append(rel_diff)
        
        for mode in ['rel', 'abs']:
            is_lowest = diff_dict[mode] < lowest_dict[mode] + return_final
            if return_final and mode == stop_mode:
                lowest_xest = batch_masked_mixing(is_lowest, fx, lowest_xest)
                lowest_xest = lowest_xest.view_as(x0).clone().detach() 
            lowest_dict[mode] = batch_masked_mixing(is_lowest, diff_dict[mode], lowest_dict[mode])
            lowest_step_dict[mode] = batch_masked_mixing(is_lowest, k+1, lowest_step_dict[mode])

        if indexing and (k+1) in indexing:
            indexing_list.append(lowest_xest)

        if trace_dict[stop_mode][-1].max() < eps:
            for _ in range(threshold-1-k):
                trace_dict[stop_mode].append(lowest_dict[stop_mode])
                trace_dict[alternative_mode].append(lowest_dict[alternative_mode])
            break
    
    # at least return the lowest value when enabling  ``indexing''
    if indexing and not indexing_list:
        indexing_list.append(lowest_xest)

    info = {
            'abs_lowest': lowest_dict['abs'],
            'rel_lowest': lowest_dict['rel'],
            'abs_trace': trace_dict['abs'],
            'rel_trace': trace_dict['rel'],
            'nstep': lowest_step_dict[stop_mode], 
            }
    return lowest_xest, indexing_list, info


solvers = {
        'anderson': anderson,
        'broyden': broyden,
        'naive_solver': naive_solver,
        }


def get_solver(key):
    assert key in solvers

    return solvers[key]
