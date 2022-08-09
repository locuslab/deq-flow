import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from termcolor import colored 

from .solvers import get_solver
from .norm import reset_weight_norm
from .grad import make_pair, backward_factory
from .jacobian import power_method


class DEQBase(nn.Module):
    def __init__(self, args):
        super(DEQBase, self).__init__()
        
        self.args = args
        self.f_solver = get_solver(args.f_solver)
        self.b_solver = get_solver(args.b_solver)
        
        self.f_thres = args.f_thres
        self.b_thres = args.b_thres
        
        self.f_eps = args.f_eps
        self.b_eps = args.b_eps
        
        self.f_stop_mode = args.f_stop_mode
        self.b_stop_mode = args.b_stop_mode

        self.eval_f_thres = args.eval_f_thres if args.eval_f_thres > 0 else int(self.f_thres * args.eval_factor) 

        self.hook = None

    def _log_convergence(self, info, name='FORWARD', color='yellow'):
        state = 'TRAIN' if self.training else 'VALID'
        alt_mode = 'rel' if self.f_stop_mode == 'abs' else 'abs'

        rel_lowest, abs_lowest = info['rel_lowest'].mean().item(), info['abs_lowest'].mean().item()
        nstep = info['nstep']

        show_str = f'{state} | {name} | rel: {rel_lowest}; abs: {abs_lowest}; nstep: {nstep}'
        print(colored(show_str, color))

    def _sradius(self, deq_func, z_star):
        with torch.enable_grad():
            new_z_star = deq_func(z_star.requires_grad_())
        _, sradius = power_method(new_z_star, z_star, n_iters=75)

        return sradius

    def _solve_fixed_point(
            self, deq_func, z_init, 
            log=False, f_thres=None, 
            **kwargs
            ):
        raise NotImplementedError
    
    def forward(
            self, deq_func, z_init, 
            log=False, sradius_mode=False, writer=None,
            **kwargs
            ):
        raise NotImplementedError


class DEQIndexing(DEQBase):
    def __init__(self, args):
        super(DEQIndexing, self).__init__(args)
        
        # Define gradient functions through the backward factory
        if args.n_losses > 1:
            n_losses = min(args.f_thres, args.n_losses)
            delta = int(args.f_thres // n_losses)
            self.indexing = [(k+1)*delta for k in range(n_losses)]
        else:
            self.indexing = [*args.indexing, args.f_thres]
        
        # By default, we use the same phantom grad for all corrections.
        # You can also set different grad steps a, b, and c for different terms by ``args.phantom_grad a b c ...''.
        indexing_pg = make_pair(self.indexing, args.phantom_grad)
        produce_grad = [
                backward_factory(grad_type=pg, tau=args.tau, sup_all=args.sup_all) for pg in indexing_pg
                ]
        if args.ift:
            # Enabling args.ift will replace the last gradient function by IFT.
            produce_grad[-1] = backward_factory(
                grad_type='ift', safe_ift=args.safe_ift, b_solver=self.b_solver,
                b_solver_kwargs=dict(threshold=args.b_thres, eps=args.b_eps, stop_mode=args.b_stop_mode)
                )

        self.produce_grad = produce_grad
    
    def _solve_fixed_point(
               self, deq_func, z_init, 
               log=False, f_thres=None, 
               **kwargs
               ):
        if f_thres is None: f_thres = self.f_thres
        indexing = self.indexing if self.training else None

        with torch.no_grad():
            z_star, trajectory, info = self.f_solver(
                    deq_func, x0=z_init, threshold=f_thres,     # To reuse previous coarse fixed points
                    eps=self.f_eps, stop_mode=self.f_stop_mode, indexing=indexing
                    )

        if log: self._log_convergence(info, name="FORWARD", color="yellow")          
        
        return z_star, trajectory, info

    def forward(
            self, deq_func, z_init,
            log=False, sradius_mode=False, writer=None,
            **kwargs
            ):
        if self.training:
            _, trajectory, info = self._solve_fixed_point(deq_func, z_init, log=log, *kwargs)
            
            z_out = []
            for z_pred, produce_grad in zip(trajectory, self.produce_grad):
                z_out += produce_grad(self, deq_func, z_pred)  # See lib/grad.py for the backward pass implementations
            
            z_out = [deq_func.vec2list(each) for each in z_out]
        else:
            # During inference, we directly solve for fixed point
            z_star, _, info = self._solve_fixed_point(deq_func, z_init, log=log, f_thres=self.eval_f_thres)
            
            sradius = self._sradius(deq_func, z_star) if sradius_mode else torch.zeros(1, device=z_star.device)
            info['sradius'] = sradius

            z_out = [deq_func.vec2list(z_star)]

        return z_out, info


class DEQSliced(DEQBase):
    def __init__(self, args):
        super(DEQSliced, self).__init__(args)
        
        # Define gradient functions through the backward factory
        if args.n_losses > 1:
            self.indexing = [int(args.f_thres // args.n_losses) for _ in range(args.n_losses)]
        else:
            self.indexing = np.diff([0, *args.indexing, args.f_thres]).tolist()
        
        # By default, we use the same phantom grad for all corrections.
        # You can also set different grad steps a, b, and c for different terms by ``args.phantom_grad a b c ...''.
        indexing_pg = make_pair(self.indexing, args.phantom_grad)
        produce_grad = [
                backward_factory(grad_type=pg, tau=args.tau, sup_all=args.sup_all) for pg in indexing_pg
                ]
        if args.ift:
            # Enabling args.ift will replace the last gradient function by IFT.
            produce_grad[-1] = backward_factory(
                grad_type='ift', safe_ift=args.safe_ift, b_solver=self.b_solver,
                b_solver_kwargs=dict(threshold=args.b_thres, eps=args.b_eps, stop_mode=args.b_stop_mode)
                )

        self.produce_grad = produce_grad
    
    def _solve_fixed_point(
            self, deq_func, z_init, 
            log=False, f_thres=None, 
            **kwargs
            ):
        with torch.no_grad():
            z_star, _, info = self.f_solver(
                    deq_func, x0=z_init, threshold=f_thres, # To reuse previous coarse fixed points
                    eps=self.f_eps, stop_mode=self.f_stop_mode
                    )

        if log: self._log_convergence(info, name="FORWARD", color="yellow")          
        
        return z_star, info

    def forward(
            self, deq_func, z_star, 
            log=False, sradius_mode=False, writer=None,
            **kwargs
            ):
        if self.training:
            z_out = []
            for f_thres, produce_grad in zip(self.indexing, self.produce_grad):
                z_star, info = self._solve_fixed_point(deq_func, z_star, f_thres=f_thres, log=log)
                z_out += produce_grad(self, deq_func, z_star, writer=writer)    # See lib/grad.py for implementations
                z_star = z_out[-1]                                              # Add the gradient chain to the solver.

            z_out = [deq_func.vec2list(each) for each in z_out]
        else:
            # During inference, we directly solve for fixed point
            z_star, info = self._solve_fixed_point(deq_func, z_star, f_thres=self.eval_f_thres, log=log)
            
            sradius = self._sradius(deq_func, z_star) if sradius_mode else torch.zeros(1, device=z_star.device)
            info['sradius'] = sradius

            z_out = [deq_func.vec2list(z_star)]

        return z_out, info


def get_deq(args):
    if args.indexing_core:
        return DEQIndexing
    else:
        return DEQSliced
