import torch

MAX_FLOW = 400

@torch.no_grad()
def compute_epe(flow_pred, flow_gt, valid, max_flow=MAX_FLOW):
    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt**2, dim=1).sqrt()
    valid = (valid >= 0.5) & (mag < max_flow)

    epe = torch.sum((flow_pred - flow_gt)**2, dim=1).sqrt()
    epe = torch.masked_fill(epe, ~valid, 0)
    
    return epe


@torch.no_grad()
def process_metrics(epe, info, **kwargs):
    epe = epe.flatten(1)
    metrics = {
            'epe': epe.mean(dim=1), 
            '1px': (epe < 1).float().mean(dim=1),
            '3px': (epe < 3).float().mean(dim=1),
            '5px': (epe < 5).float().mean(dim=1),
            'rel': info['rel_lowest'],
            'abs': info['abs_lowest'],
            }
   
    # dict: N_Metrics -> B // N_GPU
    return metrics


@torch.no_grad()
def merge_metrics(metrics):
    out = dict()

    for key, value in metrics.items():
        out[key] = value.mean().item()

    return out
