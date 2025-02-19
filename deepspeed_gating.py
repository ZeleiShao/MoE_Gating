import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from typing import Tuple
import time
# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.

def benchmark(f, warmup=5, iter=20):
    for i in range(warmup + iter):
        f()
        # We do not synchronize here in order to hide the kernel launch overhead during benchmarkining as this will also
        # happen during realistic model inference as many launches are submitted to the kernel queue.
        if i == warmup - 1:
            #torch.cuda.synchronize()
            tick = time.time()
    #torch.cuda.synchronize()
    res = (time.time() - tick) / iter
    # Make sure there is enough to "cool down" the GPU in between benchmarks to avoid throttling for later runs when
    # we execute many benchmarks consecutively
    time.sleep(1.)
    return res

def einsum(rule, a, b):
    if USE_EINSUM:
        return torch.einsum(rule, a, b)
    elif rule == 's,se->se':
        return a.reshape(a.shape[0], -1) * b
    elif rule == 'se,sc->sec':
        return a.unsqueeze(2) * b.unsqueeze(1)
    elif rule == 'se,se->s':
        return torch.bmm(a.unsqueeze(1), b.unsqueeze(2)).reshape(-1)
    elif rule == 'se,sec->sec':
        return a.unsqueeze(2) * b
    elif rule == 'sec,sm->ecm':
        s = a.shape[0]
        e = a.shape[1]
        c = a.shape[2]
        m = b.shape[1]
        return torch.matmul(a.reshape(s, -1).t(), b).reshape(e, c, m)
    elif rule == 'sec,ecm->sm':
        return torch.matmul(a.reshape(a.shape[0], -1), b.reshape(-1, b.shape[-1]))
    elif rule == 'ks,ksm->sm':
        k = b.shape[0]
        s = b.shape[1]
        m = b.shape[2]
        # [k, s] -> [s, k] -> [s, 1, k]
        a = a.t().unsqueeze(1)
        # [k,s,m] -> [k, sm] -> [sm, k] -> [s, m, k]
        b = b.reshape(k, -1).t().reshape(s, m, k)
        # bmm([s, 1, k], [s, m, k]^t) -> [s, m, 1]
        return torch.bmm(a, b.transpose(1, 2)).squeeze(2)
    else:
        return torch.einsum(rule, a, b)

@torch.jit.script
def _capacity(num_tokens : int, num_experts : int, capacity_factor: Tensor, min_capacity: Tensor) -> Tensor:
    # # gates has shape of SE
    # num_tokens = gates.shape[0]
    # num_experts = gates.shape[1]
    # to(torch.int64) works around a bug in torch.onnx.export:
    # it should cast k to int64 when converting torch.topk but it doesn't.
    capacity = torch.ceil((num_tokens / num_experts) * capacity_factor).to(torch.int64)
    if capacity < min_capacity:
        capacity = min_capacity.to(torch.int64)
    return capacity

@torch.jit.script
def _one_hot_to_float(x, num_classes):
    return F.one_hot(x, num_classes=num_classes).float()      


def topkgating(
    inputs: Tensor,
    weights: Tensor,
    k: int,
    capacity_factor: float,
    min_capacity: int,
    drop_tokens: bool = True,
    ep_group=None,
    drop_policy: str = "probs",
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    logits = inputs @ weights 
   # everything is in fp32 in this function
    # get topk gates
    top_gate, top_idx = torch.topk(logits, k=k, dim=1)
    # gating decisions
    gates = F.softmax(logits, dim=1)
    num_experts = int(gates.shape[1])
    num_tokens = int(logits.shape[0])

    # # get topk mask
    topk_masked_gates = torch.zeros_like(logits).scatter(1, top_idx, top_gate)

    mask = torch.zeros_like(gates, dtype=torch.bool).scatter_(1, top_idx, 1)

    exp_counts = torch.sum(mask, dim=0).detach().to(logits.device)

    #Compute l_aux  0.005
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask.float(), dim=0)
    l_aux = torch.mean(me * ce) * num_experts * num_experts / k

    if drop_tokens:
        # Calculate configured capacity and remove locations outside capacity from mask
        capacity = _capacity(num_tokens, num_experts, torch.tensor(capacity_factor * k), torch.tensor(min_capacity))
        # update mask and locations by capacity

        if drop_policy == 'probs':
            capacity_probs, capacity_indices = torch.topk(topk_masked_gates, k=capacity, dim=0, sorted=False)  ## topk dim0 0.015
            capacity_mask = torch.zeros_like(logits).scatter(0, capacity_indices, 1)
            mask = torch.logical_and(mask, capacity_mask)
            locations = torch.cumsum(mask, dim=0) - 1  ##0.01
            

        elif drop_policy == "position":
            locations = torch.cumsum(mask, dim=0) - 1
            mask *= torch.lt(locations, capacity)
        else:
            raise ValueError(f"Invalid drop_policy: {drop_policy}")

    # #normalize gates ##0.002
    # gates_masked = gates * mask
    # gates_s = torch.sum(gates_masked, dim=-1, keepdim=True)
    # denom_s = torch.clamp(gates_s, min=torch.finfo(gates_masked.dtype).eps)
    # gates_masked = gates_masked / denom_s

    # # dispatch_mask
    # locations_sc = _one_hot_to_float((locations * mask), capacity)

    # combine_weights = torch.einsum("se,sec->sec", gates_masked, locations_sc)

    # dispatch_mask = combine_weights.bool()

    # return l_aux, combine_weights, dispatch_mask, exp_counts 
    return  


batch_size=4096
hidden_dimension = 7168
num_experts=256
#Default Shared Memory per SM: 164 KB 
inputs = torch.randn(batch_size, hidden_dimension)
weights = torch.randn(hidden_dimension, num_experts)
res = benchmark(lambda: topkgating(inputs, weights, 8, 1.2, 1, True, "probs"))
print(res)