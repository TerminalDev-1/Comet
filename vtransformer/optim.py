"""Optimizers: Muon for 2D hidden weight matrices, AdamW for everything else.

Muon = SGD momentum whose update is orthogonalized with a few Newton-Schulz iterations.
It stores 1 momentum buffer per weight (AdamW stores 2), and usually reaches a given
loss in noticeably fewer steps.
"""
import torch


@torch.no_grad()
def newton_schulz(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    X = X / (X.norm() + 1e-7)
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if transposed else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.0, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p)
                buf = st["buf"]
                buf.mul_(g["momentum"]).add_(p.grad)
                upd = p.grad.add(buf, alpha=g["momentum"])  # Nesterov
                upd = newton_schulz(upd, g["ns_steps"]).to(p.dtype)
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                if g["weight_decay"]:
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(upd, alpha=-g["lr"] * scale)


class Combined:
    """Wraps several optimizers so train.py can treat them as one."""

    def __init__(self, opts):
        self.opts = opts
        self.param_groups = [g for o in opts for g in o.param_groups]

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.opts:
            o.step()


def build_optimizer(model, kind, lr, weight_decay, muon_lr=0.02):
    """Each param group gets a 'base_lr'; train.py scales all groups by the schedule."""
    named = list(model.named_parameters())
    if kind == "muon":
        hidden = [p for n, p in named if p.dim() == 2 and not n.startswith("tok.")]
        other_decay = [p for n, p in named if p.dim() == 2 and n.startswith("tok.")]
        other_nodecay = [p for n, p in named if p.dim() < 2]
        muon = Muon(hidden, lr=muon_lr, weight_decay=0.0)
        adam = torch.optim.AdamW(
            [{"params": other_decay, "weight_decay": weight_decay},
             {"params": other_nodecay, "weight_decay": 0.0}],
            lr=lr, betas=(0.9, 0.95))
        opt = Combined([muon, adam])
    else:
        decay = [p for n, p in named if p.dim() >= 2]
        nodecay = [p for n, p in named if p.dim() < 2]
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": weight_decay}, {"params": nodecay, "weight_decay": 0.0}],
            lr=lr, betas=(0.9, 0.95))
    for g in opt.param_groups:
        g["base_lr"] = g["lr"]
    return opt
