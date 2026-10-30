import math
import torch
from torch.optim import Optimizer


def newton_schulz_orthogonalize(matrix, steps=5, eps=1e-7):
    """Numerically safer Newton-Schulz orthogonalization of a 2D matrix."""
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D matrix.")

    original_dtype = matrix.dtype
    x = matrix.float()

    # Work with the smaller Gram matrix.
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T

    norm = torch.linalg.vector_norm(x)
    if not torch.isfinite(norm) or norm.item() <= eps:
        return torch.zeros_like(matrix)

    x = x / norm.clamp_min(eps)

    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(int(steps)):
        gram = x @ x.T
        gram_x = gram @ x
        x = a * x + b * gram_x + c * (gram @ gram_x)

    if transposed:
        x = x.T

    return x.to(dtype=original_dtype)


class TITAN(Optimizer):
    """
    Hybrid optimizer:
      * 2D parameters: exponential momentum + Newton-Schulz orthogonalization.
      * Other parameter shapes: bias-corrected Adam-style update.
      * FP32 optimizer state, decoupled weight decay and optional global clipping.

    Sparse gradients are rejected. The class follows the torch.optim.Optimizer
    interface and supports closures, AMP GradScaler and LR schedulers.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        ns_steps=5,
        matrix_mode=True,
        matrix_scale=True,
        max_grad_norm=None,
    ):
        if lr <= 0:
            raise ValueError("lr must be positive.")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        if ns_steps < 0:
            raise ValueError("ns_steps must be non-negative.")
        if not (0 <= betas[0] < 1 and 0 <= betas[1] < 1):
            raise ValueError("betas must be in [0, 1).")
        if max_grad_norm is not None and max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive.")

        defaults = dict(lr=lr, weight_decay=weight_decay)
        super().__init__(params, defaults)

        self.beta1, self.beta2 = betas
        self.eps = eps
        self.ns_steps = ns_steps
        self.matrix_mode = matrix_mode
        self.matrix_scale = matrix_scale
        self.max_grad_norm = max_grad_norm

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self.max_grad_norm is not None:
            params = [
                p for group in self.param_groups
                for p in group["params"]
                if p.grad is not None and not p.grad.is_sparse
            ]
            if params:
                torch.nn.utils.clip_grad_norm_(params, self.max_grad_norm)

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError(
                        "TITAN requires dense gradients. "
                        "Use sparse=False for embedding layers."
                    )
                if not torch.isfinite(g).all():
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(
                        p, dtype=torch.float32,
                        memory_format=torch.preserve_format
                    )
                    state["v"] = torch.zeros_like(
                        p, dtype=torch.float32,
                        memory_format=torch.preserve_format
                    )

                state["step"] += 1
                t = state["step"]
                g32 = g.float()
                m, v = state["m"], state["v"]

                m.mul_(self.beta1).add_(g32, alpha=1.0 - self.beta1)
                v.mul_(self.beta2).addcmul_(
                    g32, g32, value=1.0 - self.beta2
                )

                if self.matrix_mode and p.ndim == 2:
                    update = newton_schulz_orthogonalize(
                        m, steps=self.ns_steps, eps=self.eps
                    ).float()
                    if self.matrix_scale:
                        rows, cols = m.shape
                        scale = math.sqrt(max(1.0, rows / max(1, cols)))
                        update.mul_(scale)
                else:
                    m_hat = m / (1.0 - self.beta1 ** t)
                    v_hat = v / (1.0 - self.beta2 ** t)
                    update = m_hat / (v_hat.sqrt() + self.eps)

                if wd != 0:
                    p.mul_(1.0 - lr * wd)
                p.add_(update.to(dtype=p.dtype), alpha=-lr)

        return loss
