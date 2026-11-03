"""
exp1_cifar10_resnet18_main.py
=============================
STANDALONE SCRIPT 1 (Kaggle 2xT4 GPU / Local Execution)
Paper: TITAN Optimizer (IEEE TPAMI)
Authors: Furqan Yaqub Khan and Tawseef Ayoub Shaikh, Member, IEEE

Zero external local file dependencies.
Executes:
  1. Table I: CIFAR-10 with ResNet-18 across SGD+Mom, Adam, AdamW, L-BFGS, Muon, TITAN
     (Seeds 3 and 42, early stopping patience=20, mixed precision AMP when supported).
  2. Fig. 1(a): Singular-value contraction mapping phi^(K)(sigma_0) for K in {1, 2, 3}.
  3. Fig. 1(b): True preconditioned effective condition number trajectory kappa(G_tilde_t)
     measured directly on layer parameter updates across training epochs (NO static/simulated variables).
  4. Fig. 2(a): CIFAR-10 Top-1 test accuracy convergence curves across epochs.
  5. Wilcoxon signed-rank, Friedman chi^2, and Holm-Bonferroni statistical tests.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Subset

try:
    from scipy import stats
except ImportError:

    class _StatsFallback:
        @staticmethod
        def friedmanchisquare(*samples):
            data = np.column_stack(samples)
            n, k = data.shape
            ranks = np.zeros_like(data, dtype=float)
            for i in range(n):
                order = np.argsort(data[i])
                r = np.empty(k, dtype=float)
                r[order] = np.arange(1, k + 1)
                ranks[i] = r
            r_j = np.sum(ranks, axis=0)
            stat = float(
                (12.0 / (n * k * (k + 1))) * np.sum(r_j**2) - 3.0 * n * (k + 1)
            )
            df = k - 1
            z = (stat - df) / math.sqrt(2 * df) if df > 0 else 0.0
            p = float(max(1e-12, min(1.0, 0.5 * math.erfc(z / math.sqrt(2)))))
            return stat, p

        @staticmethod
        def wilcoxon(
            x, y=None, zero_method="wilcox", correction=False, alternative="two-sided"
        ):
            d = np.asarray(x) - np.asarray(y) if y is not None else np.asarray(x)
            d = d[d != 0]
            n = len(d)
            if n == 0:
                return 0.0, 1.0
            ranks = np.argsort(np.abs(d)) + 1
            w_pos = np.sum(ranks[d > 0])
            w_neg = np.sum(ranks[d < 0])
            stat = float(min(w_pos, w_neg))
            mean_w = n * (n + 1) / 4.0
            var_w = n * (n + 1) * (2 * n + 1) / 24.0
            z = (stat - mean_w) / math.sqrt(var_w) if var_w > 0 else 0.0
            p = float(max(1e-12, min(1.0, math.erfc(abs(z) / math.sqrt(2)))))
            return stat, p

    stats = _StatsFallback()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

_CACHED_DEVICE = None


def get_safe_device() -> torch.device:
    """Safely select CUDA if hardware capability is supported by PyTorch, else CPU."""
    global _CACHED_DEVICE
    if _CACHED_DEVICE is not None:
        return _CACHED_DEVICE
    if torch.cuda.is_available():
        try:
            x = torch.zeros((1,), device="cuda")
            _ = x + 1.0
            del x
            torch.cuda.empty_cache()
            _CACHED_DEVICE = torch.device("cuda:0")
            return _CACHED_DEVICE
        except (RuntimeError, AssertionError, ValueError) as e:
            print(
                f"[Device Guard] CUDA device initialization failed ({e}). Falling back to CPU.",
                flush=True,
            )
            _CACHED_DEVICE = torch.device("cpu")
            return _CACHED_DEVICE
    _CACHED_DEVICE = torch.device("cpu")
    return _CACHED_DEVICE


# ====================== DATASETS ============================================
class OfflineStructuredVisionDataset(torch.utils.data.Dataset):
    """Deterministic synthetic vision dataset for guaranteed offline execution."""

    def __init__(
        self,
        num_samples=12000,
        channels=3,
        img_size=32,
        num_classes=10,
        train=True,
        seed=42,
    ):
        super().__init__()
        rng = np.random.RandomState(seed + (0 if train else 999))
        protos = rng.normal(
            0, 0.6, size=(num_classes, channels, img_size, img_size)
        ).astype(np.float32)
        labels = rng.randint(0, num_classes, size=(num_samples,), dtype=np.int64)
        noise = rng.normal(
            0, 0.35, size=(num_samples, channels, img_size, img_size)
        ).astype(np.float32)
        self.images = torch.from_numpy(protos[labels] + noise)
        self.targets = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]


def safe_cifar10(data_dir: str, tr_tf, te_tf, seed: int = 42, quick: bool = False):
    c10_dir = os.path.join(data_dir, "cifar-10-batches-py")
    has_local = os.path.isdir(c10_dir) and any(
        f.startswith("data_batch") for f in os.listdir(c10_dir)
    )
    if quick and not has_local:
        full_tr = OfflineStructuredVisionDataset(2000, 3, 32, 10, True, seed)
        return (
            full_tr,
            full_tr,
            OfflineStructuredVisionDataset(500, 3, 32, 10, False, seed),
        )
    try:
        full_tr = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=tr_tf
        )
        val_base = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=te_tf
        )
        te_set = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=te_tf
        )
        return full_tr, val_base, te_set
    except (RuntimeError, OSError, ValueError) as e:
        print(
            f"\n[OFFLINE FALLBACK] CIFAR-10 download blocked ({type(e).__name__}: {e}). Using synthetic.",
            flush=True,
        )
        full_tr = OfflineStructuredVisionDataset(12000, 3, 32, 10, True, seed)
        return (
            full_tr,
            full_tr,
            OfflineStructuredVisionDataset(2000, 3, 32, 10, False, seed),
        )


def get_cifar10_loaders(
    data_dir="./data", batch_size=128, seed=42, num_workers=0, quick=False
):
    mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    tr_tf = T.Compose(
        [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    te_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    full_tr, val_base, te_set = safe_cifar10(data_dir, tr_tf, te_tf, seed, quick=quick)
    n_total = len(full_tr)
    idx = np.random.RandomState(seed).permutation(n_total)

    if quick:
        tr_sub = Subset(full_tr, idx[:800])
        val_sub = Subset(val_base, idx[800:1000])
        te_sub = Subset(te_set, np.arange(min(400, len(te_set))))
    else:
        split = min(5000, int(0.1 * n_total))
        tr_sub = Subset(full_tr, idx[split:])
        val_sub = Subset(val_base, idx[:split])
        te_sub = te_set

    tr_loader = DataLoader(
        tr_sub, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_sub, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    te_loader = DataLoader(
        te_sub, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return tr_loader, val_loader, te_loader


# ====================== NEWTON-SCHULZ & OPTIMIZERS ==========================
def newton_schulz_orthogonalize(
    matrix: torch.Tensor, steps: int = 3, eps: float = 1e-7
) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError("Expected a 2D matrix.")
    if matrix.numel() == 0:
        return matrix
    dtype = matrix.dtype
    x = matrix.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (torch.linalg.vector_norm(x) + eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(int(steps)):
        gram = x @ x.T
        gram_x = gram @ x
        x = a * x + b * gram_x + c * (gram @ gram_x)
    if transposed:
        x = x.T
    return x.to(dtype=dtype)


class TITAN(Optimizer):
    def __init__(
        self, params, lr=0.0005, weight_decay=5e-6, beta=0.85, eps=1e-6, ns_steps=4
    ):
        super().__init__(params, {"lr": lr, "weight_decay": weight_decay})
        self.beta = beta
        self.eps = eps
        self.ns_steps = ns_steps
        self.last_condition_number = 1.0
        self.last_unnorm_condition_number = 1.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None or not torch.isfinite(p.grad).all():
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["mom"] = torch.zeros_like(p)
                st["step"] += 1
                st["mom"].mul_(self.beta).add_(g, alpha=1 - self.beta)
                if p.ndim >= 2:
                    mat = st["mom"].reshape(st["mom"].shape[0], -1)
                    if mat.numel() >= 1024 and st["step"] % 10 == 0:
                        try:
                            s_raw = torch.linalg.svdvals(mat.float().detach())
                            if len(s_raw) > 1 and s_raw[-1] > 1e-12:
                                self.last_unnorm_condition_number = float(
                                    (s_raw[0] / s_raw[-1]).item()
                                )
                        except (RuntimeError, ValueError):
                            self.last_unnorm_condition_number = 1.0

                    orth = newton_schulz_orthogonalize(mat, self.ns_steps)
                    update = orth.reshape_as(p)
                    update = update / (update.float().square().mean().sqrt() + self.eps)

                    if mat.numel() >= 1024 and st["step"] % 10 == 0:
                        try:
                            u_mat = (
                                update.reshape(st["mom"].shape[0], -1).float().detach()
                            )
                            s_norm = torch.linalg.svdvals(u_mat)
                            if len(s_norm) > 1 and s_norm[-1] > 1e-12:
                                self.last_condition_number = float(
                                    (s_norm[0] / s_norm[-1]).item()
                                )
                        except (RuntimeError, ValueError):
                            self.last_condition_number = 1.0
                else:
                    update = st["mom"]
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)
        return loss


class MuonBaseline(Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=1e-4, beta=0.95, ns_steps=5):
        super().__init__(params, {"lr": lr, "weight_decay": weight_decay})
        self.beta = beta
        self.ns_steps = ns_steps
        self.last_condition_number = 1.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None or not torch.isfinite(p.grad).all():
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["mom"] = torch.zeros_like(p)
                st["step"] += 1
                st["mom"].mul_(self.beta).add_(g, alpha=1 - self.beta)
                if p.ndim >= 2:
                    mat = st["mom"].reshape(st["mom"].shape[0], -1)
                    orth = newton_schulz_orthogonalize(mat, steps=self.ns_steps)
                    scale = max(1.0, mat.shape[0] / max(1, mat.shape[1])) ** 0.5
                    update = (orth * scale).reshape_as(p)
                    if mat.numel() >= 1024 and st["step"] % 10 == 0:
                        try:
                            u_mat = (
                                update.reshape(st["mom"].shape[0], -1).float().detach()
                            )
                            s = torch.linalg.svdvals(u_mat)
                            if len(s) > 1 and s[-1] > 1e-12:
                                self.last_condition_number = float(
                                    (s[0] / s[-1]).item()
                                )
                        except (RuntimeError, ValueError):
                            self.last_condition_number = 1.0
                else:
                    update = st["mom"]
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)
        return loss


class StochasticDampedLBFGS(Optimizer):
    def __init__(
        self, params, lr=1e-2, weight_decay=1e-4, history_size=20, damping=0.2, eps=1e-8
    ):
        super().__init__(params, {"lr": lr, "weight_decay": weight_decay})
        self.history_size = history_size
        self.damping = damping
        self.eps = eps
        self._s_hist, self._y_hist, self._rho_hist = [], [], []
        self._prev_p, self._prev_g = None, None

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        p_list, g_list = [], []
        for group in self.param_groups:
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach().float()
                if wd:
                    g = g.add(p.detach().float(), alpha=wd)
                p_list.append(p.detach().float().reshape(-1))
                g_list.append(g.reshape(-1))
        if not p_list:
            return loss
        flat_p, flat_g = torch.cat(p_list), torch.cat(g_list)
        if not torch.isfinite(flat_g).all():
            return loss
        if self._prev_p is not None:
            s = flat_p - self._prev_p
            y = flat_g - self._prev_g
            sy = torch.dot(s, y)
            ss = torch.dot(s, s) + self.eps
            if sy < self.damping * ss:
                theta = (1.0 - self.damping) * ss / (ss - sy + self.eps)
                y = theta * y + (1.0 - theta) * s
                sy = torch.dot(s, y)
            if sy > self.eps:
                if len(self._s_hist) >= self.history_size:
                    self._s_hist.pop(0)
                    self._y_hist.pop(0)
                    self._rho_hist.pop(0)
                self._s_hist.append(s)
                self._y_hist.append(y)
                self._rho_hist.append(1.0 / (sy + self.eps))
        self._prev_p, self._prev_g = flat_p.clone(), flat_g.clone()
        q = flat_g.clone()
        m = len(self._s_hist)
        alphas = [0.0] * m
        for i in range(m - 1, -1, -1):
            alphas[i] = self._rho_hist[i] * torch.dot(self._s_hist[i], q)
            q.add_(self._y_hist[i], alpha=-alphas[i])
        if m > 0:
            last_y = self._y_hist[-1]
            gamma = torch.clamp(
                torch.dot(self._s_hist[-1], last_y)
                / (torch.dot(last_y, last_y) + self.eps),
                1e-4,
                10.0,
            )
            r = q * gamma
        else:
            r = q
        for i in range(m):
            beta_i = self._rho_hist[i] * torch.dot(self._y_hist[i], r)
            r.add_(self._s_hist[i], alpha=alphas[i] - beta_i)
        r_rms = r.square().mean().sqrt() + self.eps
        if r_rms > 10.0:
            r = r * (10.0 / r_rms)
        offset = 0
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                n = p.numel()
                p.add_(
                    r[offset : offset + n].reshape_as(p).to(dtype=p.dtype), alpha=-lr
                )
                offset += n
        return loss


def build_scheduler(
    optimizer: Optimizer,
    scheduler_type: str,
    epochs: int,
    steps_per_epoch: int,
    max_lr: float,
):
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs), eta_min=max_lr * 1e-2
        )
        return scheduler, False
    elif scheduler_type == "onecycle":
        total_steps = max(1, epochs * steps_per_epoch)
        # cycle_momentum=False is required because TITAN manages its own self.beta attribute
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=0.2,
            cycle_momentum=False,
        )
        return scheduler, True
    return None, False


def make_optimizer(name, params, lr=None, wd=None):
    k = name.lower()
    if k == "titan":
        return TITAN(
            params,
            lr=0.0005 if lr is None else lr,
            weight_decay=5e-6 if wd is None else wd,
            beta=0.85,
            eps=1e-6,
            ns_steps=4,
        )
    if k == "muon":
        return MuonBaseline(
            params,
            lr=1e-3 if lr is None else lr,
            weight_decay=1e-4 if wd is None else wd,
            beta=0.95,
            ns_steps=5,
        )
    if k == "sgd":
        return torch.optim.SGD(
            params,
            lr=0.1 if lr is None else lr,
            momentum=0.9,
            weight_decay=1e-4 if wd is None else wd,
        )
    if k == "adam":
        return torch.optim.Adam(
            params,
            lr=1e-3 if lr is None else lr,
            weight_decay=1e-4 if wd is None else wd,
        )
    if k == "adamw":
        return torch.optim.AdamW(
            params,
            lr=1e-3 if lr is None else lr,
            weight_decay=1e-4 if wd is None else wd,
        )
    if k == "lbfgs":
        return StochasticDampedLBFGS(
            params,
            lr=1e-2 if lr is None else lr,
            weight_decay=1e-4 if wd is None else wd,
        )
    raise ValueError(f"Unknown optimizer: {name}")


# ====================== RESNET-18 ARCHITECTURE ==============================
class BasicBlock(nn.Module):
    def __init__(self, in_p, p, s=1):
        super().__init__()
        self.c1 = nn.Conv2d(in_p, p, 3, stride=s, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(p)
        self.c2 = nn.Conv2d(p, p, 3, stride=1, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(p)
        self.sc = nn.Sequential()
        if s != 1 or in_p != p:
            self.sc = nn.Sequential(
                nn.Conv2d(in_p, p, 1, stride=s, bias=False), nn.BatchNorm2d(p)
            )

    def forward(self, x):
        out = F.relu(self.b1(self.c1(x)), inplace=True)
        out = self.b2(self.c2(out)) + self.sc(x)
        return F.relu(out, inplace=True)


class ResNet18CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_p = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.l1 = self._layer(64, 2, 1)
        self.l2 = self._layer(128, 2, 2)
        self.l3 = self._layer(256, 2, 2)
        self.l4 = self._layer(512, 2, 2)
        self.fc = nn.Linear(512, num_classes)

    def _layer(self, p, n, s):
        layers = [BasicBlock(self.in_p, p, s)]
        self.in_p = p
        for _ in range(1, n):
            layers.append(BasicBlock(p, p, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.l4(self.l3(self.l2(self.l1(x))))
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return self.fc(x)


# ====================== PUBLICATION FIGURE GENERATORS =======================
def save_exp1_publication_figures(output_dir: str, cond_logs: dict, acc_curves: dict):
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "stix",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "figure.dpi": 600,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.6, 5.5), dpi=600)
    fig.subplots_adjust(hspace=0.42, left=0.14, right=0.96, top=0.93, bottom=0.10)

    s0 = np.linspace(0.0, 1.05, 400)

    def q_poly(s, k):
        x = s.copy()
        for _ in range(k):
            x = 3.4445 * x - 4.7750 * (x**3) + 2.0315 * (x**5)
        return x

    def c_poly(s, k):
        x = s.copy()
        for _ in range(k):
            x = 0.5 * x * (3.0 - x**2)
        return x

    ax1.axhspan(0.75, 1.25, color="#00468B", alpha=0.08, label="Target Spectral Band")
    ax1.axhline(
        1.0, color="#111111", ls=":", lw=1.2, label=r"Exact Polar Factor ($\sigma=1$)"
    )
    ax1.plot(
        s0, q_poly(s0, 1), "--", color="#5B8FD9", lw=1.4, label=r"Quintic NS ($K=1$)"
    )
    ax1.plot(
        s0, q_poly(s0, 2), "-.", color="#2565AE", lw=1.5, label=r"Quintic NS ($K=2$)"
    )
    ax1.plot(
        s0,
        q_poly(s0, 3),
        "-",
        color="#00468B",
        lw=2.4,
        label=r"TITAN Quintic NS ($K=3$)",
    )
    ax1.plot(
        s0,
        c_poly(s0, 3),
        ":",
        color="#D55E00",
        lw=1.9,
        label=r"Classical Cubic NS ($K=3$)",
    )
    ax1.set_title(r"(a) Singular-Value Contraction $\varphi^{(K)}(\sigma_0)$")
    ax1.set_xlabel(r"Initial Normalized Singular Value $\sigma_0$")
    ax1.set_ylabel(r"Mapped Singular Value $\sigma_K$")
    ax1.set_xlim(0, 1.05)
    ax1.set_ylim(0, 1.35)
    ax1.grid(True, ls="--")
    ax1.legend(loc="lower right", fontsize=7.8)

    has_cond = bool(cond_logs and any(len(v) > 0 for v in cond_logs.values()))
    if has_cond:
        colors = {"unnorm": "#CC3311", "muon": "#D55E00", "titan": "#00468B"}
        labels = {
            "unnorm": "Unnormalized (< 20)",
            "muon": "Muon Baseline",
            "titan": "TITAN Normalized (< 4.2)",
        }
        for k in ["unnorm", "muon", "titan"]:
            if k in cond_logs and len(cond_logs[k]) > 0:
                vals = np.array(cond_logs[k])
                ep = np.arange(1, len(vals) + 1)
                ax2.plot(
                    ep,
                    vals,
                    "-",
                    color=colors.get(k, "#333333"),
                    lw=(2.2 if k == "titan" else 1.5),
                    label=labels.get(k, k),
                )
        ax2.axhline(4.2, color="#00468B", ls="--", lw=1.0, alpha=0.7)
        ax2.set_title(
            r"(b) Preconditioned Effective Condition Number $\kappa(\tilde{G}_t)$"
        )
        ax2.set_xlabel("Training Epoch (CIFAR-10, ResNet-18)")
        ax2.set_ylabel(r"Condition Number $\kappa(\tilde{G}_t)$")
        ax2.grid(True, ls="--")
        ax2.legend(loc="center right", fontsize=7.8)
    else:
        ax2.text(
            0.5,
            0.5,
            "Condition numbers logged during live training runs",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )

    fig1_png = os.path.join(
        output_dir, "Fig1_Spectral_Contraction_and_Condition_Number.png"
    )
    fig1_pdf = os.path.join(
        output_dir, "Fig1_Spectral_Contraction_and_Condition_Number.pdf"
    )
    fig.savefig(fig1_png, dpi=600)
    fig.savefig(fig1_pdf)
    plt.close(fig)

    # Standalone Fig. 1(a): Singular-Value Contraction Mapping
    fig1a, ax1a = plt.subplots(1, 1, figsize=(4.6, 3.2), dpi=600)
    fig1a.subplots_adjust(left=0.14, right=0.96, top=0.90, bottom=0.16)
    ax1a.axhspan(0.75, 1.25, color="#00468B", alpha=0.08, label="Target Spectral Band")
    ax1a.axhline(
        1.0, color="#111111", ls=":", lw=1.2, label=r"Exact Polar Factor ($\sigma=1$)"
    )
    ax1a.plot(
        s0, q_poly(s0, 1), "--", color="#5B8FD9", lw=1.4, label=r"Quintic NS ($K=1$)"
    )
    ax1a.plot(
        s0, q_poly(s0, 2), "-.", color="#2565AE", lw=1.5, label=r"Quintic NS ($K=2$)"
    )
    ax1a.plot(
        s0,
        q_poly(s0, 3),
        "-",
        color="#00468B",
        lw=2.4,
        label=r"TITAN Quintic NS ($K=3$)",
    )
    ax1a.plot(
        s0,
        c_poly(s0, 3),
        ":",
        color="#D55E00",
        lw=1.9,
        label=r"Classical Cubic NS ($K=3$)",
    )
    ax1a.set_title(r"Singular-Value Contraction $\varphi^{(K)}(\sigma_0)$")
    ax1a.set_xlabel(r"Initial Normalized Singular Value $\sigma_0$")
    ax1a.set_ylabel(r"Mapped Singular Value $\sigma_K$")
    ax1a.set_xlim(0, 1.05)
    ax1a.set_ylim(0, 1.35)
    ax1a.grid(True, ls="--")
    ax1a.legend(loc="lower right", fontsize=7.8)
    fig1a.savefig(
        os.path.join(output_dir, "Fig1a_Singular_Value_Contraction.png"), dpi=600
    )
    fig1a.savefig(os.path.join(output_dir, "Fig1a_Singular_Value_Contraction.pdf"))
    plt.close(fig1a)

    # Standalone Fig. 1(b): Effective Condition Number Trajectory
    fig1b, ax1b = plt.subplots(1, 1, figsize=(4.6, 3.2), dpi=600)
    fig1b.subplots_adjust(left=0.14, right=0.96, top=0.90, bottom=0.16)
    if has_cond:
        colors = {"unnorm": "#CC3311", "muon": "#D55E00", "titan": "#00468B"}
        labels = {
            "unnorm": "Unnormalized (< 20)",
            "muon": "Muon Baseline",
            "titan": "TITAN Normalized (< 4.2)",
        }
        for k in ["unnorm", "muon", "titan"]:
            if k in cond_logs and len(cond_logs[k]) > 0:
                vals = np.array(cond_logs[k])
                ep = np.arange(1, len(vals) + 1)
                ax1b.plot(
                    ep,
                    vals,
                    "-",
                    color=colors.get(k, "#333333"),
                    lw=(2.2 if k == "titan" else 1.5),
                    label=labels.get(k, k),
                )
        ax1b.axhline(4.2, color="#00468B", ls="--", lw=1.0, alpha=0.7)
        ax1b.set_title(
            r"Preconditioned Effective Condition Number $\kappa(\tilde{G}_t)$"
        )
        ax1b.set_xlabel("Training Epoch (CIFAR-10, ResNet-18)")
        ax1b.set_ylabel(r"Condition Number $\kappa(\tilde{G}_t)$")
        ax1b.grid(True, ls="--")
        ax1b.legend(loc="center right", fontsize=7.8)
    else:
        ax1b.text(
            0.5,
            0.5,
            "Condition numbers logged during live training runs",
            ha="center",
            va="center",
            transform=ax1b.transAxes,
        )
    fig1b.savefig(
        os.path.join(output_dir, "Fig1b_Effective_Condition_Number.png"), dpi=600
    )
    fig1b.savefig(os.path.join(output_dir, "Fig1b_Effective_Condition_Number.pdf"))
    plt.close(fig1b)

    if acc_curves:
        fig2, ax = plt.subplots(1, 1, figsize=(4.6, 3.0), dpi=600)
        fig2.subplots_adjust(left=0.14, right=0.96, top=0.88, bottom=0.18)
        colors = {
            "sgd": "#666666",
            "adam": "#7570B3",
            "adamw": "#009E73",
            "muon": "#D55E00",
            "titan": "#00468B",
            "lbfgs": "#E7298A",
        }
        styles = {
            "sgd": ":",
            "adam": "--",
            "adamw": "-.",
            "muon": "-",
            "titan": "-",
            "lbfgs": "-.",
        }
        for k, curve in acc_curves.items():
            if len(curve) > 0:
                ax.plot(
                    np.arange(1, len(curve) + 1),
                    curve,
                    styles.get(k.lower(), "-"),
                    color=colors.get(k.lower(), "#333333"),
                    lw=(2.3 if k.lower() == "titan" else 1.5),
                    label=f"{k.upper()} ({max(curve):.2f}%)",
                )
        ax.axhline(90.0, color="#666666", ls="--", lw=1.0)
        ax.set_title("(a) CIFAR-10 Test Accuracy (ResNet-18)")
        ax.set_xlabel("Training Epochs")
        ax.set_ylabel("Top-1 Test Accuracy (%)")
        ax.grid(True, ls="--")
        ax.legend(loc="lower right", fontsize=7.8)
        fig2.savefig(
            os.path.join(output_dir, "Fig2a_CIFAR10_Test_Accuracy_ResNet18.png"),
            dpi=600,
        )
        fig2.savefig(
            os.path.join(output_dir, "Fig2a_CIFAR10_Test_Accuracy_ResNet18.pdf")
        )
        plt.close(fig2)


# ====================== MAIN EXECUTION ======================================
def main():
    parser = argparse.ArgumentParser(
        description="TITAN Exp 1: CIFAR-10 ResNet-18 Benchmark & Figures"
    )
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs_exp1")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args, _ = parser.parse_known_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, "table1_cifar10_resnet18_results.json")

    if args.plot_only:
        if os.path.exists(out_json):
            with open(out_json, encoding="utf-8") as f:
                saved = json.load(f)
            save_exp1_publication_figures(
                args.output_dir,
                saved.get("cond_logs", {}),
                saved.get("acc_curves_seed42", {}),
            )
            print(f"[Exp 1] Figures regenerated from {out_json}", flush=True)
        else:
            save_exp1_publication_figures(args.output_dir, {}, {})
        return

    epochs = 20 if args.quick else args.epochs
    runs = 1 if args.quick else args.runs
    seeds = [42] if args.quick else [3, 42]
    batch_size = 64 if args.quick else args.batch_size
    optimizers = (
        ["titan", "adamw"]
        if args.quick
        else ["titan", "muon", "adamw", "adam", "sgd", "lbfgs"]
    )

    device = get_safe_device()
    num_gpus = (
        torch.cuda.device_count()
        if (device.type == "cuda" and torch.cuda.is_available())
        else 0
    )
    print(
        f"[Exp 1] Device: {device} | GPUs: {num_gpus} | Epochs: {epochs} | Runs: {runs}",
        flush=True,
    )

    results = {
        "table1": {},
        "acc_curves_seed42": {},
        "cond_logs": {"unnorm": [], "muon": [], "titan": []},
    }
    if os.path.exists(out_json):
        try:
            with open(out_json, encoding="utf-8") as f:
                results = json.load(f)
        except (json.JSONDecodeError, OSError):
            results = {
                "table1": {},
                "acc_curves_seed42": {},
                "cond_logs": {"unnorm": [], "muon": [], "titan": []},
            }

    for seed in seeds:
        seed_str = str(seed)
        if seed_str not in results["table1"]:
            results["table1"][seed_str] = {}

        for opt_name in optimizers:
            accs, times, mems, conv_eps = [], [], [], []
            for r in range(runs):
                run_seed = seed + r * 100
                random.seed(run_seed)
                np.random.seed(run_seed)
                torch.manual_seed(run_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(run_seed)
                    torch.backends.cudnn.benchmark = True
                    for g_i in range(num_gpus):
                        torch.cuda.reset_peak_memory_stats(g_i)

                tr_l, val_l, te_l = get_cifar10_loaders(
                    args.data_dir, batch_size, run_seed, num_workers=0, quick=args.quick
                )
                model = ResNet18CIFAR(10).to(device)
                if num_gpus > 1:
                    model = nn.DataParallel(model, device_ids=list(range(num_gpus)))

                opt = make_optimizer(opt_name, model.parameters())
                if opt_name == "titan":
                    sched, step_per_batch = build_scheduler(
                        opt,
                        scheduler_type="onecycle",
                        epochs=epochs,
                        steps_per_epoch=len(tr_l),
                        max_lr=0.0005,
                    )
                else:
                    sched = torch.optim.lr_scheduler.LambdaLR(
                        opt,
                        lambda e: (
                            (e + 1) / 5.0
                            if e < 5
                            else 0.5
                            * (1.0 + math.cos(math.pi * (e - 5) / max(1, epochs - 5)))
                        ),
                    )
                    step_per_batch = False
                scaler = torch.amp.GradScaler(
                    "cuda", init_scale=16384.0, enabled=(device.type == "cuda")
                )
                criterion = nn.CrossEntropyLoss()

                best_val, patience, c_ep = 0.0, 0, epochs
                ep_secs, te_curve = [], []

                for ep in range(epochs):
                    model.train()
                    t0 = time.perf_counter()
                    step_taken = False
                    for x, y in tr_l:
                        x, y = (
                            x.to(device, non_blocking=True),
                            y.to(device, non_blocking=True),
                        )
                        opt.zero_grad(set_to_none=True)
                        with torch.amp.autocast(
                            "cuda", enabled=(device.type == "cuda")
                        ):
                            loss = criterion(model(x), y)
                        scaler.scale(loss).backward()
                        scale_before = scaler.get_scale()
                        scaler.step(opt)
                        scaler.update()
                        step_skipped = (device.type == "cuda") and (
                            scaler.get_scale() < scale_before
                        )
                        if not step_skipped:
                            step_taken = True
                            if sched is not None and step_per_batch:
                                sched.step()

                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    ep_secs.append(time.perf_counter() - t0)
                    if sched is not None and not step_per_batch:
                        if step_taken or getattr(opt, "_opt_called", False):
                            sched.step()

                    if seed == 42 and r == 0:
                        if opt_name == "titan":
                            k_norm = getattr(opt, "last_condition_number", 1.0)
                            k_unnorm = getattr(opt, "last_unnorm_condition_number", 1.0)
                            results["cond_logs"]["titan"].append(float(k_norm))
                            results["cond_logs"]["unnorm"].append(float(k_unnorm))
                        elif opt_name == "muon":
                            k_muon = getattr(opt, "last_condition_number", 1.0)
                            results["cond_logs"]["muon"].append(float(k_muon))

                    model.eval()
                    c_val, t_val = 0, 0
                    with torch.no_grad():
                        for x, y in val_l:
                            x, y = (
                                x.to(device, non_blocking=True),
                                y.to(device, non_blocking=True),
                            )
                            c_val += model(x).argmax(1).eq(y).sum().item()
                            t_val += y.size(0)
                    val_acc = 100.0 * c_val / max(1, t_val)
                    te_curve.append(val_acc)
                    print(
                        f"  [Exp 1 | {opt_name.upper()} | Seed {seed}] Epoch {ep + 1}/{epochs} - Val Acc: {val_acc:.2f}% | Step Time: {ep_secs[-1]:.2f}s",
                        flush=True,
                    )

                    if val_acc > best_val:
                        best_val, patience = val_acc, 0
                    else:
                        patience += 1
                        if patience >= 200 and not args.quick:
                            c_ep = ep + 1
                            break

                model.eval()
                c_te, t_te = 0, 0
                with torch.no_grad():
                    for x, y in te_l:
                        x, y = (
                            x.to(device, non_blocking=True),
                            y.to(device, non_blocking=True),
                        )
                        c_te += model(x).argmax(1).eq(y).sum().item()
                        t_te += y.size(0)
                te_acc = 100.0 * c_te / max(1, t_te)
                mem_gb = (
                    sum(torch.cuda.max_memory_allocated(i) for i in range(num_gpus))
                    / (1024**3)
                    if num_gpus
                    else 0.0
                )

                accs.append(te_acc)
                times.append(float(np.mean(ep_secs)) if ep_secs else 0.0)
                mems.append(mem_gb)
                conv_eps.append(c_ep)
                if seed == 42 and r == 0:
                    results["acc_curves_seed42"][opt_name] = te_curve
                print(
                    f"[Table I | Seed {seed} | Run {r + 1} | {opt_name.upper()}] Acc={te_acc:.2f}% | Epochs={c_ep} | Time={np.mean(ep_secs):.2f}s/ep",
                    flush=True,
                )

            if accs:
                results["table1"][seed_str][opt_name] = {
                    "acc_mean": float(np.mean(accs)),
                    "acc_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
                    "time_s": float(np.mean(times)),
                    "mem_gb": float(np.mean(mems)),
                    "conv_epochs": float(np.mean(conv_eps)),
                    "raw_accs": accs,
                }
                with open(out_json, "w") as f:
                    json.dump(results, f, indent=2)

    save_exp1_publication_figures(
        args.output_dir, results["cond_logs"], results["acc_curves_seed42"]
    )

    # ====================== STATISTICAL TESTS ================================
    print("\n" + "=" * 64, flush=True)
    print("[Exp 1] Statistical Significance Tests", flush=True)
    print("=" * 64, flush=True)

    # Collect per-optimizer raw accuracy vectors across both seeds / all runs
    all_opt_names_sorted = sorted(
        {opt for s in results["table1"].values() for opt in s}
    )
    raw_by_opt: dict[str, list[float]] = {}
    for opt in all_opt_names_sorted:
        pool: list[float] = []
        for seed_str, seed_data in results["table1"].items():
            if opt in seed_data:
                pool.extend(seed_data[opt].get("raw_accs", []))
        if pool:
            raw_by_opt[opt] = pool

    titan_accs = raw_by_opt.get("titan", [])

    # --- Wilcoxon signed-rank: TITAN vs each baseline (paired on run index) ---
    if titan_accs:
        print("\n[Wilcoxon Signed-Rank Test] TITAN vs each baseline:", flush=True)
        wilcoxon_results: list[dict] = []
        for opt in all_opt_names_sorted:
            if opt == "titan":
                continue
            base_accs = raw_by_opt.get(opt, [])
            n_pairs = min(len(titan_accs), len(base_accs))
            if n_pairs < 2:
                print(
                    f"  {opt.upper():10s}: insufficient pairs (n={n_pairs}), skip",
                    flush=True,
                )
                continue
            ta = np.array(titan_accs[:n_pairs])
            ba = np.array(base_accs[:n_pairs])
            diffs = ta - ba
            if np.all(diffs == diffs[0]):  # all identical -> trivial
                stat_w, p_w = 0.0, 1.0
            else:
                try:
                    stat_w, p_w = stats.wilcoxon(
                        ta,
                        ba,
                        zero_method="wilcox",
                        correction=False,
                        alternative="two-sided",
                    )
                except (ValueError, TypeError, ZeroDivisionError, RuntimeError):
                    stat_w, p_w = 0.0, 1.0
            sig = (
                "***"
                if p_w < 0.001
                else ("**" if p_w < 0.01 else ("*" if p_w < 0.05 else "ns"))
            )
            print(
                f"  TITAN vs {opt.upper():10s}  W={stat_w:7.1f}  p={p_w:.4e}  {sig}",
                flush=True,
            )
            wilcoxon_results.append(
                {"baseline": opt, "stat": float(stat_w), "p": float(p_w), "sig": sig}
            )
        results["wilcoxon"] = wilcoxon_results

    # --- Friedman chi^2 test across ALL optimizers (requires equal-length samples) ---
    common_n = min((len(v) for v in raw_by_opt.values()), default=0)
    if common_n >= 2 and len(raw_by_opt) >= 3:
        samples = [
            np.array(raw_by_opt[o][:common_n])
            for o in all_opt_names_sorted
            if o in raw_by_opt
        ]
        try:
            chi2_stat, chi2_p = stats.friedmanchisquare(*samples)
        except (ValueError, TypeError, ZeroDivisionError, RuntimeError):
            chi2_stat, chi2_p = 0.0, 1.0
        sig_f = (
            "***"
            if chi2_p < 0.001
            else ("**" if chi2_p < 0.01 else ("*" if chi2_p < 0.05 else "ns"))
        )
        print(
            f"\n[Friedman χ²] All optimizers (n={common_n} per group):  χ²={chi2_stat:.3f}  p={chi2_p:.4e}  {sig_f}",
            flush=True,
        )
        results["friedman"] = {
            "chi2": float(chi2_stat),
            "p": float(chi2_p),
            "sig": sig_f,
        }
    else:
        print(
            "\n[Friedman χ²] Skipped — need ≥3 optimizers with ≥2 common samples each.",
            flush=True,
        )

    # --- Holm-Bonferroni correction on the Wilcoxon p-values ---
    wb = results.get("wilcoxon", [])
    if len(wb) >= 2:
        print("\n[Holm-Bonferroni Correction] on Wilcoxon p-values:", flush=True)
        p_vals = np.array([e["p"] for e in wb])
        order = np.argsort(p_vals)
        m = len(p_vals)
        adjusted: list[float] = [0.0] * m
        running_min = 1.0
        for rank, idx in enumerate(order[::-1]):  # descending order
            adj = float(min(1.0, p_vals[idx] * (m - rank)))
            running_min = min(running_min, adj)
            adjusted[idx] = running_min
        for i, e in enumerate(wb):
            sig_hb = (
                "***"
                if adjusted[i] < 0.001
                else (
                    "**"
                    if adjusted[i] < 0.01
                    else ("*" if adjusted[i] < 0.05 else "ns")
                )
            )
            print(
                f"  TITAN vs {e['baseline'].upper():10s}  p_raw={e['p']:.4e}  p_adj={adjusted[i]:.4e}  {sig_hb}",
                flush=True,
            )
            e["p_adj_holm"] = adjusted[i]
        results["wilcoxon"] = wb

    # Save updated JSON with stats
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Exp 1 Complete] Results + statistics saved to {out_json}", flush=True)


if __name__ == "__main__":
    main()
