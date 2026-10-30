"""
exp3_ablations_and_sensitivity.py
=================================
STANDALONE SCRIPT 3 (Kaggle 2xT4 GPU / Local Execution)
Paper: TITAN Optimizer (IEEE TPAMI)
Authors: Furqan Yaqub Khan and Tawseef Ayoub Shaikh, Member, IEEE

Zero external local file dependencies.
Executes:
  1. Table IV: Component Ablation Study on CIFAR-10 (ResNet-18) and FMNIST (LeNet-5):
     - TITAN (Full)
     - TITANAdaptive (Eq. 18, alpha = 0.5)
     - - Newton-Schulz
     - - RMS/Polar Norm
     - - Momentum
     - Muon Baseline
  2. Fig. 3(a): Trade-off across Newton-Schulz iteration depth K in {1, 2, 3, 4, 5, 7}.
  3. Fig. 3(b): Learning-rate robustness across eta in {1e-4, 3e-4, 1e-3, 3e-3} for AdamW, Muon, TITAN.
  4. Saves 600-DPI IEEE TPAMI Fig. 3 (PNG + Vector PDF) and Table IV JSON.
"""

from __future__ import annotations
import os
import sys
import time
import json
import math
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shutil, tarfile

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
        except Exception as e:
            print(f"[Device Guard] CUDA device initialization failed ({e}). Falling back to CPU.", flush=True)
            _CACHED_DEVICE = torch.device("cpu")
            return _CACHED_DEVICE
    _CACHED_DEVICE = torch.device("cpu")
    return _CACHED_DEVICE


# ====================== DATASETS ============================================
class OfflineStructuredVisionDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=12000, channels=3, img_size=32, num_classes=10, train=True, seed=42):
        super().__init__()
        rng = np.random.RandomState(seed + (0 if train else 999))
        protos = rng.normal(0, 0.6, size=(num_classes, channels, img_size, img_size)).astype(np.float32)
        labels = rng.randint(0, num_classes, size=(num_samples,), dtype=np.int64)
        noise = rng.normal(0, 0.35, size=(num_samples, channels, img_size, img_size)).astype(np.float32)
        self.images = torch.from_numpy(protos[labels] + noise)
        self.targets = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]


def safe_cifar10(data_dir: str, tr_tf, te_tf, seed: int = 42, quick: bool = False):
    c10_dir = os.path.join(data_dir, "cifar-10-batches-py")
    has_local = os.path.isdir(c10_dir) and any(f.startswith("data_batch") for f in os.listdir(c10_dir))
    if quick and not has_local:
        full_tr = OfflineStructuredVisionDataset(2000, 3, 32, 10, True, seed)
        return full_tr, full_tr, OfflineStructuredVisionDataset(500, 3, 32, 10, False, seed)
    try:
        full_tr = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=(not has_local), transform=tr_tf)
        val_base = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=False, transform=te_tf)
        te_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=(not has_local), transform=te_tf)
        return full_tr, val_base, te_set
    except Exception as e:
        print(f"\n[OFFLINE FALLBACK] CIFAR-10 download blocked ({type(e).__name__}: {e}). Using synthetic.", flush=True)
        full_tr = OfflineStructuredVisionDataset(12000, 3, 32, 10, True, seed)
        return full_tr, full_tr, OfflineStructuredVisionDataset(2000, 3, 32, 10, False, seed)


def safe_fmnist(data_dir: str, tr_tf, te_tf, seed: int = 42):
    raw_dir = os.path.join(data_dir, "FashionMNIST", "raw")
    has_local = os.path.isdir(raw_dir) and any(f.startswith("train-images") for f in os.listdir(raw_dir))
    try:
        full_tr = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=(not has_local), transform=tr_tf)
        val_base = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=False, transform=te_tf)
        te_set = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=(not has_local), transform=te_tf)
        return full_tr, val_base, te_set
    except Exception as e:
        print(f"\n[OFFLINE FALLBACK] FMNIST download blocked ({type(e).__name__}: {e}). Using synthetic.", flush=True)
        full_tr = OfflineStructuredVisionDataset(12000, 1, 28, 10, True, seed)
        return full_tr, full_tr, OfflineStructuredVisionDataset(2000, 1, 28, 10, False, seed)


# ====================== OPTIMIZERS ==========================================
def newton_schulz_orthogonalize(matrix: torch.Tensor, steps: int = 3, eps: float = 1e-7) -> torch.Tensor:
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
    def __init__(self, params, lr=1e-3, weight_decay=1e-4, beta=0.9, eps=1e-8, ns_steps=3):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay))
        self.beta = beta
        self.eps = eps
        self.ns_steps = ns_steps

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
                    update = newton_schulz_orthogonalize(mat, self.ns_steps).reshape_as(p)
                    update = update / (update.float().square().mean().sqrt() + self.eps)
                else:
                    update = st["mom"]
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)
        return loss


class TITANAdaptive(Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=1e-4, beta1=0.9, beta2=0.999, alpha=0.5, eps=1e-8, ns_steps=3):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay))
        self.beta1 = beta1
        self.beta2 = beta2
        self.alpha = alpha
        self.eps = eps
        self.ns_steps = ns_steps

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
                    st["var"] = torch.zeros_like(p)
                st["step"] += 1
                st["mom"].mul_(self.beta1).add_(g, alpha=1 - self.beta1)
                st["var"].mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)

                if p.ndim >= 2:
                    mat = st["mom"].reshape(st["mom"].shape[0], -1)
                    u_ns = newton_schulz_orthogonalize(mat, self.ns_steps).reshape_as(p)
                    u_ns = u_ns / (u_ns.float().square().mean().sqrt() + self.eps)
                else:
                    u_ns = st["mom"]

                bias_c2 = 1.0 - self.beta2 ** st["step"]
                u_adam = st["mom"] / (st["var"].div(bias_c2).sqrt().add_(self.eps))
                update = self.alpha * u_ns + (1.0 - self.alpha) * u_adam
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)
        return loss


class TITANAblation(Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=1e-4, beta=0.9, eps=1e-8, ns_steps=3,
                 use_ns=True, use_rms_norm=True, use_momentum=True):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay))
        self.beta = beta if use_momentum else 0.0
        self.eps = eps
        self.ns_steps = ns_steps
        self.use_ns = use_ns
        self.use_rms_norm = use_rms_norm
        self.use_momentum = use_momentum

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
                if self.use_momentum:
                    st["mom"].mul_(self.beta).add_(g, alpha=1 - self.beta)
                    base = st["mom"]
                else:
                    base = g.clone()

                if p.ndim >= 2 and self.use_ns:
                    mat = base.reshape(base.shape[0], -1)
                    update = newton_schulz_orthogonalize(mat, self.ns_steps).reshape_as(p)
                else:
                    update = base

                if self.use_rms_norm:
                    update = update / (update.float().square().mean().sqrt() + self.eps)

                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)
        return loss


class MuonBaseline(Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=1e-4, beta=0.95, ns_steps=5):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay))
        self.beta = beta
        self.ns_steps = ns_steps

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
                else:
                    update = st["mom"]
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)
        return loss


# ====================== MODELS ==============================================
class BasicBlock(nn.Module):
    def __init__(self, in_p, p, s=1):
        super().__init__()
        self.c1 = nn.Conv2d(in_p, p, 3, stride=s, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(p)
        self.c2 = nn.Conv2d(p, p, 3, stride=1, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(p)
        self.sc = nn.Sequential()
        if s != 1 or in_p != p:
            self.sc = nn.Sequential(nn.Conv2d(in_p, p, 1, stride=s, bias=False), nn.BatchNorm2d(p))

    def forward(self, x):
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x)), inplace=True))) + self.sc(x), inplace=True)


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
        return self.fc(F.adaptive_avg_pool2d(x, (1, 1)).flatten(1))


class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.c1 = nn.Conv2d(1, 32, 5, padding=2); self.b1 = nn.BatchNorm2d(32)
        self.c2 = nn.Conv2d(32, 64, 5, padding=0); self.b2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.b1(self.c1(x))), 2)
        x = F.max_pool2d(F.relu(self.b2(self.c2(x))), 2).flatten(1)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


# ====================== PUBLICATION FIGURE 3 ================================
def save_exp3_publication_figure(output_dir: str, k_sweep: dict, lr_sweep: dict):
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "stix", "font.size": 9.5,
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "figure.dpi": 600,
        "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.6, 5.5), dpi=600)
    fig.subplots_adjust(hspace=0.44, left=0.14, right=0.86, top=0.93, bottom=0.10)

    # Panel 3(a): K-depth sweep
    k_vals = np.array([1, 2, 3, 4, 5, 7])
    has_k = k_sweep and any(str(k) in k_sweep for k in k_vals)
    if has_k:
        avail_k = [k for k in k_vals if str(k) in k_sweep]
        acc_k = np.array([k_sweep[str(k)]["acc"] for k in avail_k])
        time_k = np.array([k_sweep[str(k)]["time_s"] for k in avail_k])

        ax1_r = ax1.twinx()
        l1 = ax1.plot(avail_k, acc_k, "-o", color="#00468B", lw=2.2, markersize=6, label="Test Accuracy (%)")
        l2 = ax1_r.plot(avail_k, time_k, "--s", color="#D55E00", lw=1.9, markersize=5.5, label="Runtime (s/epoch)")
        if 3 in avail_k:
            idx3 = avail_k.index(3)
            ax1.axvline(3, color="#009E73", ls="-.", lw=1.2)
            ax1.annotate(
                f"Default $K=3$\n({acc_k[idx3]:.2f}%, {time_k[idx3]:.1f}s)",
                xy=(3, acc_k[idx3]), xytext=(1.15, max(acc_k) - 0.22),
                fontsize=7.8, bbox=dict(boxstyle="round,pad=0.28", facecolor="#F4F7FB", edgecolor="#99B2CC"),
                arrowprops=dict(arrowstyle="->", color="#00468B", lw=1.0),
            )
        ax1.set_title(r"(a) Trade-off Across NS Depth $K$ (CIFAR-10)")
        ax1.set_xlabel(r"Newton-Schulz Iteration Depth $K$")
        ax1.set_ylabel("Top-1 Accuracy (%)", color="#00468B")
        ax1_r.set_ylabel("Runtime (s/epoch)", color="#D55E00")
        ax1.set_xticks(avail_k); ax1.grid(True, ls="--")
        lines = l1 + l2
        ax1.legend(lines, [l.get_label() for l in lines], loc="lower right", fontsize=7.8)
    else:
        ax1.text(0.5, 0.5, "K-depth sweep recorded during training", ha="center", va="center", transform=ax1.transAxes)

    # Panel 3(b): LR sensitivity
    lrs = [r"$10^{-4}$", r"$3\times 10^{-4}$", r"$10^{-3}$", r"$3\times 10^{-3}$"]
    x = np.arange(len(lrs)); w = 0.25
    has_lr = lr_sweep and all(k in lr_sweep for k in ["adamw", "muon", "titan"]) and all(len(lr_sweep[k]) == 4 for k in ["adamw", "muon", "titan"])
    if has_lr:
        a_aw = lr_sweep["adamw"]
        a_mu = lr_sweep["muon"]
        a_ti = lr_sweep["titan"]
        delta_mu = max(a_mu) - min(a_mu)
        delta_ti = max(a_ti) - min(a_ti)
        ax2.bar(x - w, a_aw, width=w, color="#009E73", alpha=0.88, label="AdamW")
        ax2.bar(x, a_mu, width=w, color="#D55E00", alpha=0.88, label=rf"Muon ($\Delta={delta_mu:.1f}\%$)")
        ax2.bar(x + w, a_ti, width=w, color="#00468B", alpha=0.95, label=rf"TITAN ($\Delta={delta_ti:.1f}\%$)")
        ax2.set_title("(b) Learning-Rate Robustness (ResNet-18)")
        ax2.set_xlabel(r"Base Learning Rate $\eta$")
        ax2.set_ylabel("CIFAR-10 Accuracy (%)")
        ax2.set_xticks(x); ax2.set_xticklabels(lrs)
        ax2.grid(True, ls="--", axis="y")
        ax2.legend(loc="upper center", ncol=3, fontsize=7.8)
    else:
        ax2.text(0.5, 0.5, "LR sweep recorded during training", ha="center", va="center", transform=ax2.transAxes)

    fig_png = os.path.join(output_dir, "Fig3_NS_Depth_Tradeoff_and_LR_Sensitivity.png")
    fig_pdf = os.path.join(output_dir, "Fig3_NS_Depth_Tradeoff_and_LR_Sensitivity.pdf")
    fig.savefig(fig_png, dpi=600); fig.savefig(fig_pdf)
    plt.close(fig)


# ====================== TRAIN / EVAL HELPER =================================
def train_and_eval(model, opt, tr_l, te_l, epochs, device, desc=""):
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    criterion = nn.CrossEntropyLoss()
    ep_secs = []
    for ep in range(epochs):
        model.train()
        t0 = time.perf_counter()
        loss_val = 0.0
        for x, y in tr_l:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            loss_val = loss.item()
        if device.type == "cuda": torch.cuda.synchronize()
        ep_secs.append(time.perf_counter() - t0)
        sched.step()
        print(f"  [{desc}] Epoch {ep+1}/{epochs} - Loss: {loss_val:.4f} | Time: {ep_secs[-1]:.2f}s", flush=True)

    model.eval()
    c, tot = 0, 0
    with torch.no_grad():
        for x, y in te_l:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            c += model(x).argmax(1).eq(y).sum().item(); tot += y.size(0)
    acc = 100.0 * c / max(1, tot)
    return acc, float(np.mean(ep_secs)) if ep_secs else 0.0


# ====================== MAIN EXECUTION ======================================
def main():
    parser = argparse.ArgumentParser(description="TITAN Exp 3: Table IV Ablations & Fig 3 Sweeps")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs_exp3")
    parser.add_argument("--epochs-c10", type=int, default=100)
    parser.add_argument("--epochs-fm", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args, _ = parser.parse_known_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, "table4_ablation_and_fig3_results.json")

    if args.plot_only:
        if os.path.exists(out_json):
            saved = json.load(open(out_json))
            save_exp3_publication_figure(args.output_dir, saved.get("k_sweep", {}), saved.get("lr_sweep", {}))
            print(f"[Exp 3] Figure 3 regenerated from {out_json}", flush=True)
        else:
            save_exp3_publication_figure(args.output_dir, {}, {})
        return

    epochs_c10 = 2 if args.quick else args.epochs_c10
    epochs_fm = 2 if args.quick else args.epochs_fm
    runs = 1 if args.quick else args.runs
    batch_size = 64 if args.quick else args.batch_size

    device = get_safe_device()
    num_gpus = torch.cuda.device_count() if (device.type == "cuda" and torch.cuda.is_available()) else 0
    print(f"[Exp 3] Device: {device} | GPUs: {num_gpus} | C10 Ep: {epochs_c10} | FM Ep: {epochs_fm}", flush=True)

    variants = [
        ("TITAN (Full)", lambda p, lr: TITAN(p, lr=lr, weight_decay=1e-4, ns_steps=3)),
        ("TITANAdaptive", lambda p, lr: TITANAdaptive(p, lr=lr, weight_decay=1e-4, ns_steps=3)),
        ("- Newton-Schulz", lambda p, lr: TITANAblation(p, lr=lr, weight_decay=1e-4, use_ns=False)),
        ("- RMS/Polar Norm", lambda p, lr: TITANAblation(p, lr=lr, weight_decay=1e-4, use_rms_norm=False)),
        ("- Momentum", lambda p, lr: TITANAblation(p, lr=lr, weight_decay=1e-4, use_momentum=False)),
        ("Muon Baseline", lambda p, lr: MuonBaseline(p, lr=lr, weight_decay=1e-4)),
    ]
    if args.quick:
        variants = [variants[0], variants[1], variants[5]]

    results = {"table4_fmnist": {}, "table4_cifar10": {}, "k_sweep": {}, "lr_sweep": {}}
    if os.path.exists(out_json):
        try: results = json.load(open(out_json))
        except Exception: pass

    # 1. Fashion-MNIST LeNet-5 Ablations (Table IV)
    mean_fm, std_fm = (0.2860,), (0.3530,)
    tr_fm, _, te_fm = safe_fmnist(
        args.data_dir,
        T.Compose([T.RandomCrop(28, padding=2), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean_fm, std_fm)]),
        T.Compose([T.ToTensor(), T.Normalize(mean_fm, std_fm)])
    )
    n_fm = len(tr_fm)
    sub_n_fm = 800 if args.quick else n_fm
    tr_l_fm = DataLoader(Subset(tr_fm, np.arange(sub_n_fm)), batch_size=batch_size, shuffle=True, num_workers=0)
    te_l_fm = DataLoader(Subset(te_fm, np.arange(min(400, len(te_fm)))) if args.quick else te_fm, batch_size=batch_size, shuffle=False, num_workers=0)

    for v_name, opt_fn in variants:
        accs, times = [], []
        for r in range(runs):
            torch.manual_seed(42 + r * 100)
            model = LeNet5(10).to(device)
            acc, sec = train_and_eval(model, opt_fn(model.parameters(), 1e-3), tr_l_fm, te_l_fm, epochs_fm, device, desc=f"FMNIST {v_name}")
            accs.append(acc); times.append(sec)
        results["table4_fmnist"][v_name] = {
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
            "time_s": float(np.mean(times)),
        }
        print(f"[Table IV FMNIST | {v_name}] Acc={np.mean(accs):.2f}% | Time={np.mean(times):.2f}s", flush=True)
        with open(out_json, "w") as f: json.dump(results, f, indent=2)

    # 2. CIFAR-10 ResNet-18 Ablations (Table IV)
    mean_c, std_c = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    tr_c10, _, te_c10 = safe_cifar10(
        args.data_dir,
        T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean_c, std_c)]),
        T.Compose([T.ToTensor(), T.Normalize(mean_c, std_c)]),
        quick=args.quick
    )
    n_c10 = len(tr_c10)
    sub_n_c10 = 128 if args.quick else n_c10
    tr_l_c10 = DataLoader(Subset(tr_c10, np.arange(sub_n_c10)), batch_size=batch_size, shuffle=True, num_workers=0)
    te_l_c10 = DataLoader(Subset(te_c10, np.arange(min(64, len(te_c10)))) if args.quick else te_c10, batch_size=batch_size, shuffle=False, num_workers=0)

    for v_name, opt_fn in variants:
        torch.manual_seed(42)
        model = ResNet18CIFAR(10).to(device)
        acc, sec = train_and_eval(model, opt_fn(model.parameters(), 1e-3), tr_l_c10, te_l_c10, epochs_c10, device, desc=f"C10 {v_name}")
        results["table4_cifar10"][v_name] = {"acc": acc, "time_s": sec}
        print(f"[Table IV CIFAR-10 | {v_name}] Acc={acc:.2f}% | Time={sec:.2f}s", flush=True)
        with open(out_json, "w") as f: json.dump(results, f, indent=2)

    # 3. K-sweep (Fig 3a)
    k_list = [1, 3] if args.quick else [1, 2, 3, 4, 5, 7]
    for k in k_list:
        torch.manual_seed(42)
        model = ResNet18CIFAR(10).to(device)
        opt = TITAN(model.parameters(), lr=1e-3, weight_decay=1e-4, ns_steps=k)
        acc, sec = train_and_eval(model, opt, tr_l_c10, te_l_c10, epochs_c10, device, desc=f"K={k}")
        results["k_sweep"][str(k)] = {"acc": acc, "time_s": sec}
        print(f"[Fig 3a K={k}] Acc={acc:.2f}% | Time={sec:.2f}s", flush=True)
        with open(out_json, "w") as f: json.dump(results, f, indent=2)

    # 4. LR-sweep (Fig 3b)
    lrs = [1e-4, 1e-3] if args.quick else [1e-4, 3e-4, 1e-3, 3e-3]
    sweep_opts = ["titan", "adamw"] if args.quick else ["titan", "muon", "adamw"]
    for opt_key in sweep_opts:
        results["lr_sweep"][opt_key] = []
        for lr_val in lrs:
            torch.manual_seed(42)
            model = ResNet18CIFAR(10).to(device)
            if opt_key == "titan": opt = TITAN(model.parameters(), lr=lr_val, weight_decay=1e-4)
            elif opt_key == "muon": opt = MuonBaseline(model.parameters(), lr=lr_val, weight_decay=1e-4)
            else: opt = torch.optim.AdamW(model.parameters(), lr=lr_val, weight_decay=1e-4)
            acc, _ = train_and_eval(model, opt, tr_l_c10, te_l_c10, epochs_c10, device, desc=f"{opt_key} lr={lr_val}")
            results["lr_sweep"][opt_key].append(acc)
            print(f"[Fig 3b LR Sweep | {opt_key.upper()} | lr={lr_val}] Acc={acc:.2f}%", flush=True)
        with open(out_json, "w") as f: json.dump(results, f, indent=2)

    save_exp3_publication_figure(args.output_dir, results["k_sweep"], results["lr_sweep"])
    print(f"\n[Exp 3 Complete] Results saved to {out_json}", flush=True)


if __name__ == "__main__":
    main()
