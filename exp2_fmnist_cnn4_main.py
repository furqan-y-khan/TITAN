"""
exp2_fmnist_cnn4_main.py
========================
STANDALONE SCRIPT 2 (Kaggle 2xT4 GPU / Local Execution)
Paper: TITAN Optimizer (IEEE TPAMI)
Authors: Furqan Yaqub Khan and Tawseef Ayoub Shaikh, Member, IEEE

Zero external local file dependencies.
Executes:
  1. Table II: Fashion-MNIST with CNN-4 across SGD+Mom, Adam, AdamW, L-BFGS, Muon, TITAN
     (5 independent runs, 100 epochs, early stopping patience=20).
  2. Fig. 2(b) & Full Combined Fig. 2:
     - (a) CIFAR-10 Top-1 Test Accuracy (loaded from Exp 1 real outcomes if present).
     - (b) Fashion-MNIST Training Cross-Entropy Loss curves directly from real training losses.
     - Saves 600-DPI IEEE TPAMI publication figures (PNG + Vector PDF).
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
            stat = float((12.0 / (n * k * (k + 1))) * np.sum(r_j ** 2) - 3.0 * n * (k + 1))
            df = k - 1
            z = (stat - df) / math.sqrt(2 * df) if df > 0 else 0.0
            p = float(max(1e-12, min(1.0, 0.5 * math.erfc(z / math.sqrt(2)))))
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
            print(f"[Device Guard] CUDA device initialization failed ({e}). Falling back to CPU.", flush=True)
            _CACHED_DEVICE = torch.device("cpu")
            return _CACHED_DEVICE
    _CACHED_DEVICE = torch.device("cpu")
    return _CACHED_DEVICE


# ====================== DATASETS ============================================
class OfflineStructuredVisionDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=12000, channels=1, img_size=28, num_classes=10, train=True, seed=42):
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


def safe_fmnist(data_dir: str, tr_tf, te_tf, seed: int = 42):
    raw_dir = os.path.join(data_dir, "FashionMNIST", "raw")
    has_local_raw = os.path.isdir(raw_dir) and any(f.startswith("train-images") for f in os.listdir(raw_dir))
    try:
        full_tr = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=(not has_local_raw), transform=tr_tf)
        val_base = torchvision.datasets.FashionMNIST(root=data_dir, train=True, download=False, transform=te_tf)
        te_set = torchvision.datasets.FashionMNIST(root=data_dir, train=False, download=(not has_local_raw), transform=te_tf)
        return full_tr, val_base, te_set
    except (RuntimeError, OSError, ValueError) as e:
        print(f"\n[OFFLINE FALLBACK] Fashion-MNIST download blocked ({type(e).__name__}: {e}). Using synthetic.", flush=True)
        full_tr = OfflineStructuredVisionDataset(12000, 1, 28, 10, True, seed)
        return full_tr, full_tr, OfflineStructuredVisionDataset(2000, 1, 28, 10, False, seed)


def get_fmnist_loaders(data_dir="./data", batch_size=128, seed=42, num_workers=0, quick=False):
    mean, std = (0.2860,), (0.3530,)
    tr_tf = T.Compose([T.RandomCrop(28, padding=2), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean, std)])
    te_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    full_tr, val_base, te_set = safe_fmnist(data_dir, tr_tf, te_tf, seed)
    n_tot = len(full_tr)
    idx = np.random.RandomState(seed).permutation(n_tot)

    if quick:
        tr_sub = Subset(full_tr, idx[:800])
        val_sub = Subset(val_base, idx[800:1000])
        te_sub = Subset(te_set, np.arange(min(400, len(te_set))))
    else:
        split = min(6000, int(0.1 * n_tot))
        tr_sub = Subset(full_tr, idx[split:])
        val_sub = Subset(val_base, idx[:split])
        te_sub = te_set

    tr_l = DataLoader(tr_sub, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    va_l = DataLoader(val_sub, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    te_l = DataLoader(te_sub, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return tr_l, va_l, te_l


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
    def __init__(self, params, lr=1e-3, weight_decay=0.01, beta=0.9, eps=1e-8, ns_steps=3):
        super().__init__(params, {"lr": lr, "weight_decay": weight_decay})
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


class MuonBaseline(Optimizer):
    def __init__(self, params, lr=1e-3, weight_decay=0.01, beta=0.95, ns_steps=5):
        super().__init__(params, {"lr": lr, "weight_decay": weight_decay})
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


class StochasticDampedLBFGS(Optimizer):
    def __init__(self, params, lr=1e-2, weight_decay=1e-4, history_size=20, damping=0.2, eps=1e-8):
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
                if p.grad is None: continue
                g = p.grad.detach().float()
                if wd: g = g.add(p.detach().float(), alpha=wd)
                p_list.append(p.detach().float().reshape(-1))
                g_list.append(g.reshape(-1))
        if not p_list: return loss
        flat_p, flat_g = torch.cat(p_list), torch.cat(g_list)
        if not torch.isfinite(flat_g).all(): return loss
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
                    self._s_hist.pop(0); self._y_hist.pop(0); self._rho_hist.pop(0)
                self._s_hist.append(s); self._y_hist.append(y); self._rho_hist.append(1.0 / (sy + self.eps))
        self._prev_p, self._prev_g = flat_p.clone(), flat_g.clone()
        q = flat_g.clone()
        m = len(self._s_hist)
        alphas = [0.0] * m
        for i in range(m - 1, -1, -1):
            alphas[i] = self._rho_hist[i] * torch.dot(self._s_hist[i], q)
            q.add_(self._y_hist[i], alpha=-alphas[i])
        if m > 0:
            last_y = self._y_hist[-1]
            gamma = torch.clamp(torch.dot(self._s_hist[-1], last_y) / (torch.dot(last_y, last_y) + self.eps), 1e-4, 10.0)
            r = q * gamma
        else:
            r = q
        for i in range(m):
            beta_i = self._rho_hist[i] * torch.dot(self._y_hist[i], r)
            r.add_(self._s_hist[i], alpha=alphas[i] - beta_i)
        r_rms = r.square().mean().sqrt() + self.eps
        if r_rms > 10.0: r = r * (10.0 / r_rms)
        offset = 0
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None: continue
                n = p.numel()
                p.add_(r[offset:offset + n].reshape_as(p).to(dtype=p.dtype), alpha=-lr)
                offset += n
        return loss


# ====================== CNN-4 ARCHITECTURE ==================================
class CNN4(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ====================== PUBLICATION FIGURE 2 GENERATOR ======================
def save_exp2_publication_figure(output_dir: str, loss_curves: dict, c10_curves: dict | None = None):
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "stix", "font.size": 9.5,
        "axes.titlesize": 10.5, "axes.titleweight": "bold", "figure.dpi": 600,
        "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.6, 5.5), dpi=600)
    fig.subplots_adjust(hspace=0.42, left=0.14, right=0.96, top=0.93, bottom=0.10)

    colors = {"sgd": "#666666", "adam": "#7570B3", "adamw": "#009E73", "muon": "#D55E00", "titan": "#00468B", "lbfgs": "#E7298A"}
    styles = {"sgd": ":", "adam": "--", "adamw": "-.", "muon": "-", "titan": "-", "lbfgs": "-."}

    if c10_curves and any(len(c) > 0 for c in c10_curves.values()):
        for k, curve in c10_curves.items():
            if len(curve) > 0:
                ax1.plot(np.arange(1, len(curve) + 1), curve, styles.get(k.lower(), "-"),
                         color=colors.get(k.lower(), "#333333"),
                         lw=(2.3 if k.lower() == "titan" else 1.5),
                         label=f"{k.upper()} ({max(curve):.2f}%)")
        ax1.legend(loc="lower right", fontsize=7.8)
    else:
        ax1.text(0.5, 0.5, "CIFAR-10 curves loaded from Exp 1 results", ha="center", va="center", transform=ax1.transAxes)

    ax1.axhline(90.0, color="#666666", ls="--", lw=1.0)
    ax1.set_title("(a) CIFAR-10 Test Accuracy (ResNet-18)")
    ax1.set_xlabel("Training Epochs"); ax1.set_ylabel("Top-1 Test Accuracy (%)")
    ax1.grid(True, ls="--")

    if loss_curves and any(len(c) > 0 for c in loss_curves.values()):
        for k, curve in loss_curves.items():
            if len(curve) > 0:
                ax2.plot(np.arange(1, len(curve) + 1), curve, styles.get(k.lower(), "-"),
                         color=colors.get(k.lower(), "#333333"),
                         lw=(2.3 if k.lower() == "titan" else 1.5),
                         label=f"{k.upper()} (loss={min(curve):.3f})")
        ax2.set_title("(b) Fashion-MNIST Cross-Entropy Loss (CNN-4)")
        ax2.set_xlabel("Training Epochs"); ax2.set_ylabel("Training Cross-Entropy Loss")
        ax2.grid(True, ls="--")
        ax2.legend(loc="upper right", fontsize=7.8)
    else:
        ax2.text(0.5, 0.5, "Loss curves logged dynamically during training", ha="center", va="center", transform=ax2.transAxes)

    fig_png = os.path.join(output_dir, "Fig2_CIFAR10_Acc_and_FMNIST_Loss.png")
    fig_pdf = os.path.join(output_dir, "Fig2_CIFAR10_Acc_and_FMNIST_Loss.pdf")
    fig.savefig(fig_png, dpi=600); fig.savefig(fig_pdf)
    plt.close(fig)

    # Standalone Fig. 2(b): Fashion-MNIST Training Loss Curves
    fig2b, ax_b = plt.subplots(1, 1, figsize=(4.6, 3.0), dpi=600)
    fig2b.subplots_adjust(left=0.14, right=0.96, top=0.88, bottom=0.18)
    if loss_curves and any(len(c) > 0 for c in loss_curves.values()):
        for k, curve in loss_curves.items():
            if len(curve) > 0:
                ax_b.plot(np.arange(1, len(curve) + 1), curve, styles.get(k.lower(), "-"),
                          color=colors.get(k.lower(), "#333333"),
                          lw=(2.3 if k.lower() == "titan" else 1.5),
                          label=f"{k.upper()} (loss={min(curve):.3f})")
        ax_b.set_title("Fashion-MNIST Cross-Entropy Loss (CNN-4)")
        ax_b.set_xlabel("Training Epochs"); ax_b.set_ylabel("Training Cross-Entropy Loss")
        ax_b.grid(True, ls="--")
        ax_b.legend(loc="upper right", fontsize=7.8)
    else:
        ax_b.text(0.5, 0.5, "Loss curves logged dynamically during training", ha="center", va="center", transform=ax_b.transAxes)
    fig2b.savefig(os.path.join(output_dir, "Fig2b_FashionMNIST_Loss_CNN4.png"), dpi=600)
    fig2b.savefig(os.path.join(output_dir, "Fig2b_FashionMNIST_Loss_CNN4.pdf"))
    plt.close(fig2b)

    # Standalone Fig. 2(a) if CIFAR-10 data was loaded
    if c10_curves and any(len(c) > 0 for c in c10_curves.values()):
        fig2a, ax_a = plt.subplots(1, 1, figsize=(4.6, 3.0), dpi=600)
        fig2a.subplots_adjust(left=0.14, right=0.96, top=0.88, bottom=0.18)
        for k, curve in c10_curves.items():
            if len(curve) > 0:
                ax_a.plot(np.arange(1, len(curve) + 1), curve, styles.get(k.lower(), "-"),
                          color=colors.get(k.lower(), "#333333"),
                          lw=(2.3 if k.lower() == "titan" else 1.5),
                          label=f"{k.upper()} ({max(curve):.2f}%)")
        ax_a.axhline(90.0, color="#666666", ls="--", lw=1.0)
        ax_a.set_title("(a) CIFAR-10 Test Accuracy (ResNet-18)")
        ax_a.set_xlabel("Training Epochs"); ax_a.set_ylabel("Top-1 Test Accuracy (%)")
        ax_a.grid(True, ls="--")
        ax_a.legend(loc="lower right", fontsize=7.8)
        fig2a.savefig(os.path.join(output_dir, "Fig2a_CIFAR10_Test_Accuracy_ResNet18.png"), dpi=600)
        fig2a.savefig(os.path.join(output_dir, "Fig2a_CIFAR10_Test_Accuracy_ResNet18.pdf"))
        plt.close(fig2a)


# ====================== MAIN EXECUTION ======================================
def main():
    parser = argparse.ArgumentParser(description="TITAN Exp 2: Fashion-MNIST CNN-4 Benchmark & Fig 2")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs_exp2")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args, _ = parser.parse_known_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, "table2_fmnist_cnn4_results.json")

    exp1_json = os.path.join(os.path.dirname(args.output_dir), "outputs_exp1", "table1_cifar10_resnet18_results.json")
    c10_curves = {}
    if os.path.exists(exp1_json):
        try:
            with open(exp1_json, encoding="utf-8") as f:
                c10_curves = json.load(f).get("acc_curves_seed42", {})
        except (json.JSONDecodeError, OSError):
            c10_curves = {}

    if args.plot_only:
        if os.path.exists(out_json):
            with open(out_json, encoding="utf-8") as f:
                saved = json.load(f)
            save_exp2_publication_figure(args.output_dir, saved.get("loss_curves", {}), c10_curves)
            print(f"[Exp 2] Figure 2 regenerated from {out_json}", flush=True)
        else:
            save_exp2_publication_figure(args.output_dir, {}, c10_curves)
        return

    epochs = 2 if args.quick else args.epochs
    runs = 1 if args.quick else args.runs
    batch_size = 64 if args.quick else args.batch_size
    optimizers = ["titan", "adamw"] if args.quick else ["titan", "muon", "adamw", "adam", "sgd", "lbfgs"]

    device = get_safe_device()
    num_gpus = torch.cuda.device_count() if (device.type == "cuda" and torch.cuda.is_available()) else 0
    print(f"[Exp 2] Device: {device} | GPUs: {num_gpus} | Epochs: {epochs} | Runs: {runs}", flush=True)

    results = {"table2": {}, "loss_curves": {}}
    if os.path.exists(out_json):
        try:
            with open(out_json, encoding="utf-8") as f:
                results = json.load(f)
        except (json.JSONDecodeError, OSError):
            results = {"table2": {}, "loss_curves": {}}

    for opt_name in optimizers:
        accs, times, mems = [], [], []
        for r in range(runs):
            seed = 42 + r * 100
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.benchmark = True
                for g_i in range(num_gpus): torch.cuda.reset_peak_memory_stats(g_i)

            tr_l, va_l, te_l = get_fmnist_loaders(args.data_dir, batch_size, seed, num_workers=0, quick=args.quick)
            model = CNN4(10).to(device)
            if num_gpus > 1: model = nn.DataParallel(model, device_ids=list(range(num_gpus)))

            if opt_name == "titan": opt = TITAN(model.parameters(), lr=1e-3, weight_decay=1e-4)
            elif opt_name == "muon": opt = MuonBaseline(model.parameters(), lr=1e-3, weight_decay=1e-4)
            elif opt_name == "sgd": opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
            elif opt_name == "adam": opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            elif opt_name == "adamw": opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            else: opt = StochasticDampedLBFGS(model.parameters(), lr=1e-2, weight_decay=1e-4)

            sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda e: (e + 1) / 5.0 if e < 5 else 0.5 * (1.0 + math.cos(math.pi * (e - 5) / max(1, epochs - 5)))
            )
            scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
            criterion = nn.CrossEntropyLoss()

            best_val, patience = 0.0, 0
            ep_secs, tr_losses = [], []

            for ep in range(epochs):
                model.train()
                t0 = time.perf_counter()
                loss_sum, n_tot = 0.0, 0
                for x, y in tr_l:
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                        loss = criterion(model(x), y)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                    loss_sum += loss.item() * x.size(0); n_tot += x.size(0)

                if device.type == "cuda": torch.cuda.synchronize()
                ep_secs.append(time.perf_counter() - t0)
                mean_loss = float(loss_sum / max(1, n_tot))
                tr_losses.append(mean_loss)
                sched.step()

                model.eval()
                c_val, t_val = 0, 0
                with torch.no_grad():
                    for x, y in va_l:
                        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                        c_val += model(x).argmax(1).eq(y).sum().item(); t_val += y.size(0)
                val_acc = 100.0 * c_val / max(1, t_val)
                print(f"  [Exp 2 | {opt_name.upper()} | Run {r+1}] Epoch {ep+1}/{epochs} - Tr Loss: {mean_loss:.4f} | Val Acc: {val_acc:.2f}% | Step: {ep_secs[-1]:.2f}s", flush=True)

                if val_acc > best_val: best_val, patience = val_acc, 0
                else:
                    patience += 1
                    if patience >= 20 and not args.quick: break

            model.eval()
            c_te, t_te = 0, 0
            with torch.no_grad():
                for x, y in te_l:
                    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                    c_te += model(x).argmax(1).eq(y).sum().item(); t_te += y.size(0)
            te_acc = 100.0 * c_te / max(1, t_te)
            mem_gb = sum(torch.cuda.max_memory_allocated(i) for i in range(num_gpus)) / (1024 ** 3) if num_gpus else 0.0

            accs.append(te_acc); times.append(float(np.mean(ep_secs)) if ep_secs else 0.0)
            mems.append(mem_gb)
            if r == 0: results["loss_curves"][opt_name] = tr_losses

            print(f"[Table II | Run {r+1} | {opt_name.upper()}] Acc={te_acc:.2f}% | Time={np.mean(ep_secs):.2f}s/ep", flush=True)

        if accs:
            results["table2"][opt_name] = {
                "acc_mean": float(np.mean(accs)),
                "acc_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
                "time_s": float(np.mean(times)),
                "mem_gb": float(np.mean(mems)),
                "raw_accs": accs,
            }
            with open(out_json, "w") as f: json.dump(results, f, indent=2)

    save_exp2_publication_figure(args.output_dir, results["loss_curves"], c10_curves)

    # ====================== STATISTICAL TESTS ================================
    print("\n" + "=" * 64, flush=True)
    print("[Exp 2] Fashion-MNIST Statistical Significance Tests", flush=True)
    print("=" * 64, flush=True)

    all_opts = sorted(results["table2"])
    raw_by_opt = {o: results["table2"][o].get("raw_accs", []) for o in all_opts}
    titan_accs = raw_by_opt.get("titan", [])

    if titan_accs:
        print("\n[Wilcoxon Signed-Rank Test] TITAN vs each baseline (Table II):", flush=True)
        wilcoxon_results = []
        for opt in all_opts:
            if opt == "titan":
                continue
            base_accs = raw_by_opt.get(opt, [])
            n_pairs = min(len(titan_accs), len(base_accs))
            if n_pairs < 2:
                continue
            ta = np.array(titan_accs[:n_pairs])
            ba = np.array(base_accs[:n_pairs])
            try:
                stat_w, p_w = stats.wilcoxon(ta, ba, zero_method="wilcox", correction=False, alternative="two-sided")
            except (ValueError, TypeError, ZeroDivisionError, RuntimeError):
                stat_w, p_w = 0.0, 1.0
            sig = "***" if p_w < 0.001 else ("**" if p_w < 0.01 else ("*" if p_w < 0.05 else "ns"))
            print(f"  TITAN vs {opt.upper():10s}  W={stat_w:7.1f}  p={p_w:.4e}  {sig}", flush=True)
            wilcoxon_results.append({"baseline": opt, "stat": float(stat_w), "p": float(p_w), "sig": sig})
        results["wilcoxon"] = wilcoxon_results

    # Friedman chi^2
    common_n = min((len(v) for v in raw_by_opt.values()), default=0)
    if common_n >= 2 and len(raw_by_opt) >= 3:
        samples = [np.array(raw_by_opt[o][:common_n]) for o in all_opts]
        try:
            chi2_stat, chi2_p = stats.friedmanchisquare(*samples)
        except (ValueError, TypeError, ZeroDivisionError, RuntimeError):
            chi2_stat, chi2_p = 0.0, 1.0
        sig_f = "***" if chi2_p < 0.001 else ("**" if chi2_p < 0.01 else ("*" if chi2_p < 0.05 else "ns"))
        print(f"\n[Friedman χ²] All optimizers (n={common_n} per group):  χ²={chi2_stat:.3f}  p={chi2_p:.4e}  {sig_f}", flush=True)
        results["friedman"] = {"chi2": float(chi2_stat), "p": float(chi2_p), "sig": sig_f}

    # Holm-Bonferroni correction
    wb = results.get("wilcoxon", [])
    if len(wb) >= 2:
        print("\n[Holm-Bonferroni Correction] on Wilcoxon p-values:", flush=True)
        p_vals = np.array([e["p"] for e in wb])
        order = np.argsort(p_vals)
        m = len(p_vals)
        adjusted = [0.0] * m
        running_min = 1.0
        for rank, idx in enumerate(order[::-1]):
            adj = float(min(1.0, p_vals[idx] * (m - rank)))
            running_min = min(running_min, adj)
            adjusted[idx] = running_min
        for i, e in enumerate(wb):
            sig_hb = "***" if adjusted[i] < 0.001 else ("**" if adjusted[i] < 0.01 else ("*" if adjusted[i] < 0.05 else "ns"))
            print(f"  TITAN vs {e['baseline'].upper():10s}  p_raw={e['p']:.4e}  p_adj={adjusted[i]:.4e}  {sig_hb}", flush=True)
            e["p_adj_holm"] = adjusted[i]
        results["wilcoxon"] = wb

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Exp 2 Complete] Results + statistics saved to {out_json}", flush=True)


if __name__ == "__main__":
    main()
