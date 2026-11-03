"""
TITAN Optimizer Hyperparameter Optimization (CIFAR-10 & Fashion-MNIST)
======================================================================
Kaggle-ready script using Optuna (TPE Sampler + Median Pruner) to find the
optimal learning rate and hyperparameters (`lr`, `weight_decay`, `beta`,
`ns_steps`, `eps`, `batch_size`, and LR schedule) for the TITAN optimizer on
both Fashion-MNIST and CIFAR-10, followed by full retraining, test evaluation,
and diagnostic plotting.

Compatible with:
- Kaggle Notebooks (paste directly into a cell; safely ignores Jupyter CLI flags)
- Command-line execution (`python titan_optuna_kaggle.py --trials 25 --search-epochs 10`)
"""

import argparse
import copy
import json
import math
import os
import random
import subprocess
import sys
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as T

# Ensure Optuna is available (pre-installed on Kaggle, fallback auto-install)
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "optuna"])
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler


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


# ====================== REPRODUCIBILITY & UTILITIES =========================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def get_default_dirs() -> Tuple[str, str]:
    """Detect Kaggle environment paths automatically."""
    if os.path.exists("/kaggle/working"):
        data_dir = "/kaggle/working/data"
        out_dir = "/kaggle/working/titan_optuna_results"
    else:
        data_dir = "./data"
        out_dir = "./titan_optuna_results"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    return data_dir, out_dir


# ====================== MODEL ARCHITECTURES =================================
def conv_bn_act(in_c: int, out_c: int, pool: bool = False) -> nn.Sequential:
    layers = [
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.GELU(),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    """
    Fast, high-accuracy ResNet-9 architecture tailored for 28x28 (FMNIST)
    and 32x32 (CIFAR-10) images. Ideal for rapid hyperparameter sweeps.
    """

    def __init__(
        self, in_channels: int = 3, num_classes: int = 10, dropout: float = 0.1
    ):
        super().__init__()
        self.conv1 = conv_bn_act(in_channels, 64)
        self.conv2 = conv_bn_act(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_bn_act(128, 128), conv_bn_act(128, 128))

        self.conv3 = conv_bn_act(128, 256, pool=True)
        self.conv4 = conv_bn_act(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_bn_act(512, 512), conv_bn_act(512, 512))

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        return self.classifier(out)


def build_model(dataset_name: str, arch: str = "resnet9") -> nn.Module:
    in_channels = 1 if dataset_name.lower() == "fmnist" else 3
    if arch.lower() == "resnet18":
        model = torchvision.models.resnet18(weights=None, num_classes=10)
        # Adapt stem for small 28x28 / 32x32 images so spatial resolution is preserved
        model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = nn.Identity()
        return model
    return ResNet9(in_channels=in_channels, num_classes=10)


# ====================== DATASET & DATALOADERS ===============================
def load_datasets(dataset_name: str, data_dir: str, smoke_test: bool = False):
    name = dataset_name.lower()
    if smoke_test:
        img_size = (1, 28, 28) if name == "fmnist" else (3, 32, 32)
        train_ds = torchvision.datasets.FakeData(
            size=256, image_size=img_size, num_classes=10, transform=T.ToTensor()
        )
        test_ds = torchvision.datasets.FakeData(
            size=64, image_size=img_size, num_classes=10, transform=T.ToTensor()
        )
        return train_ds, test_ds

    if name == "fmnist":
        mean, std = (0.2860,), (0.3530,)
        train_tf = T.Compose(
            [
                T.RandomCrop(28, padding=3),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )
        test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        train_ds = torchvision.datasets.FashionMNIST(
            root=data_dir, train=True, download=True, transform=train_tf
        )
        test_ds = torchvision.datasets.FashionMNIST(
            root=data_dir, train=False, download=True, transform=test_tf
        )
    elif name == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        train_tf = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )
        test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        train_ds = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=train_tf
        )
        test_ds = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=test_tf
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return train_ds, test_ds


def create_search_loaders(
    train_ds,
    batch_size: int,
    subset_fraction: float = 1.0,
    val_split: float = 0.1,
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation splits from the training set so the official
    test set remains strictly unseen during Optuna hyperparameter tuning.
    """
    n_total = len(train_ds)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)

    if 0.0 < subset_fraction < 1.0:
        n_keep = max(128, int(n_total * subset_fraction))
        indices = indices[:n_keep]

    n_val = max(64, int(len(indices) * val_split))
    val_idx = indices[:n_val].tolist()
    train_idx = indices[n_val:].tolist()

    # Use deterministic normalization-only transforms on the validation split
    val_base_ds = copy.copy(train_ds)
    if hasattr(train_ds, "transform") and hasattr(train_ds.transform, "transforms"):
        norm_transforms = [
            t
            for t in train_ds.transform.transforms
            if isinstance(t, (T.ToTensor, T.Normalize))
        ]
        if norm_transforms:
            val_base_ds.transform = T.Compose(norm_transforms)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        Subset(val_base_ds, val_idx),
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


# ====================== TRAINING & EVALUATION LOOPS =========================
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: TITAN,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    scheduler=None,
    step_scheduler_per_batch: bool = False,
    use_amp: bool = True,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    amp_enabled = use_amp and (device.type == "cuda")

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        if not torch.isfinite(loss):
            return float("nan"), 0.0

        scaler.scale(loss).backward()
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()

        # Only advance per-batch scheduler if GradScaler did not skip optimizer.step()
        step_skipped = amp_enabled and (scaler.get_scale() < scale_before)
        if scheduler is not None and step_scheduler_per_batch and not step_skipped:
            scheduler.step()

        batch_sz = targets.size(0)
        running_loss += loss.item() * batch_sz
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += batch_sz

    avg_loss = running_loss / max(1, total)
    accuracy = 100.0 * correct / max(1, total)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    amp_enabled = use_amp and (device.type == "cuda")

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        batch_sz = targets.size(0)
        running_loss += loss.item() * batch_sz
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += batch_sz

    avg_loss = running_loss / max(1, total)
    accuracy = 100.0 * correct / max(1, total)
    return avg_loss, accuracy


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


# ====================== OPTUNA OBJECTIVE & STUDY ============================
def run_optuna_study_for_dataset(
    dataset_name: str,
    data_dir: str,
    out_dir: str,
    device: torch.device,
    arch: str = "resnet9",
    n_trials: int = 20,
    search_epochs: int = 10,
    final_epochs: int = 25,
    subset_fraction: float = 1.0,
    num_workers: int = 2,
    use_amp: bool = True,
    seed: int = 42,
    smoke_test: bool = False,
) -> Dict:
    print(f"\n{'=' * 74}")
    print(f"Starting TITAN Hyperparameter Optimization for {dataset_name.upper()}")
    print(
        f"Device: {device} | Arch: {arch} | Trials: {n_trials} | "
        f"Search Epochs: {search_epochs} | AMP: {use_amp and device.type == 'cuda'}"
    )
    print(f"{'=' * 74}")

    train_ds, test_ds = load_datasets(dataset_name, data_dir, smoke_test=smoke_test)
    amp_enabled = use_amp and (device.type == "cuda")

    def objective(trial: optuna.Trial) -> float:
        set_seed(seed + trial.number)

        # ---- TITAN Hyperparameter Search Space ----
        lr = trial.suggest_float("lr", 2e-4, 3e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        beta = trial.suggest_float("beta", 0.80, 0.98)
        ns_steps = trial.suggest_int("ns_steps", 2, 5)
        eps = trial.suggest_categorical("eps", [1e-8, 1e-7, 1e-6])
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        scheduler_type = trial.suggest_categorical(
            "scheduler", ["cosine", "onecycle", "none"]
        )
        label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.1)

        train_loader, val_loader = create_search_loaders(
            train_ds,
            batch_size=batch_size,
            subset_fraction=subset_fraction,
            val_split=0.1,
            num_workers=num_workers,
            seed=seed,
        )

        model = build_model(dataset_name, arch=arch).to(device)
        optimizer = TITAN(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            beta=beta,
            eps=eps,
            ns_steps=ns_steps,
        )
        scheduler, step_per_batch = build_scheduler(
            optimizer, scheduler_type, search_epochs, len(train_loader), lr
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        best_val_acc = 0.0
        for epoch in range(search_epochs):
            train_loss, _ = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                scaler=scaler,
                scheduler=scheduler,
                step_scheduler_per_batch=step_per_batch,
                use_amp=use_amp,
            )
            if math.isnan(train_loss):
                raise optuna.TrialPruned("NaN training loss encountered.")

            if scheduler is not None and not step_per_batch:
                scheduler.step()

            _, val_acc = evaluate(model, val_loader, criterion, device, use_amp=use_amp)
            best_val_acc = max(best_val_acc, val_acc)

            # Report intermediate validation accuracy to Optuna pruner
            trial.report(val_acc, step=epoch)
            if trial.should_prune():
                print(
                    f"  [Trial {trial.number + 1:02d}/{n_trials:02d}] "
                    f"Pruned at epoch {epoch + 1}/{search_epochs} (Val Acc: {val_acc:.2f}%)"
                )
                raise optuna.TrialPruned()

        trial.set_user_attr("cond_num", float(optimizer.last_condition_number))
        trial.set_user_attr(
            "unnorm_cond_num", float(optimizer.last_unnorm_condition_number)
        )
        print(
            f"  [Trial {trial.number + 1:02d}/{n_trials:02d}] "
            f"Best Val Acc: {best_val_acc:.2f}% | lr={lr:.5f}, wd={weight_decay:.2e}, "
            f"beta={beta:.3f}, ns={ns_steps}, bs={batch_size}, sched={scheduler_type}"
        )
        return best_val_acc

    sampler = TPESampler(seed=seed, multivariate=True)
    pruner = MedianPruner(
        n_startup_trials=max(3, n_trials // 4),
        n_warmup_steps=max(2, search_epochs // 3),
    )

    study = optuna.create_study(
        study_name=f"titan_{dataset_name.lower()}",
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    print(
        f"\n[{dataset_name.upper()}] Best Validation Accuracy: {study.best_value:.2f}%"
    )
    print(f"[{dataset_name.upper()}] Best TITAN Hyperparameters:")
    for k, v in best_params.items():
        print(f"  - {k}: {v}")

    # Save Optuna trials history to CSV if pandas is installed
    try:
        trials_df = study.trials_dataframe()
        csv_path = os.path.join(out_dir, f"optuna_trials_{dataset_name.lower()}.csv")
        trials_df.to_csv(csv_path, index=False)
        print(f"[{dataset_name.upper()}] Saved trial history to {csv_path}")
    except Exception as exc:
        print(f"[{dataset_name.upper()}] Skipped CSV export: {exc}")

    # ---- Final Retraining on Full Training Dataset & Test Set Evaluation ----
    print(
        f"\n[{dataset_name.upper()}] Training final model with best hyperparameters "
        f"for {final_epochs} epochs..."
    )
    set_seed(seed)
    pin_memory = torch.cuda.is_available()
    full_train_loader = DataLoader(
        train_ds,
        batch_size=best_params["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=best_params["batch_size"] * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    final_model = build_model(dataset_name, arch=arch).to(device)
    final_optimizer = TITAN(
        final_model.parameters(),
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
        beta=best_params["beta"],
        eps=best_params["eps"],
        ns_steps=best_params["ns_steps"],
    )
    final_scheduler, step_per_batch = build_scheduler(
        final_optimizer,
        best_params["scheduler"],
        final_epochs,
        len(full_train_loader),
        best_params["lr"],
    )
    final_criterion = nn.CrossEntropyLoss(
        label_smoothing=best_params["label_smoothing"]
    )
    eval_criterion = nn.CrossEntropyLoss()
    final_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_test_acc = 0.0
    history = []

    for epoch in range(1, final_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            final_model,
            full_train_loader,
            final_optimizer,
            final_criterion,
            device,
            scaler=final_scaler,
            scheduler=final_scheduler,
            step_scheduler_per_batch=step_per_batch,
            use_amp=use_amp,
        )
        if final_scheduler is not None and not step_per_batch:
            final_scheduler.step()

        te_loss, te_acc = evaluate(
            final_model, test_loader, eval_criterion, device, use_amp=use_amp
        )
        elapsed = time.time() - t0

        if te_acc > best_test_acc:
            best_test_acc = te_acc
            ckpt_path = os.path.join(
                out_dir, f"titan_best_{dataset_name.lower()}_model.pt"
            )
            torch.save(
                {
                    "dataset": dataset_name,
                    "arch": arch,
                    "best_params": best_params,
                    "test_accuracy": best_test_acc,
                    "model_state_dict": final_model.state_dict(),
                },
                ckpt_path,
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(tr_loss, 4),
                "train_acc": round(tr_acc, 2),
                "test_loss": round(te_loss, 4),
                "test_acc": round(te_acc, 2),
                "cond_num": round(final_optimizer.last_condition_number, 3),
                "unnorm_cond_num": round(
                    final_optimizer.last_unnorm_condition_number, 3
                ),
            }
        )
        print(
            f"  Epoch {epoch:02d}/{final_epochs:02d} ({elapsed:.1f}s) | "
            f"Train Loss: {tr_loss:.4f} ({tr_acc:.2f}%) | "
            f"Test Acc: {te_acc:.2f}% (Best: {best_test_acc:.2f}%) | "
            f"NS Cond: {final_optimizer.last_condition_number:.2f} "
            f"(Raw: {final_optimizer.last_unnorm_condition_number:.1f})"
        )

    return {
        "dataset": dataset_name,
        "best_validation_accuracy": round(float(study.best_value), 2),
        "best_test_accuracy": round(float(best_test_acc), 2),
        "best_hyperparameters": best_params,
        "final_condition_number": round(
            float(final_optimizer.last_condition_number), 4
        ),
        "final_unnorm_condition_number": round(
            float(final_optimizer.last_unnorm_condition_number), 4
        ),
        "training_history": history,
    }


def save_training_plots(summary_results: Dict, out_dir: str) -> None:
    """Save accuracy and condition number curves for each dataset."""
    try:
        import matplotlib.pyplot as plt

        n_ds = len(summary_results)
        if n_ds == 0:
            return
        fig, axes = plt.subplots(n_ds, 2, figsize=(12, 4.5 * n_ds), squeeze=False)
        for idx, (ds_name, info) in enumerate(summary_results.items()):
            hist = info["training_history"]
            epochs = [h["epoch"] for h in hist]
            tr_acc = [h["train_acc"] for h in hist]
            te_acc = [h["test_acc"] for h in hist]
            cond = [h["cond_num"] for h in hist]

            ax_acc = axes[idx, 0]
            ax_acc.plot(epochs, tr_acc, label="Train Accuracy (%)", linewidth=2)
            ax_acc.plot(epochs, te_acc, label="Test Accuracy (%)", linewidth=2)
            ax_acc.set_title(
                f"{ds_name.upper()} Accuracy (Best Test: {info['best_test_accuracy']:.2f}%)"
            )
            ax_acc.set_xlabel("Epoch")
            ax_acc.set_ylabel("Accuracy (%)")
            ax_acc.grid(True, alpha=0.3)
            ax_acc.legend()

            ax_cond = axes[idx, 1]
            ax_cond.plot(
                epochs,
                cond,
                color="tab:purple",
                label="Post-NS Condition Number",
                linewidth=2,
            )
            ax_cond.set_title(
                f"{ds_name.upper()} TITAN Orthogonalized Condition Number"
            )
            ax_cond.set_xlabel("Epoch")
            ax_cond.set_ylabel("Condition Number")
            ax_cond.grid(True, alpha=0.3)
            ax_cond.legend()

        plt.tight_layout()
        plot_path = os.path.join(out_dir, "titan_training_curves.png")
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Saved training and condition-number plots to: {plot_path}")
    except Exception as exc:
        print(f"Plot generation skipped: {exc}")


# ====================== MAIN ENTRYPOINT =====================================
def parse_args():
    default_data_dir, default_out_dir = get_default_dirs()
    parser = argparse.ArgumentParser(
        description="Tune TITAN Optimizer on Fashion-MNIST and CIFAR-10 with Optuna"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["fmnist", "cifar10"],
        choices=["fmnist", "cifar10"],
        help="Datasets to optimize TITAN on (default: both fmnist and cifar10).",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="resnet9",
        choices=["resnet9", "resnet18"],
        help="Model architecture (resnet9 is fast and high-accuracy).",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of Optuna trials per dataset (default: 20).",
    )
    parser.add_argument(
        "--search-epochs",
        type=int,
        default=10,
        help="Number of training epochs per Optuna trial (default: 10).",
    )
    parser.add_argument(
        "--final-epochs",
        type=int,
        default=25,
        help="Number of epochs for final retraining with best hyperparameters.",
    )
    parser.add_argument(
        "--subset-fraction",
        type=float,
        default=1.0,
        help="Fraction of training data used during Optuna search (e.g., 0.5 for faster sweeps).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="DataLoader worker processes (2 is recommended on Kaggle).",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision (AMP) on CUDA.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=default_data_dir,
        help="Dataset cache directory.",
    )
    parser.add_argument(
        "--out-dir", type=str, default=default_out_dir, help="Output results directory."
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a quick synthetic offline test to verify all code paths.",
    )
    # Use parse_known_args so Jupyter/Kaggle kernel '-f' flags never trigger errors
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    summary_results = {}
    for ds_name in args.datasets:
        res = run_optuna_study_for_dataset(
            dataset_name=ds_name,
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            device=device,
            arch=args.arch,
            n_trials=args.trials,
            search_epochs=args.search_epochs,
            final_epochs=args.final_epochs,
            subset_fraction=args.subset_fraction,
            num_workers=args.num_workers,
            use_amp=not args.no_amp,
            seed=args.seed,
            smoke_test=args.smoke_test,
        )
        summary_results[ds_name] = res

    summary_path = os.path.join(args.out_dir, "titan_best_hyperparameters.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)

    save_training_plots(summary_results, args.out_dir)

    print(f"\n{'=' * 74}")
    print("FINAL SUMMARY: TITAN OPTIMIZER BEST HYPERPARAMETERS & TEST ACCURACY")
    print(f"{'=' * 74}")
    for ds_name, info in summary_results.items():
        print(
            f"\n* {ds_name.upper()}:"
            f"\n  - Best Validation Accuracy : {info['best_validation_accuracy']:.2f}%"
            f"\n  - Best Test Accuracy       : {info['best_test_accuracy']:.2f}%"
            f"\n  - Optimal Learning Rate    : {info['best_hyperparameters']['lr']:.6g}"
            f"\n  - Optimal Weight Decay     : {info['best_hyperparameters']['weight_decay']:.6g}"
            f"\n  - Optimal Momentum (beta)  : {info['best_hyperparameters']['beta']:.4f}"
            f"\n  - Optimal NS Steps         : {info['best_hyperparameters']['ns_steps']}"
            f"\n  - Optimal Epsilon (eps)    : {info['best_hyperparameters']['eps']}"
            f"\n  - Optimal Batch Size       : {info['best_hyperparameters']['batch_size']}"
            f"\n  - Optimal LR Scheduler     : {info['best_hyperparameters']['scheduler']}"
            f"\n  - Optimal Label Smoothing  : {info['best_hyperparameters']['label_smoothing']:.4f}"
        )
    print(f"\nSaved full summary JSON to: {summary_path}")


if __name__ == "__main__":
    main()
