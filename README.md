# TITAN Optimizer: IEEE TPAMI Empirical Benchmark Suite

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-ee4c2c.svg)](https://pytorch.org/)
[![Target: IEEE TPAMI](https://img.shields.io/badge/Target-IEEE%20TPAMI-red.svg)](https://www.computer.org/csdl/journal/tp)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official empirical benchmark suite and reproducibility harness for the paper:
> **"TITAN Optimizer: Hybrid Spectral-Adaptive Training via Numerically Safer Quintic Newton-Schulz Orthogonalization and Aspect-Ratio Scaling"**  
> *Furqan Yaqub Khan and Tawseef Ayoub Shaikh, Member, IEEE*  
> *Department of Computer Science and Engineering, National Institute of Technology (NIT) Srinagar, Jammu & Kashmir 190006, India*

---

## 📌 Key Architectural Innovations of TITAN

`TITAN` addresses the spectral instabilities, condition number degradation, and aspect-ratio distortions inherent in matrix-space second-order preconditioning (e.g., Muon, Shampoo) through four synergistic mechanisms:

1. **Frobenius Pre-Scaling:** $\tilde{M}_t = \frac{M_t}{\|M_t\|_F + \epsilon_0}$, ensuring the spectral radius satisfies $\rho(\tilde{M}_t) \le 1.0$ prior to iterative refinement.
2. **Minimax Quintic Newton–Schulz Recurrence ($K=3$):**  
   $$X_{k+1} = a X_k + b (X_k X_k^T) X_k + c (X_k X_k^T)^2 X_k$$  
   with optimal expansion coefficients $a = 3.4445$, $b = -4.7750$, $c = 2.0315$, producing rapid singular value contraction into the target band $[0.90, 1.10]$ with origin slope $\phi'(0) = 3.4445$ (compared to 1.5 for classical cubic NS).
3. **Unit-RMS Post-Normalization:** Stabilizes parameter updates across disparate weight dimensions without norm blowup.
4. **Aspect-Ratio Scaling:** Scales update variance by $\max(1, \sqrt{m/n})$ for rectangular weight matrices $W \in \mathbb{R}^{m \times n}$.
5. **Hybrid Fallback:** Vector parameters (biases, 1D normalization gains, embeddings) are preconditioned with decoupled AdamW.

---

## 📂 Repository Structure

```text
TITAN_OptimizerBench_Local_Kaggle/
│
├── titan_optimizer.py               # Core reference TITAN optimizer implementation
├── benchmark.py                     # Master orchestrator & dynamic 600-DPI figure generator
│
├── exp1_cifar10_resnet18_main.py    # Exp 1: CIFAR-10 on ResNet-18 (Table I, Fig 1, Fig 2a)
├── exp2_fmnist_cnn4_main.py         # Exp 2: Fashion-MNIST on CNN-4 (Table II, Fig 2b)
├── exp3_ablations_and_sensitivity.py# Exp 3: Component Ablations, K-Depth, LR-Sweep (Table IV, Fig 3)
├── exp4_resnet50_scaling_pareto.py  # Exp 4: ResNet-50 Pareto Compute-Accuracy Frontier (Table III, Fig 4a)
├── exp5_cross_arch_wrn_vit_wikitext2.py # Exp 5: Cross-Architecture (WRN-28-10, ViT-B/16, WikiText-2, Fig 4b)
│
├── generate_ieee_tpami_figures.py   # Standalone publication figure generator
├── requirements.txt                 # Minimum Python dependencies
├── .gitignore                       # Git exclusion rules
│
├── titan_tpami_figures/             # 600-DPI PNG and Vector PDF IEEE TPAMI figures
│   ├── Fig1_Spectral_Contraction_and_Condition_Number.png / .pdf
│   ├── Fig2_CIFAR10_Acc_and_FMNIST_Loss.png / .pdf
│   ├── Fig2a_CIFAR10_Test_Accuracy_ResNet18.png / .pdf
│   ├── Fig3_NS_Depth_Tradeoff_and_LR_Sensitivity.png / .pdf
│   └── Fig4_ResNet50_Pareto_and_CrossArch.png / .pdf
│
└── outputs_exp*/                    # Raw JSON experimental logs (100% dynamic, zero hardcoding)
```

---

## ⚙️ Hardware & Environment Setup

### 1. Local NVIDIA GPU Compatibility (Pascal sm_61 / GTX 1050)

> [!IMPORTANT]
> The NVIDIA GeForce GTX 1050 belongs to the **Pascal architecture (Compute Capability 6.1 / `sm_61`)**.  
> Starting in PyTorch 2.4+, official CUDA 12/13 wheels dropped precompiled kernels for `sm_61`.  
> If using `torch+cu126` or `torch+cu132`, PyTorch raises `torch.AcceleratorError: no kernel image is available`.  
> All scripts in this repository include an automatic **`get_safe_device()`** guard that seamlessly falls back to CPU without crashing.

To enable **hardware GPU acceleration on GTX 1050**, use **Python 3.11** with PyTorch `cu118` (which includes native `sm_61` kernels):

```bash
# Create Python 3.11 virtual environment
py -3.11 -m venv env311
env311\Scripts\activate

# Install PyTorch with CUDA 11.8 support for Pascal GPUs
pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib numpy
```

### 2. Standard Local CPU / Modern CUDA (RTX 30xx/40xx, A100, H100)

```bash
python -m venv env
# Windows:
env\Scripts\activate
# Linux / macOS:
source env/bin/activate

pip install -r requirements.txt
```

### 3. Kaggle (Dual NVIDIA T4 GPUs)

1. Create a Kaggle Notebook and enable **GPU T4 x 2** in Notebook Settings.
2. In the first notebook cell:
```bash
!pip install -q matplotlib torchvision
!git clone https://github.com/YourUsername/TITAN_OptimizerBench.git /kaggle/working/titan
%cd /kaggle/working/titan
!python benchmark.py --exp all
```

---

## 🚀 Running the Benchmarks

The master runner `benchmark.py` provides unified execution, validation, and automated figure rendering.

### Fast Smoke-Test / Quick Verification Mode (`--quick`)

Runs all 5 experiments with reduced epochs (2 epochs each, subsampled data) to verify complete execution and end-to-end figure generation in under 3 minutes:

```bash
python benchmark.py --exp all --quick
```

You can also run any specific experiment in quick mode:
```bash
python benchmark.py --exp 1 --quick    # Exp 1: CIFAR-10 ResNet-18
python benchmark.py --exp 2 --quick    # Exp 2: Fashion-MNIST CNN-4
python benchmark.py --exp 3 --quick    # Exp 3: Ablations + K-Sweep + LR-Sweep
python benchmark.py --exp 4 --quick    # Exp 4: ResNet-50 Pareto
python benchmark.py --exp 5 --quick    # Exp 5: WRN-28-10, ViT-B/16, WikiText-2
```

### Full Publication-Scale Training (Zero Time Constraints)

```bash
# Execute the complete TPAMI empirical benchmark suite (unconstrained full run)
python benchmark.py --exp all
```

Or execute individual standalone scripts directly:
```bash
python exp1_cifar10_resnet18_main.py --epochs 200 --runs 5
python exp2_fmnist_cnn4_main.py --epochs 100 --runs 5
python exp3_ablations_and_sensitivity.py --epochs-c10 200 --epochs-fm 100
python exp4_resnet50_scaling_pareto.py --epochs 160
python exp5_cross_arch_wrn_vit_wikitext2.py --epochs-wrn 100 --epochs-vit 100 --epochs-lm 40
```

### Regenerating Figures Immediately (`--plot-only`)

Renders all IEEE TPAMI publication figures directly from existing JSON logs without re-running model training:

```bash
python benchmark.py --plot-only
```

---

## 📊 Generated IEEE TPAMI Publication Figures

All output figures are automatically saved to `./titan_tpami_figures/` in both **600-DPI PNG** (rasterized) and **Vector PDF** (with Type 42 fonts embedded conforming to IEEE TPAMI camera-ready requirements):

| Figure File | Description | Panels |
|:---|:---|:---|
| `Fig1_Spectral_Contraction_and_Condition_Number` | Spectral contraction mapping & empirical condition number | **(a)** Quintic vs. Cubic polynomial contraction $\phi^{(K)}(\sigma_0)$<br>**(b)** Measured $\kappa(\tilde{G}_t)$ trajectory ($< 4.2$) |
| `Fig2_CIFAR10_Acc_and_FMNIST_Loss` | Learning curves & convergence descent | **(a)** CIFAR-10 ResNet-18 Top-1 test accuracy<br>**(b)** Fashion-MNIST CNN-4 cross-entropy loss |
| `Fig3_NS_Depth_Tradeoff_and_LR_Sensitivity` | Algorithmic depth trade-off & hyperparameter robustness | **(a)** Accuracy vs. per-epoch runtime across $K \in \{1,2,3,4,5,7\}$<br>**(b)** Learning rate sensitivity across $\eta \in [10^{-4}, 3\times 10^{-3}]$ |
| `Fig4_ResNet50_Pareto_and_CrossArch` | Efficiency scaling & architectural generalization | **(a)** ResNet-50 compute-accuracy Pareto frontier (bubble area = peak VRAM)<br>**(b)** Cross-architecture Top-1 accuracy (ResNet-18, ResNet-50, WRN-28-10, ViT-B/16) |
| `Fig2a_CIFAR10_Test_Accuracy_ResNet18` | Standalone high-res test accuracy curve | CIFAR-10 ResNet-18 Top-1 accuracy comparison |

> [!NOTE]
> **Zero Static / Synthetic Values:** All plotted curves, condition numbers, execution runtimes, and Pareto frontiers are read dynamically from actual execution logs saved in `outputs_exp*/`. No hardcoded fallback curves are rendered.

---

## 📜 Citation

If you find this work or implementation useful in your research, please cite:

```bibtex
@article{khan2026titan,
  title={TITAN Optimizer: Hybrid Spectral-Adaptive Training via Numerically Safer Quintic Newton-Schulz Orthogonalization and Aspect-Ratio Scaling},
  author={Khan, Furqan Yaqub and Shaikh, Tawseef Ayoub},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year={2026}
}
```

---

## 📄 License
This repository is open-source under the [MIT License](LICENSE).
