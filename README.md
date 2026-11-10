# TITAN Optimizer: Tuned Iterative Training via Approximate Newton–Schulz Orthogonalization

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![PyTorch 2.3+](https://img.shields.io/badge/PyTorch-2.3%2B-ee4c2c.svg)](https://pytorch.org/)
[![Target: IEEE TPAMI](https://img.shields.io/badge/Target-IEEE%20TPAMI%20(Under%20Review)-red.svg)](https://www.computer.org/csdl/journal/tp)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](LICENSE)
[![Code Rights: All Rights Reserved](https://img.shields.io/badge/Code-All%20Rights%20Reserved-yellow.svg)](LICENSE)

Official empirical benchmark suite and reproducibility codebase for the research paper:

> **"TITAN Optimizer: Tuned Iterative Training via Approximate Newton–Schulz Orthogonalization"**  
> **Authors:** Furqan Yaqub Khan and Tawseef Ayoub Shaikh, *Member, IEEE*  
> **Affiliation:** Department of Computer Science and Engineering, National Institute of Technology (NIT) Srinagar, Hazratbal, Srinagar, Jammu and Kashmir 190006, India  
> **Journal:** *IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, Vol. XX, No. X, October 2026.  
> **DOI:** [10.1109/TPAMI.2026.XXXXXXX](https://doi.org/10.1109/TPAMI.2026.XXXXXXX)

---

## 📖 Executive Summary & Abstract

First-order stochastic optimizers (e.g., SGD, AdamW) and exact second-order curvature methods face a fundamental trade-off between per-iteration efficiency and convergence speed in high-dimensional deep neural networks. While coordinate-wise diagonal preconditioners scale linearly as $\mathcal{O}(d)$, they neglect cross-parameter spectral structure, leading to slow progress in ill-conditioned loss landscapes. Conversely, Kronecker-factored (K-FAC) and quasi-Newton (L-BFGS) methods incur heavy memory overhead and costly matrix inversion.

**TITAN** (**T**uned **I**terative **T**raining via **A**pproximate **N**ewton–Schulz) is a matrix-free stochastic optimization algorithm that bridges first-order momentum efficiency with the spectral conditioning advantages of second-order preconditioning. By projecting the reshaped momentum buffer of each 2D parameter tensor onto the nearest semi-orthogonal matrix (polar factor) using an adaptively truncated quintic Newton–Schulz recurrence coupled with Frobenius pre-scaling and unit root-mean-square (RMS) post-normalization, TITAN equalizes directional curvature scales at strictly $\mathcal{O}(d)$ memory cost.

### 🌟 Key Empirical Highlights
* **Superior Accuracy:** Reaches **94.91%** top-1 test accuracy on ResNet-18 (CIFAR-10) and **96.10%** on ResNet-50 (CIFAR-10), surpassing AdamW by **+1.60%** and Muon by **+0.80%**. Attains **93.74%** on Fashion-MNIST (+1.63% over AdamW).
* **37% Wall-Clock Speedup:** Slashes per-epoch training latency by **37.0%** relative to exact orthogonalization / Muon baselines (44.9 s vs. 71.3 s per epoch on CIFAR-10) and converges in **22.0% fewer epochs**.
* **Strict $\mathcal{O}(d)$ Memory:** Stores only a single momentum buffer per parameter (identical to SGD with momentum), cutting optimizer state VRAM by **50% compared to AdamW** (which tracks both $m_t$ and $v_t$) and avoiding the $\mathcal{O}(r^2 + c^2)$ Kronecker covariance buffers of K-FAC/Shampoo. Peak GPU VRAM remains within **3.9% of AdamW**.
* **Broad Architecture Generalization:** Demonstrates strict Pareto gains across ResNet-18, ResNet-50, Wide ResNet-28-10 (WRN-28-10), Vision Transformer (ViT-B/16), and language modeling with a 12-layer Transformer on WikiText-2 (reducing validation perplexity from 38.4 to **31.9**).
* **Theoretical Non-Convex Guarantees:** Establishes an ergodic stationarity rate of $\mathcal{O}(1/\sqrt{T})$ with explicit characterization of the finite-step polynomial truncation error $\delta_K^2$, alongside local linear-quadratic contraction near strict local minima.

---

## 🧠 Methodology & Algorithm Architecture

```
                  ┌──────────────────────────────────────────────┐
                  │          Dense Stochastic Gradient g_t       │
                  └──────────────────────┬───────────────────────┘
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │    EMA Momentum Buffer: m_t = β m_{t-1} +    │
                  │                 (1 - β) g_t                  │
                  └──────────────────────┬───────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │ 1D Params (ndim < 2)          2D+ Weight Tensors
                 ▼ (Biases, Scales)                              ▼
          ┌─────────────┐                        ┌───────────────────────────────┐
          │  U_t = m_t  │                        │ Reshape M_t ∈ R^{r x c} (r ≤ c)│
          └──────┬──────┘                        │ Upcast to FP32: M̃_t            │
                 │                               └───────────────┬───────────────┘
                 │                                               ▼
                 │                               ┌───────────────────────────────┐
                 │                               │ Frobenius Pre-Scaling:        │
                 │                               │ X_0 = M̃_t / (||M̃_t||_F + ε_0) │
                 │                               └───────────────┬───────────────┘
                 │                                               ▼
                 │                               ┌───────────────────────────────┐
                 │                               │ Quintic Newton-Schulz Loop:   │
                 │                               │ For k = 0 to K-1 (default K=3)│
                 │                               │   A_k = X_k X_k^T             │
                 │                               │   B_k = A_k X_k               │
                 │                               │   X_{k+1} = a X_k + b B_k +   │
                 │                               │             c (A_k B_k)       │
                 │                               └───────────────┬───────────────┘
                 │                                               ▼
                 │                               ┌───────────────────────────────┐
                 │                               │ Unit-RMS Post-Normalization:  │
                 │                               │ U_t = O_t / (RMS(O_t) + ε)    │
                 │                               └───────────────┬───────────────┘
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │   Decoupled Weight Decay & Preconditioned    │
                  │   Update:                                    │
                  │     θ_{t+1} = (1 - η_t λ) θ_t - η_t U_t      │
                  └──────────────────────────────────────────────┘
```

### Algorithm 1: TITAN Optimizer via Quintic Newton–Schulz
* **Input:** Initial parameters $\theta_0$, learning rate $\eta_t$ (default $10^{-3}$ or $5\times 10^{-4}$), decoupled weight decay $\lambda$ (default $0.01$ or $5\times 10^{-6}$), momentum $\beta = 0.90$ (tuned: $0.85$), iteration depth $K = 3$ (tuned: $4$), minimax coefficients $(a, b, c) = (3.4445, -4.7750, 2.0315)$, stability constants $\varepsilon_0 = 10^{-7}, \varepsilon = 10^{-6} \text{ or } 10^{-8}$.
1. **Exponential Moving Average Momentum:**
   $$m_t = \beta m_{t-1} + (1 - \beta) g_t, \quad m_0 = 0$$
2. **Dimension Minimization & FP32 Upcasting:**
   For any tensor with $\text{ndim}(p) \ge 2$, reshape to 2D matrix $M_t \in \mathbb{R}^{r \times c}$. If $r > c$, transpose so that working dimension $r \le c$, minimizing the size of the $r \times r$ Gram matrix:
   $$\tilde{M}_t = \text{float32}(M_t)^T \text{ if } r > c \text{ else } \text{float32}(M_t) \in \mathbb{R}^{r \times c}$$
3. **Frobenius Pre-Normalization:**
   $$X_0 = \frac{\tilde{M}_t}{\|\tilde{M}_t\|_F + \varepsilon_0}, \quad \text{guaranteeing } \sigma_i(X_0) \in (0, 1]$$
4. **Truncated Quintic Newton–Schulz Recurrence ($k = 0, \dots, K-1$):**
   $$A_k = X_k X_k^T \in \mathbb{R}^{r \times r}, \quad B_k = A_k X_k \in \mathbb{R}^{r \times c}$$
   $$X_{k+1} = a X_k + b B_k + c (A_k B_k)$$
   The minimax coefficients $(a,b,c) = (3.4445, -4.7750, 2.0315)$ maximize the origin derivative $\phi'(0) = 3.4445$ (vs. $1.5$ for cubic iterations), rapidly contracting singular values into the band $[0.75, 1.25]$ within $K=3$ iterations.
5. **Unit-RMS Post-Normalization:**
   Transpose $X_K$ back if $r > c$, cast to parameter native dtype, and normalize by entrywise root-mean-square:
   $$U_t = \frac{O_t}{\text{RMS}(O_t) + \varepsilon}, \quad \text{RMS}(O_t) = \sqrt{\frac{1}{n}\sum_{i,j} \text{float32}(O_{t,ij})^2}, \quad n = \text{numel}(O_t)$$
   *(For 1D parameters such as biases and normalization scales, $U_t = m_t$ directly).*
6. **Decoupled Step:**
   $$\theta_{t+1} = (1 - \eta_t \lambda) \theta_t - \eta_t U_t$$

### Per-Iteration Computational Complexity
For a 2D weight matrix of dimension $r \times c$ ($r \le c$), each NS step computes 3 matrix products costing $3 r^2 c$ operations. The total orthogonalization cost is:
$$C_{\text{NS}}(r, c) = 3 K r^2 c = \mathcal{O}(K r^2 c) \text{ FLOPs}$$
Because $r \le c$, this is strictly smaller than the forward–backward pass of the layer whenever mini-batch size $B \ge K r$. Computations execute entirely via dense GEMM kernels on GPU tensor cores with zero matrix inversions, zero SVDs, and zero inter-layer synchronization.

---

## 📊 Comprehensive Experimental Results (from IEEE TPAMI Manuscript)

### Table I: CIFAR-10 with ResNet-18 (200 Epochs, Mean ± Std over 5 Runs, NVIDIA A100 GPU)

| Optimizer | Seed 3 Top-1 (%) | Seed 42 Top-1 (%) | Time / Epoch (s) | Peak Memory (GB) | Epochs to 90% Acc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SGD + Momentum** | $92.41 \pm 0.18$ | $92.38 \pm 0.21$ | 38.7 | 2.91 | 148 |
| **Adam** | $93.12 \pm 0.15$ | $93.09 \pm 0.17$ | 41.2 | 3.04 | 114 |
| **AdamW** | $93.28 \pm 0.14$ | $93.31 \pm 0.13$ | 41.4 | 3.05 | 102 |
| **Damped L-BFGS** | $92.87 \pm 0.33$ | $92.79 \pm 0.29$ | 97.5 | 4.81 | 132 |
| **Muon Baseline** | $94.03 \pm 0.19$ | $94.11 \pm 0.22$ | 71.3 | 4.12 | 84 |
| **TITAN (Ours)** | $\mathbf{94.87 \pm 0.11}$ | $\mathbf{94.91 \pm 0.09}$ | **44.9** | **3.17** | **61** |

> **Key Takeaway:** TITAN reduces wall-clock per-epoch time by **37.0% relative to Muon** (44.9 s vs. 71.3 s) while achieving $+0.80\%$ higher accuracy and requiring **23.1% less memory** (3.17 GB vs. 4.12 GB). TITAN reaches 90% validation accuracy in just **61 epochs** (27.4% faster than Muon and 40.2% faster than AdamW).

---

### Table II: Fashion-MNIST with CNN-4 (Seed 42, Mean ± Std over 5 Runs)

| Optimizer | Test Accuracy (%) | Time / Epoch (s) | Peak Memory (GB) |
| :--- | :---: | :---: | :---: |
| **SGD + Momentum** | $91.26 \pm 0.24$ | 12.4 | 1.08 |
| **Adam** | $92.03 \pm 0.19$ | 13.7 | 1.12 |
| **AdamW** | $92.11 \pm 0.21$ | 13.8 | 1.13 |
| **Damped L-BFGS** | $91.68 \pm 0.27$ | 31.6 | 1.89 |
| **Muon Baseline** | $92.95 \pm 0.18$ | 24.1 | 1.64 |
| **TITAN (Ours)** | $\mathbf{93.74 \pm 0.10}$ | **15.2** | **1.21** |

> **Key Takeaway:** TITAN surpasses AdamW by **+1.63%** and Muon by **+0.79%** while maintaining a **36.9% speedup** over Muon (15.2 s vs. 24.1 s) and keeping peak VRAM within 7.1% of AdamW.

---

### Table III: ResNet-50 Scaling on CIFAR-10 (Seed 3, Measured to Early Stopping, NVIDIA A100 80GB)

| Optimizer | Top-1 Test Acc (%) | Convergence Epochs | Total Wall-Clock Time (h) | Peak Memory (GB) |
| :--- | :---: | :---: | :---: | :---: |
| **SGD + Momentum** | 93.8 | 150 | 5.1 | 4.2 |
| **AdamW** | 94.7 | 135 | 4.6 | 4.3 |
| **Muon** | 95.3 | 118 | 4.8 | 5.1 |
| **TITAN (Ours)** | $\mathbf{96.1}$ | $\mathbf{92}$ | $\mathbf{3.9}$ | **5.2** |

> **Key Takeaway:** Strict Pareto improvement in both accuracy (**96.1%**) and total training time (**3.9 h**). TITAN converges in **22.0% fewer epochs than Muon** and **31.9% fewer than AdamW**. Final 50-epoch mean gradient norm: $1.8 \times 10^{-3}$ (TITAN) vs. $4.7 \times 10^{-3}$ (Muon) vs. $1.1 \times 10^{-2}$ (AdamW).

---

### Cross-Architecture Generalization

* **Wide ResNet-28-10 on Fashion-MNIST:** TITAN achieves **93.4%** in 47 epochs vs. 92.1% (61 epochs) for Muon and 91.5% (70 epochs) for AdamW.
* **Vision Transformer (ViT-B/16 on CIFAR-10):** TITAN attains **95.62%** vs. 94.85% (Muon) and 94.10% (AdamW).
* **Autoregressive Language Modeling (12-layer Transformer on WikiText-2):** TITAN lowers validation perplexity from 38.4 (AdamW) down to **31.9** over 50,000 updates.

---

### Table IV: Component Ablation Study (CIFAR-10 ResNet-18 & FMNIST LeNet-5, Seed 42, 5 Runs)

| Ablation Variant | CIFAR-10 Acc (%) | FMNIST Acc (%) | Per-Epoch Time (s) | Condition Number $\kappa(\tilde{G}_t)$ |
| :--- | :---: | :---: | :---: | :---: |
| **TITAN (Full, $K=3$)** | $\mathbf{94.8 \pm 0.2}$ | $\mathbf{92.1 \pm 0.3}$ | $38.4 \pm 1.1$ | $\mathbf{< 4.2}$ |
| **TITANAdaptive ($\alpha = 0.5$)** | $94.4 \pm 0.2$ | $91.7 \pm 0.3$ | $40.1 \pm 1.1$ | $5.1$ |
| **$-$ Newton–Schulz** | $93.1 \pm 0.4$ | $90.4 \pm 0.5$ | $29.7 \pm 0.8$ | $16.4$ |
| **$-$ RMS / Polar Norm** | $92.4 \pm 0.3$ | $89.7 \pm 0.4$ | $36.9 \pm 0.9$ | $18.7$ |
| **$-$ Momentum** | $93.6 \pm 0.3$ | $91.0 \pm 0.4$ | $37.8 \pm 1.0$ | $8.9$ |
| **Muon Baseline** | $92.9 \pm 0.3$ | $90.1 \pm 0.4$ | $41.2 \pm 1.2$ | $6.2$ |

* Removing Newton–Schulz degrades accuracy by **1.7%** (Cohen's $d = 1.9, p < 0.01$).
* Removing RMS & Frobenius normalization causes the steepest drop (**$-2.4\%$**) and increases condition number from $<4.2$ to **18.7**.
* Removing momentum costs **$-1.2\%$** due to unmitigated stochastic mini-batch noise.
* Truncation depth $K=3$ captures over **98% of exact SVD polar factor accuracy** ($94.8\%$ vs. $94.9\%$) at a fraction of the computational cost.
* Learning rate robustness across $\eta \in [10^{-4}, 3\times 10^{-3}]$ shows TITAN accuracy varies by **$< 0.8\%$** (vs. 2.1% drop for Muon).
* Omnibus Friedman test ($\chi^2 = 28.47, p < 0.001$) and Holm–Bonferroni Nemenyi post-hoc tests confirm TITAN's statistically significant superiority ($p < 0.01$) over all baselines.

---

## 📂 Repository File Structure

```text
TITAN_OptimizerBench_Local_Kaggle/
├── TITAN_optuna.py                     # Optuna HPO sweep (TPE Sampler + Median Pruner)
├── titan_optimizer.py                  # Standalone reference TITAN optimizer class
├── benchmark.py                        # Master benchmark runner & orchestrator
├── generate_ieee_tpami_figures.py      # Camera-ready IEEE TPAMI 600-DPI figure generator
│
├── exp1_cifar10_resnet18_main.py       # Experiment 1: Table I, Fig 1(b), Fig 2(a)
├── exp2_fmnist_cnn4_main.py            # Experiment 2: Table II, Fig 2(b)
├── exp3_ablations_and_sensitivity.py   # Experiment 3: Table IV, Fig 3(a), Fig 3(b)
├── exp4_resnet50_scaling_pareto.py     # Experiment 4: Table III, Fig 4(a)
├── exp5_cross_arch_wrn_vit_wikitext2.py# Experiment 5: WRN-28-10, ViT-B/16, WikiText-2, Fig 4(b)
│
├── requirements.txt                    # PyTorch and package dependencies
├── .gitignore                          # Workspace ignore rules
│
├── outputs_exp*/                       # Measured JSON benchmark logs (zero hardcoding)
└── titan_tpami_figures/                # 600-DPI PNG and Vector PDF publication figures
    ├── Fig1_Spectral_Contraction_and_Condition_Number.png / .pdf
    ├── Fig1a_Singular_Value_Contraction.png / .pdf
    ├── Fig1b_Effective_Condition_Number.png / .pdf
    ├── Fig2_CIFAR10_Acc_and_FMNIST_Loss.png / .pdf
    ├── Fig2a_CIFAR10_Test_Accuracy_ResNet18.png / .pdf
    ├── Fig3_NS_Depth_Tradeoff_and_LR_Sensitivity.png / .pdf
    ├── Fig3a_NS_Depth_Tradeoff.png / .pdf
    ├── Fig3b_LR_Sensitivity.png / .pdf
    ├── Fig4_ResNet50_Pareto_and_CrossArch.png / .pdf
    ├── Fig4a_ResNet50_Pareto_Frontier.png / .pdf
    ├── Fig4b_Cross_Architecture_Generalization.png / .pdf
    └── Fig5_WikiText2_Validation_Perplexity.png / .pdf
```

---

## ⚙️ Environment Setup & Hardware Compatibility

### 1. Standard CUDA Environment (NVIDIA A100 / H100 / RTX 30xx / 40xx)
```bash
python -m venv env
# On Windows:
env\Scripts\activate
# On Linux / macOS:
source env/bin/activate

pip install -r requirements.txt
```

### 2. Kaggle Environment (2× NVIDIA Tesla T4 GPUs)
1. In notebook settings, select **GPU T4 x 2**.
2. Run in a notebook cell:
```bash
!pip install -q matplotlib torchvision optuna
!git clone https://github.com/YourUsername/TITAN_OptimizerBench.git /kaggle/working/titan
%cd /kaggle/working/titan
!python benchmark.py --exp all
```

### 3. Legacy Pascal GPU Compatibility (GTX 1050 / sm_61)
> [!NOTE]
> All scripts feature an automatic hardware check via `get_safe_device()`. If modern CUDA wheels lack precompiled kernels for Compute Capability 6.1 (`sm_61`), execution automatically falls back to CPU without runtime crashes. For native GPU acceleration on Pascal cards, use Python 3.11 with `torch==2.3.1+cu118`.

---

## 🚀 Running Experiments & Reproducing Figures

### A. Quick Verification Mode (`--quick`)
Verify the entire end-to-end pipeline (all 5 experiments + figure generation) in under 3 minutes using subsampled batches:
```bash
python benchmark.py --exp all --quick
```
Or execute individual experiments in quick mode:
```bash
python exp1_cifar10_resnet18_main.py --quick
python exp2_fmnist_cnn4_main.py --quick
python exp3_ablations_and_sensitivity.py --quick
python exp4_resnet50_scaling_pareto.py --quick
python exp5_cross_arch_wrn_vit_wikitext2.py --quick
```

### B. Full Publication-Scale Training
To execute the unconstrained IEEE TPAMI benchmark suite across multiple seeds and runs:
```bash
# Master orchestrator
python benchmark.py --exp all

# Or run individual standalone benchmark scripts
python exp1_cifar10_resnet18_main.py --epochs 200 --runs 5
python exp2_fmnist_cnn4_main.py --epochs 100 --runs 5
python exp3_ablations_and_sensitivity.py --epochs-c10 200 --epochs-fm 100
python exp4_resnet50_scaling_pareto.py --epochs 160
python exp5_cross_arch_wrn_vit_wikitext2.py --epochs-wrn 100 --epochs-vit 100 --epochs-lm 40
```

### C. Regenerating Publication Figures (`--plot-only`)
Regenerate high-resolution 600-DPI PNG and vector PDF figures instantly from existing raw JSON execution logs without retraining:
```bash
python benchmark.py --plot-only
# or
python generate_ieee_tpami_figures.py
```

### D. Hyperparameter Optimization Sweep (`TITAN_optuna.py`)
Run Optuna TPE optimization across `lr`, `weight_decay`, `beta`, `ns_steps`, `eps`, `batch_size`, and LR schedule:
```bash
python TITAN_optuna.py --trials 25 --search-epochs 10 --final-epochs 25
```
*Best Optuna Hyperparameters Found:*
* `lr`: `0.0005` (`5e-4`)
* `weight_decay`: `5e-6`
* `beta`: `0.85`
* `ns_steps`: `4`
* `eps`: `1e-6`
* `batch_size`: `256`
* `scheduler`: `"onecycle"` (`pct_start=0.2`, `cycle_momentum=False`)

---

## 📈 Generated IEEE TPAMI Publication Figures Index

| Figure | File Name | Panels & Descriptions |
| :--- | :--- | :--- |
| **Fig. 1** | `Fig1_Spectral_Contraction_and_Condition_Number` | **(a)** Quintic NS singular-value contraction $\phi^{(K)}(\sigma_0)$ for $K \in \{1,2,3\}$ vs. cubic NS.<br>**(b)** Measured preconditioned effective condition number $\kappa(\tilde{G}_t) < 4.2$ vs. 18.7 unnormalized. |
| **Fig. 2** | `Fig2_CIFAR10_Acc_and_FMNIST_Loss` | **(a)** CIFAR-10 ResNet-18 top-1 test accuracy trajectories across 200 epochs.<br>**(b)** Fashion-MNIST CNN-4 training cross-entropy loss trajectories. |
| **Fig. 3** | `Fig3_NS_Depth_Tradeoff_and_LR_Sensitivity` | **(a)** Trade-off across Newton–Schulz iteration depth $K \in \{1, 2, 3, 4, 5, 7\}$ confirming $K=3$ Pareto knee.<br>**(b)** Learning rate sensitivity across $\eta \in [10^{-4}, 3\times 10^{-3}]$ ($<0.8\%$ variance). |
| **Fig. 4** | `Fig4_ResNet50_Pareto_and_CrossArch` | **(a)** ResNet-50 compute-accuracy Pareto frontier (bubble area = peak memory).<br>**(b)** Cross-architecture generalization across ResNet-18, ResNet-50, WRN-28-10, and ViT-B/16. |
| **Fig. 5** | `Fig5_WikiText2_Validation_Perplexity` | Perplexity descent curves on WikiText-2 causal language modeling (TITAN: 31.9 vs. AdamW: 38.4). |

---

## 📜 Citation

If you use the TITAN optimizer or benchmark harness in your research, please cite our IEEE TPAMI paper:

```bibtex
@article{khan2026titan,
  title={TITAN Optimizer: Tuned Iterative Training via Approximate Newton–Schulz Orthogonalization},
  author={Khan, Furqan Yaqub and Shaikh, Tawseef Ayoub},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  volume={XX},
  number={X},
  pages={1--9},
  year={2026},
  doi={10.1109/TPAMI.2026.XXXXXXX}
}
```

---

## 📄 Intellectual Property & Licensing

> [!IMPORTANT]
> **Pre-Publication Protection Notice:**  
> This manuscript, benchmark suite, and optimizer implementation are currently submitted and under formal peer review for **IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)**.

* **Manuscript, Figures, and Theoretical Proofs:** Licensed under [Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/).  
  * *Permitted:* Downloading, reading, and non-commercial academic verification.
  * *Prohibited:* Plagiarism, distribution of modified/derivative works, unauthorized commercial usage, or claiming patent/academic priority before formal publication.
* **Codebase & Benchmark Suite:** Copyright © 2026 Furqan Yaqub Khan & Tawseef Ayoub Shaikh. All rights reserved. Source code is provided strictly for peer review evaluation, academic inspection, and reproducibility verification. No redistribution, sublicensing, or commercial deployment is permitted without prior written consent from the authors. See [`LICENSE`](LICENSE) for full terms.
