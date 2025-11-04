# Data-Driven Discovery of Governing Equations via Reservoir Computing Data Augmentation

A novel framework that combines Echo State Networks (ESNs) with Sparse Identification of Nonlinear Dynamics (SINDy) to discover governing equations of chaotic systems from incomplete and noisy data.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![ReservoirPy](https://img.shields.io/badge/ReservoirPy-required-orange.svg)
![PySINDy](https://img.shields.io/badge/PySINDy-required-red.svg)

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [Methodology](#methodology)
- [Installation](#installation)
- [Output Visualizations](#output-visualizations)
- [Results](#results)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

This project addresses a fundamental challenge in scientific computing: **discovering interpretable governing equations from limited, noisy observational data**. Traditional equation discovery methods like SINDy struggle with sparse or incomplete datasets. This framework solves that problem by using reservoir computing to generate high-quality synthetic data that augments the original observations.

### The Problem

- Real-world data is often incomplete (sensor failures, sampling limitations)
- Noisy observations obscure underlying dynamics
- Limited data leads to poor equation discovery
- SINDy requires sufficient high-quality data to identify correct terms

### Proposal

1. **Train** an Echo State Network on incomplete/noisy data with masking
2. **Generate** synthetic trajectories with uncertainty quantification via Monte Carlo sampling
3. **Augment** original data with high-confidence synthetic observations
4. **Discover** improved governing equations using SINDy on the extended dataset

## Motivation

**Data augmentation for equation discovery**: Rather than using reservoir computing solely for prediction, this framework leverages ESNs to intelligently extend training datasets, enabling SINDy to discover more accurate and sparse governing equations from originally insufficient data.

### Core Ideas

- **ESNs capture dynamics** from limited noisy data through reservoir state space
- **Monte Carlo sampling** provides uncertainty estimates, filtering unreliable predictions
- **Extended datasets** give SINDy more information for sparse regression
- **Adaptive thresholding** ensures only high-confidence data augments the training set

## Methodology

```
┌─────────────────┐
│ Original Data   │
│ (incomplete,    │
│  noisy)         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Apply Masking   │
│ (spatial +      │
│  temporal)      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Train ESN       │
│ (with           │
│  imputation)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ MC Generation   │
│ (N runs with    │
│  noise)         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Uncertainty     │
│ Filtering       │
│ (adaptive       │
│  threshold)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Data            │
│ Augmentation    │
│ (original +     │
│  synthetic)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ SINDy Discovery │
│ • Baseline      │
│ • Extended      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Governing       │
│ Equations       │
└─────────────────┘
```

### Supported Dynamical Systems

Nine nonlinear systems spanning chaotic and linear regimes:

#### Chaotic Systems
- **Lorenz Attractor**: Classic butterfly pattern (σ=10, ρ=28, β=8/3)
- **Rössler System**: Single-banded chaos (a=0.2, b=0.2, c=5.7)
- **Chen Attractor**: Bridge system (a=35, b=3, c=28)
- **Thomas Attractor**: Cyclically symmetric (b=0.208186)
- **Halvorsen System**: Complex three-wing attractor (a=1.4)
- **Rabinovich-Fabrikant**: Parameter-dependent chaos (α=0.14, γ=0.1)

#### Linear Systems
- **Sea-Surface Temperature**: Coupled oscillators with complex eigenvalues
- **Ozone Dynamics**: Linear system with constant forcing
- **Cylinder Flow**: 3D projection of 6D von Kármán vortex street

## Installation

```bash
pip install requirements.txt
```

## Output Visualizations

The framework generates four comprehensive figures:

### 1. Time Series Analysis
- Original noisy trajectory (gray)
- ESN fit on training data (blue)
- Masked observations highlighted (red points)
- MC mean predictions (red line)
- Uncertainty bands (±1 std, shaded)

### 2. 3D Phase Space
- **Left panel**: Full trajectory with MC ensemble members
- **Right panel**: Warmup-to-generation transition
- Visualization of attractor structure preservation

### 3. Masking Pattern Visualization
- Per-channel comparison: original vs. imputed
- Masked point locations
- Effect of imputation strategy

### 4. Mask Statistics Heatmap
- **Temporal mask**: Timestep-wise pattern
- **Spatial mask**: Channel × timestep distribution
- Masking density visualization

## Results

### Lorenz

Discovered baseline SINDy model (original data only):
(X)' = 0.065 1 + -9.956 X + 9.955 Y
(Y)' = -0.113 1 + 27.754 X + -0.954 Y + -0.993 X Z
(Z)' = -2.651 Z + 0.995 X Y


Discovered extended SINDy model (with MC predictions):
(X)' = 0.075 1 + -9.956 X + 9.956 Y
(Y)' = -0.068 1 + 27.733 X + -0.946 Y + -0.993 X Z
(Z)' = -2.653 Z + 0.995 X Y

After extensive testing across all nine dynamical systems, the framework shows marginal improvements for equation discovery. The Lorenz system shows negligible benefit from data augmentation, while other systems perform comparably between baseline and extended models.

### Why This Happens: Three Fundamental Limitations

#### 1. **Insufficient Training Data → Poor ESN Fit**

When the original dataset is too small (the scenario where augmentation should help most), the ESN cannot learn accurate dynamics:

```
Small data → Poor ESN training → Inaccurate predictions → Bad augmentation
```

- ESNs need ~200-300 timesteps minimum to capture attractor geometry
- With heavy masking (20%+), effective training data becomes insufficient
- Poor ESN fit (R² < 0.95) produces synthetic data that misleads SINDy
- **Paradox**: Augmentation is most needed when ESNs are least reliable

#### 2. **Error Accumulation: Slop Creates Slop**

The data augmentation pipeline compounds errors at each stage:

```
Original noise → ESN approximation error → MC sampling noise → SINDy regression error
```

- ESN predictions are never perfect
- Each generated timestep introduces small systematic biases
- Monte Carlo noise injection (necessary for uncertainty) adds variance
- SINDy trained on augmented data learns from corrupted dynamics
- **Result**: Extended model fits noisy synthetic data, not true equations

#### 3. **Large Dataset Regime: Augmentation Becomes Redundant**

When sufficient original data exists, SINDy already performs optimally:

```
Large data → Good baseline SINDy → Augmentation adds no information
```

- With 300+ clean timesteps, SINDy reliably identifies correct equations
- ESN predictions don't add new dynamical information, just interpolation
- Baseline and extended models converge to same solution
- **Observation**: The "sweet spot" (data-limited but ESN-trainable) is narrow

## License

```
MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Acknowledgments

- **ReservoirPy** team for excellent reservoir computing tools
- **PySINDy** developers for sparse identification framework
- Jaideep Pathak et al. for pioneering reservoir computing for chaos
- Steven Brunton for SINDy methodology and open-source commitment

---
