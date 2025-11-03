# Data-Driven Discovery of Governing Equations via Reservoir Computing Data Augmentation

A novel framework that combines Echo State Networks (ESNs) with Sparse Identification of Nonlinear Dynamics (SINDy) to discover governing equations of chaotic systems from incomplete and noisy data.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![ReservoirPy](https://img.shields.io/badge/ReservoirPy-required-orange.svg)
![PySINDy](https://img.shields.io/badge/PySINDy-required-red.svg)

## Table of Contents

- [Overview](#overview)
- [Key Innovation](#key-innovation)
- [Methodology](#methodology)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Output Visualizations](#output-visualizations)
- [Performance Metrics](#performance-metrics)
- [Supported Systems](#supported-systems)
- [Use Cases](#use-cases)
- [Examples](#examples)
- [Results Interpretation](#results-interpretation)
- [Advanced Configuration](#advanced-configuration)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)
- [Contributing](#contributing)

## Overview

This project addresses a fundamental challenge in scientific computing: **discovering interpretable governing equations from limited, noisy observational data**. Traditional equation discovery methods like SINDy struggle with sparse or incomplete datasets. This framework solves that problem by using reservoir computing to generate high-quality synthetic data that augments the original observations.

### The Problem

- Real-world data is often incomplete (sensor failures, sampling limitations)
- Noisy observations obscure underlying dynamics
- Limited data leads to poor equation discovery
- SINDy requires sufficient high-quality data to identify correct terms

### The Solution

1. **Train** an Echo State Network on incomplete/noisy data with masking
2. **Generate** synthetic trajectories with uncertainty quantification via Monte Carlo sampling
3. **Augment** original data with high-confidence synthetic observations
4. **Discover** improved governing equations using SINDy on the extended dataset

## Key Innovation

**Data augmentation for equation discovery**: Rather than using reservoir computing solely for prediction, this framework leverages ESNs to intelligently extend training datasets, enabling SINDy to discover more accurate and sparse governing equations from originally insufficient data.

### Why This Works

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

## Features

### 🎯 Core Capabilities

- **ESN-Based Data Extension**: Generate physics-informed synthetic trajectories
- **Uncertainty Quantification**: Monte Carlo ensemble predictions with confidence intervals
- **Adaptive Data Selection**: Intelligent filtering of synthetic data based on uncertainty
- **Comparative Analysis**: Baseline vs. augmented SINDy model evaluation
- **Missing Data Handling**: Robust training with spatial and temporal gaps

### 🔬 Equation Discovery Pipeline

- **Baseline SINDy**: Discover equations from original data alone
- **Extended SINDy**: Discover equations from original + synthetic data
- **Model Comparison**: Sparsity, accuracy, and interpretability metrics
- **Validation**: Derivative fitting and long-term simulation accuracy

### 🌊 Supported Dynamical Systems

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

### 🎭 Data Masking Capabilities

**Spatial Masking** - Variable-specific missing data:
- Simulate sensor-specific failures
- Channel-selective dropout
- Configurable masking ratios per variable

**Temporal Masking** - Time-specific gaps:
- **Random**: Sporadic data loss
- **Contiguous**: Extended observation gaps
- **Periodic**: Regular sampling intervals (e.g., satellite passes)

### 🔧 Imputation Strategies

- **Zero Filling**: Simple replacement
- **Mean Imputation**: Statistical filling
- **Forward Fill**: Propagate last valid observation (recommended for chaotic systems)
- **Linear Interpolation**: Smooth gap filling

## Installation

### Dependencies

```bash
# Core requirements
pip install reservoirpy numpy matplotlib

# For equation discovery
pip install pysindy

# Optional: for enhanced differentiation
pip install scikit-learn scipy
```

### System Requirements

- Python 3.8+
- 4GB RAM minimum (8GB recommended for large ensembles)
- CPU sufficient (GPU not required)

## Quick Start

```python
# 1. Select a dynamical system
SYSTEM = 'lorenz'

# 2. Configure masking to simulate incomplete data
SPATIAL_MASK_ENABLED = True
SPATIAL_MASK_CHANNELS = [0, 1]  # Mask X and Y variables
SPATIAL_MASK_RATIO = 0.05       # 5% missing observations

TEMPORAL_MASK_ENABLED = True
TEMPORAL_MASK_RATIO = 0.05      # 5% missing timesteps

# 3. Run the discovery pipeline
python reservoir_masking_demo.py
```

### Expected Output

```
====================================================
Running Reservoir Computing on: Lorenz System
====================================================

MASKING CONFIGURATION
Temporal Masking: ENABLED
  Masked timesteps: 15/300 (5.0%)
Spatial Masking: ENABLED
  Channel 0 (X): 15/300 masked (5.0%)
  Channel 1 (Y): 15/300 masked (5.0%)

Training ESN on masked data...
Training performance on unmasked data: R²=0.99987, NRMSE=0.01142

Running 300 Monte-Carlo generations...

====================================================
SINDy Model Discovery
====================================================

=== Baseline SINDy (Original Data Only) ===
Discovered baseline SINDy model:
  dX/dt = 10.023(Y - X)
  dY/dt = 27.891X - Y - XZ
  dZ/dt = XY - 2.667Z
  
Sparsity: 0.867

=== Extended SINDy (Original + MC Predictions) ===
Accepting 284 generated timesteps (94.7%)
Combined dataset: 585 timesteps

Discovered extended SINDy model:
  dX/dt = 9.998(Y - X)
  dY/dt = 28.002X - Y - XZ
  dZ/dt = XY - 2.666Z

Sparsity: 0.900
R² (simulation): 0.99912
```

## Configuration

### Reservoir Hyperparameters

```python
# Network architecture
units = 400                # Reservoir neurons (larger = more capacity)
leak_rate = 0.3           # Memory vs. nonlinearity tradeoff (0-1)
spectral_radius = 0.95    # Stability (typically 0.9-1.2)
connectivity = 0.1        # Reservoir sparsity

# Training
regularization = 1e-6     # Ridge regression penalty
input_scaling = 1.0       # Input weight magnitude
```

### Data Masking Configuration

```python
# Spatial masking (channel-specific)
SPATIAL_MASK_ENABLED = True
SPATIAL_MASK_CHANNELS = [0, 1, 2]  # Which variables to mask
SPATIAL_MASK_RATIO = 0.05          # Fraction masked per channel

# Temporal masking (entire timesteps)
TEMPORAL_MASK_ENABLED = True
TEMPORAL_MASK_RATIO = 0.05         # Fraction of timesteps masked
TEMPORAL_MASK_TYPE = 'random'      # 'random', 'contiguous', 'periodic'
TEMPORAL_MASK_BLOCK_SIZE = 5       # For contiguous/periodic modes

# Missing data handling
IMPUTATION_METHOD = 'forward_fill'  # 'zero', 'mean', 'forward_fill', 'interpolate'
```

### Normalization

```python
NORM_METHOD = 'robust_scale'  # Options:
# 'none'         - Original coordinates (not recommended)
# 'robust_scale' - Scale to [-1, 1] (RECOMMENDED)
# 'minmax'       - Scale to [0, 1]
```

### Monte Carlo Generation

```python
# Ensemble configuration
n_runs = 300                    # Number of MC samples
generations = 300               # Prediction horizon (timesteps)
warmup_steps = 100             # Reservoir warmup length

# Uncertainty injection
feedback_noise_scale = 0.8      # Multiplier for observation noise
```

### SINDy Configuration

System-specific optimization (automatically selected):

```python
# Example: Lorenz system
'lorenz': {
    'sindy_threshold': 0.05,     # Sparsity promotion
    'sindy_degree': 2,           # Polynomial degree
    'feature_library': 'polynomial'
}

# Example: Thomas attractor (needs trig functions)
'thomas': {
    'sindy_threshold': 0.05,
    'sindy_degree': 5,
    'feature_library': 'fourier'  # For sin/cos terms
}
```

### Uncertainty Thresholding

Five strategies for accepting synthetic data:

```python
THRESHOLD_MODE = 'adaptive_growth'  # Options:

# 'fixed' - Constant threshold
uncertainty_threshold = 0.5

# 'percentile' - Based on uncertainty distribution
percentile_value = 50

# 'baseline_multiple' - Multiple of observation noise
multiplier = 1.5

# 'adaptive_growth' - Based on initial uncertainty (RECOMMENDED)
growth_factor = 2.0

# 'snr' - Signal-to-noise ratio based
target_snr = 10.0
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

## Performance Metrics

### Training Phase Metrics

```
Training performance on unmasked data:
  R² = 0.99987    # Coefficient of determination (higher is better)
  NRMSE = 0.01142 # Normalized root mean square error (lower is better)
```

### Prediction Phase Metrics

```
MC generative metrics:
  R² = 0.99523
  NRMSE = 0.02187

Empirical coverage (95% confidence band):
  X: 96.3%        # How often true trajectory falls in uncertainty band
  Y: 94.7%
  Z: 95.1%
  All channels: 93.8%
```

### SINDy Discovery Metrics

```
Baseline model:
  Nonzero coefficients: 12
  Sparsity: 0.867           # Fraction of zeros (higher = simpler)
  R² (derivative fit): 0.9876

Extended model:
  Nonzero coefficients: 9
  Sparsity: 0.900
  R² (derivative fit): 0.9923
  R² (simulation): 0.9991   # Long-term prediction accuracy

Change in sparsity: +0.033  # Improvement in model simplicity
```

## Supported Systems

| System | Type | Dimensions | Key Features |
|--------|------|-----------|-------------|
| Lorenz | Chaotic | 3D | Butterfly attractor, sensitive dependence |
| Rössler | Chaotic | 3D | Single band, spiral attractor |
| Chen | Chaotic | 3D | Multi-scroll attractor |
| Thomas | Chaotic | 3D | Cyclically symmetric, requires trig functions |
| Halvorsen | Chaotic | 3D | Three-wing attractor |
| Rabinovich-Fabrikant | Chaotic | 3D | Parameter-dependent chaos |
| Sea-Surface Temp | Linear | 3D | Complex eigenvalues, oscillatory |
| Ozone | Linear | 3D | Constant forcing, environmental model |
| Cylinder Flow | Linear | 3D | Projection of 6D fluid system |

## Use Cases

### 1. Sparse Sensor Networks
**Problem**: Limited sensors measuring complex dynamics  
**Solution**: Use available sensors + ESN augmentation to discover full equations

### 2. Climate Data Analysis
**Problem**: Satellite observations with temporal gaps  
**Solution**: Fill gaps with physics-informed predictions for equation discovery

### 3. Experimental Fluid Dynamics
**Problem**: Expensive PIV measurements, limited data  
**Solution**: Extend experimental runs synthetically for better model identification

### 4. Biological Systems
**Problem**: Irregular sampling of gene expression dynamics  
**Solution**: Augment sparse timeseries for regulatory network discovery

### 5. Financial Markets
**Problem**: Missing data from market closures or sensor failures  
**Solution**: Generate continuous dynamics for better model identification

### 6. Model Validation
**Problem**: Testing equation discovery robustness  
**Solution**: Systematic evaluation with controlled data degradation

## Examples

### Example 1: Basic Discovery (No Masking)

```python
# Establish baseline: perfect data scenario
SYSTEM = 'lorenz'
SPATIAL_MASK_ENABLED = False
TEMPORAL_MASK_ENABLED = False
n_runs = 300
generations = 300

# Expected: Near-perfect equation recovery
# Baseline and extended models should be nearly identical
```

### Example 2: Sensor Failure Recovery

```python
# Simulate two failed sensors
SYSTEM = 'rossler'
SPATIAL_MASK_ENABLED = True
SPATIAL_MASK_CHANNELS = [0, 1]  # X and Y sensors down
SPATIAL_MASK_RATIO = 0.10       # 10% dropout
IMPUTATION_METHOD = 'forward_fill'

# Expected: Extended model significantly outperforms baseline
```

### Example 3: Sparse Temporal Sampling

```python
# Simulate satellite observations (periodic gaps)
SYSTEM = 'chen'
TEMPORAL_MASK_ENABLED = True
TEMPORAL_MASK_TYPE = 'periodic'
TEMPORAL_MASK_RATIO = 0.20      # 20% missing
TEMPORAL_MASK_BLOCK_SIZE = 15   # 15-step gaps

# Expected: ESN fills gaps, enabling better SINDy fit
```

### Example 4: Extreme Data Scarcity

```python
# Combined challenges: multiple failure modes
SYSTEM = 'thomas'
SPATIAL_MASK_ENABLED = True
SPATIAL_MASK_CHANNELS = [2]
SPATIAL_MASK_RATIO = 0.15

TEMPORAL_MASK_ENABLED = True
TEMPORAL_MASK_TYPE = 'contiguous'
TEMPORAL_MASK_RATIO = 0.10
TEMPORAL_MASK_BLOCK_SIZE = 20

# Expected: Demonstrates framework robustness under adversity
```

### Example 5: Threshold Strategy Comparison

```python
# Test different uncertainty filters
SYSTEM = 'halvorsen'
THRESHOLD_MODE = 'adaptive_growth'  # Try all 5 modes

# Compare accepted timesteps and resulting SINDy accuracy
# Adaptive methods typically perform best
```

## Results Interpretation

### What to Look For

#### ✅ Success Indicators

1. **Extended model has higher sparsity** - Found simpler, more correct equations
2. **Simulation R² > 0.99** - Long-term predictions match true dynamics
3. **Coverage ≈ 95%** - Uncertainty estimates are well-calibrated
4. **Accepted timesteps > 200** - Sufficient data augmentation occurred
5. **Derivative fit R² improves** - Better capture of instantaneous dynamics

#### ⚠️ Warning Signs

1. **Sparsity decreases** - Model may be overfitting to noise
2. **Many spurious terms** - Need to increase SINDy threshold
3. **Low acceptance rate (< 30%)** - Increase reservoir size or reduce noise
4. **Poor coverage (< 80%)** - Uncertainty underestimated, reduce feedback noise
5. **Baseline already perfect** - Data not limiting; masking had no effect

### Understanding the Comparison

```python
# Example output interpretation:
Baseline model: 12 terms, sparsity=0.867
Extended model: 9 terms, sparsity=0.900
Change in sparsity: +0.033

# Interpretation:
# ✓ Extended model is SIMPLER (fewer terms)
# ✓ Sparsity INCREASED (closer to true equations)
# ✓ Data augmentation successfully improved discovery
```

### System-Specific Expectations

- **Lorenz/Rössler/Chen**: Should recover 11-12 terms with high accuracy
- **Thomas**: Requires Fourier library; expect ~9-12 terms with trig functions
- **Linear systems**: Should identify exactly n² + n terms (coefficients + forcing)
- **Highly chaotic**: May need more MC runs (500+) for stable augmentation

## Advanced Configuration

### Fine-Tuning Reservoir Performance

```python
# For highly chaotic systems:
units = 600
spectral_radius = 1.1
leak_rate = 0.2

# For smoother systems:
units = 300
spectral_radius = 0.9
leak_rate = 0.5

# For noisy data:
regularization = 1e-5  # Stronger regularization
```

### Optimizing SINDy Discovery

```python
# Increase sparsity (fewer terms):
sindy_threshold = 0.10

# Allow more terms (complex systems):
sindy_threshold = 0.01
sindy_degree = 3

# Better handling of noise:
differentiation_method = ps.SmoothedFiniteDifference(
    smoother_kws={'window_length': 15, 'polyorder': 5}
)
```

### Uncertainty Control

```python
# More conservative (higher quality data):
THRESHOLD_MODE = 'adaptive_growth'
growth_factor = 1.5  # Lower = stricter

# More aggressive (more augmentation):
growth_factor = 3.0  # Higher = more permissive

# Disable filtering entirely (use all predictions):
# Set THRESHOLD_MODE = 'fixed' and uncertainty_threshold = 1e6
```

## Troubleshooting

### Issue: Poor Baseline SINDy Performance

**Symptoms**: Low R², many incorrect terms  
**Solutions**:
- Check data quality (plot trajectories)
- Reduce noise_level parameter
- Increase train_len
- Adjust sindy_threshold
- Try different feature libraries

### Issue: Extended Model Worse Than Baseline

**Symptoms**: Lower sparsity, worse simulation accuracy  
**Solutions**:
- Reduce feedback_noise_scale (currently 0.8)
- Use stricter uncertainty threshold
- Increase n_runs for better MC statistics
- Check if enough timesteps were accepted
- Verify ESN training R² is high (> 0.98)

### Issue: Low Acceptance Rate

**Symptoms**: < 50 timesteps accepted  
**Solutions**:
- Reduce THRESHOLD_MODE growth_factor
- Increase reservoir units
- Train longer (increase train_len)
- Reduce masking ratios
- Check for ESN instability (R² dropping rapidly)

### Issue: MC Predictions Diverge

**Symptoms**: Uncertainty explodes, no timesteps accepted  
**Solutions**:
- Reduce spectral_radius (< 1.0 for stability)
- Increase warmup_steps
- Reduce feedback_noise_scale
- Check for NaN values in training

### Issue: Perfect Baseline (No Improvement)

**Symptoms**: Both models identical, high accuracy  
**Solutions**:
- This is expected if masking is too light
- Increase masking ratios (10-20%)
- Add more noise (noise_level = 0.1)
- Reduce train_len to limit available data
- Try more challenging system (e.g., Thomas)

### Issue: SINDy Finds No Terms

**Symptoms**: All coefficients zero  
**Solutions**:
- Decrease sindy_threshold (try 0.01)
- Check data normalization (use 'robust_scale')
- Verify system equations are discoverable with chosen library
- For Thomas system, ensure Fourier library is used

### Issue: Memory Error

**Symptoms**: System runs out of RAM  
**Solutions**:
- Reduce n_runs (try 100)
- Reduce units (try 200)
- Reduce generations (try 200)
- Process in batches

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{reservoir_sindy_discovery,
  title={Data-Driven Discovery of Governing Equations via Reservoir Computing Data Augmentation},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/reservoir-sindy-discovery}
}
```

### Key References

```bibtex
@article{jaeger2001echo,
  title={The "echo state" approach to analysing and training recurrent neural networks},
  author={Jaeger, Herbert},
  journal={GMD Report 148, German National Research Center for Information Technology},
  year={2001}
}

@article{brunton2016discovering,
  title={Discovering governing equations from data by sparse identification of nonlinear dynamical systems},
  author={Brunton, Steven L and Proctor, Joshua L and Kutz, J Nathan},
  journal={Proceedings of the National Academy of Sciences},
  volume={113},
  number={15},
  pages={3932--3937},
  year={2016},
  doi={10.1073/pnas.1517384113}
}

@article{pathak2018model,
  title={Model-free prediction of large spatiotemporally chaotic systems from data: A reservoir computing approach},
  author={Pathak, Jaideep and Hunt, Brian and Girvan, Michelle and Lu, Zhixin and Ott, Edward},
  journal={Physical Review Letters},
  volume={120},
  number={2},
  pages={024102},
  year={2018}
}
```

## License

```
MIT License

Copyright (c) 2024

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

## Contributing

Contributions are welcome! This is an active research area with many opportunities for enhancement.

### Priority Areas

1. **New Systems**: Add more dynamical systems (4D, 5D, PDEs)
2. **Advanced ESN Architectures**: Deep reservoirs, hierarchical networks
3. **Imputation Methods**: Neural ODEs, Gaussian processes
4. **Uncertainty Methods**: Ensemble ESNs, Bayesian approaches
5. **Real Data**: Apply to experimental datasets
6. **Performance**: GPU acceleration, parallel MC sampling

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/NewSystem`)
3. Add tests for your changes
4. Update documentation
5. Submit a pull request

### Code Style

- Follow PEP 8
- Add docstrings to new functions
- Include example usage in comments
- Update README with new features

## Acknowledgments

- **ReservoirPy** team for excellent reservoir computing tools
- **PySINDy** developers for sparse identification framework
- Jaideep Pathak et al. for pioneering reservoir computing for chaos
- Steven Brunton for SINDy methodology and open-source commitment

---

**Discovering equations from data, one reservoir at a time** 🌊➡️📐