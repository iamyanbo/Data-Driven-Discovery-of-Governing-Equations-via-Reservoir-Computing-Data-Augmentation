# Reservoir Computing — Multi-System generative demo with Spatial & Temporal Masking
# Save and run in an environment with ReservoirPy installed (pip install reservoirpy).
# This script trains a single ESN on various dynamical systems with masking capabilities,
# estimates observational noise, then performs Monte‑Carlo closed-loop generations.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============== SELECT DYNAMICAL SYSTEM ==============
SYSTEM = 'thomas'  # Options: 'lorenz', 'rossler', 'chen', 'thomas', 'halvorsen', 'rabinovich_fabrikant', 'sea_surface_temp', 'ozone', 'cylinder_flow'
# =====================================================

# ============== MASKING CONFIGURATION ==============
# Spatial masking: mask specific channels/variables
SPATIAL_MASK_ENABLED = True
SPATIAL_MASK_CHANNELS = [0, 1, 2]
SPATIAL_MASK_RATIO = 0.05

# Temporal masking: mask entire timesteps across all channels
TEMPORAL_MASK_ENABLED = True
TEMPORAL_MASK_RATIO = 0.05
TEMPORAL_MASK_TYPE = 'random'
TEMPORAL_MASK_BLOCK_SIZE = 5

NORM_METHOD = 'robust_scale'  # 'none', 'robust_scale', or 'minmax'

# Missing data handling strategy
IMPUTATION_METHOD = 'forward_fill'  # 'zero', 'mean', 'forward_fill', or 'interpolate'
# ===================================================

# ---------------- CONFIG --------------------------------
noise_level = 0.05
seed = 42
np.random.seed(seed)

# Reservoir hyperparameters
units = 400
leak_rate = 0.3
spectral_radius = 0.95
input_scaling = 1.0
connectivity = 0.1
input_connectivity = 0.2
regularization = 1e-6

# generative params
warmup_steps = 100
generations = 300

# MC params
n_runs = 300
feedback_noise_scale = 0.8

# --------------------------------------------------
# Define dynamical systems with system-specific parameters

def lorenz_step(x, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x0, y0, z0 = x
    dx = sigma * (y0 - x0)
    dy = x0 * (rho - z0) - y0
    dz = x0 * y0 - beta * z0
    return np.array([dx, dy, dz])

def rossler_step(x, a=0.2, b=0.2, c=5.7):
    x0, y0, z0 = x
    dx = -y0 - z0
    dy = x0 + a * y0
    dz = b + z0 * (x0 - c)
    return np.array([dx, dy, dz])

def chen_step(x, a=35.0, b=3.0, c=28.0):
    x0, y0, z0 = x
    dx = a * (y0 - x0)
    dy = (c - a) * x0 - x0 * z0 + c * y0
    dz = x0 * y0 - b * z0
    return np.array([dx, dy, dz])

def thomas_step(x, b=0.208186):
    x0, y0, z0 = x
    dx = np.sin(y0) - b * x0
    dy = np.sin(z0) - b * y0
    dz = np.sin(x0) - b * z0
    return np.array([dx, dy, dz])

def halvorsen_step(x, a=1.4):
    x0, y0, z0 = x
    dx = -a * x0 - 4 * y0 - 4 * z0 - y0**2
    dy = -a * y0 - 4 * z0 - 4 * x0 - z0**2
    dz = -a * z0 - 4 * x0 - 4 * y0 - x0**2
    return np.array([dx, dy, dz])

def rabinovich_fabrikant_step(x, alpha=0.14, gamma=0.1):
    x0, y0, z0 = x
    dx = y0 * (z0 - 1 + x0**2) + gamma * x0
    dy = x0 * (3 * z0 + 1 - x0**2) + gamma * y0
    dz = -2 * z0 * (alpha + x0 * y0)
    return np.array([dx, dy, dz])

def sea_surface_temp_step(x):
    z1, z2, z3 = x
    dz1 = -0.01 * z1 + 6.24 * z2
    dz2 = -6.24 * z1 - 0.01 * z2
    dz3 = 0.02 * z3
    return np.array([dz1, dz2, dz3])

def ozone_step(x):
    z1, z2, z3 = x
    dz1 = -0.003 * z1 + 0.0079 * z2 - 0.002
    dz2 = -0.0079 * z1 - 0.003 * z2
    dz3 = -0.003 * z3 + 0.002
    return np.array([dz1, dz2, dz3])

def cylinder_flow_step(x):
    z1, z2, z3 = x
    dz1 = -0.01 * z1 + 1.52 * z2
    dz2 = -1.52 * z1 - 0.01 * z2
    dz3 = -0.20 * z3
    return np.array([dz1, dz2, dz3])

SYSTEMS = {
    'lorenz': {
        'step_fn': lorenz_step,
        'x0': [1.0, 1.0, 1.0],
        'name': 'Lorenz',
        'var_names': ['X', 'Y', 'Z'],
        'equations': [
            'dX/dt = σ(Y - X)',
            'dY/dt = X(ρ - Z) - Y',
            'dZ/dt = XY - βZ'
        ],
        'params': 'σ=10, ρ=28, β=8/3',
        'dt': 0.01,
        'total_steps': 400,
        'train_len': 300,
        'sindy_threshold': 0.05,
        'sindy_degree': 2,
    },
    'rossler': {
        'step_fn': rossler_step,
        'x0': [1.0, 1.0, 1.0],
        'name': 'Rössler',
        'var_names': ['X', 'Y', 'Z'],
        'equations': [
            'dX/dt = -Y - Z',
            'dY/dt = X + aY',
            'dZ/dt = b + Z(X - c)'
        ],
        'params': 'a=0.2, b=0.2, c=5.7',
        'dt': 0.01,
        'total_steps': 400,
        'train_len': 300,
        'sindy_threshold': 0.05,
        'sindy_degree': 2,
    },
    'chen': {
        'step_fn': chen_step,
        'x0': [1.0, 1.0, 1.0],
        'name': 'Chen',
        'var_names': ['X', 'Y', 'Z'],
        'equations': [
            'dX/dt = a(Y - X)',
            'dY/dt = (c - a)X - XZ + cY',
            'dZ/dt = XY - bZ'
        ],
        'params': 'a=35, b=3, c=28',
        'dt': 0.01,
        'total_steps': 400,
        'train_len': 300,
        'sindy_threshold': 0.05,
        'sindy_degree': 2,
    },
    'thomas': {
        'step_fn': thomas_step,
        'x0': [0.1, 0.0, 0.0],
        'name': 'Thomas',
        'var_names': ['X', 'Y', 'Z'],
        'equations': [
            'dX/dt = sin(Y) - bX',
            'dY/dt = sin(Z) - bY',
            'dZ/dt = sin(X) - bZ'
        ],
        'params': 'b=0.208186',
        'dt': 0.01,
        'total_steps': 400,
        'train_len': 300,
        'sindy_threshold': 0.05,
        'sindy_degree': 5,
    },
    'halvorsen': {
        'step_fn': halvorsen_step,
        'x0': [1.0, 0.0, 0.0],
        'name': 'Halvorsen',
        'var_names': ['X', 'Y', 'Z'],
        'equations': [
            'dX/dt = -aX - 4Y - 4Z - Y²',
            'dY/dt = -aY - 4Z - 4X - Z²',
            'dZ/dt = -aZ - 4X - 4Y - X²'
        ],
        'params': 'a=1.4',
        'dt': 0.01,
        'total_steps': 400,
        'train_len': 300,
        'sindy_threshold': 0.05,
        'sindy_degree': 2,
    },
    'rabinovich_fabrikant': {
        'step_fn': rabinovich_fabrikant_step,
        'x0': [0.1, 0.1, 0.1],
        'name': 'Rabinovich-Fabrikant',
        'var_names': ['X', 'Y', 'Z'],
        'equations': [
            'dX/dt = Y(Z - 1 + X²) + γX',
            'dY/dt = X(3Z + 1 - X²) + γY',
            'dZ/dt = -2Z(α + XY)'
        ],
        'params': 'α=0.14, γ=0.1',
        'dt': 0.01,
        'total_steps': 400,
        'train_len': 300,
        'sindy_threshold': 0.05,
        'sindy_degree': 3,
    },
    'sea_surface_temp': {
        'step_fn': sea_surface_temp_step,
        'x0': [1.0, 0.5, 0.5],
        'name': 'Sea-Surface Temperature',
        'var_names': ['Z₁', 'Z₂', 'Z₃'],
        'equations': [
            'dZ₁/dt = -0.01Z₁ + 6.24Z₂',
            'dZ₂/dt = -6.24Z₁ - 0.01Z₂',
            'dZ₃/dt = 0.02Z₃'
        ],
        'params': 'Linear system with complex eigenvalues',
        'dt': 0.01,
        'total_steps': 400,
        'train_len': 300,
        'sindy_threshold': 0.05,
        'sindy_degree': 1,
    },
    'ozone': {
        'step_fn': ozone_step,
        'x0': [1.0, 0.5, 0.5],
        'name': 'Ozone',
        'var_names': ['Z₁', 'Z₂', 'Z₃'],
        'equations': [
            'dZ₁/dt = -0.003Z₁ + 0.0079Z₂ - 0.002',
            'dZ₂/dt = -0.0079Z₁ - 0.003Z₂',
            'dZ₃/dt = -0.003Z₃ + 0.002'
        ],
        'params': 'Linear system with forcing',
        'dt': 0.01,
        'total_steps': 400,
        'train_len': 300,
        'sindy_threshold': 0.05,
        'sindy_degree': 1,
    },
    'cylinder_flow': {
        'step_fn': cylinder_flow_step,
        'x0': [1.0, 0.5, 0.2],
        'name': 'Cylinder Flow',
        'var_names': ['Z₁', 'Z₂', 'Z₃'],
        'equations': [
            'dZ₁/dt = -0.01Z₁ + 1.52Z₂',
            'dZ₂/dt = -1.52Z₁ - 0.01Z₂',
            'dZ₃/dt = -0.20Z₃'
        ],
        'params': '3D projection of 6D flow system',
        'dt': 0.01,
        'total_steps': 400,
        'train_len': 300,
        'sindy_threshold': 0.05,
        'sindy_degree': 1,
    }
}

def integrate_system(step_fn, x0, steps, dt=0.01):
    """RK4 integration for any 3D dynamical system"""
    traj = np.zeros((steps, 3), dtype=float)
    x = np.array(x0, dtype=float)
    for i in range(steps):
        k1 = step_fn(x)
        k2 = step_fn(x + 0.5 * dt * k1)
        k3 = step_fn(x + 0.5 * dt * k2)
        k4 = step_fn(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj[i] = x
    return traj

# --------------------------------------------------
# MASKING FUNCTIONS

def create_temporal_mask(n_timesteps, mask_ratio, mask_type='random', block_size=5):
    """Create a temporal mask (mask entire timesteps across all channels)"""
    mask = np.zeros(n_timesteps, dtype=bool)
    n_to_mask = int(n_timesteps * mask_ratio)
    
    if mask_type == 'random':
        mask_indices = np.random.choice(n_timesteps, size=n_to_mask, replace=False)
        mask[mask_indices] = True
    elif mask_type == 'contiguous':
        n_blocks = max(1, n_to_mask // block_size)
        for _ in range(n_blocks):
            start_idx = np.random.randint(0, n_timesteps - block_size + 1)
            mask[start_idx:start_idx + block_size] = True
    elif mask_type == 'periodic':
        period = int(1.0 / mask_ratio) if mask_ratio > 0 else n_timesteps
        for i in range(0, n_timesteps, period):
            end_idx = min(i + block_size, n_timesteps)
            mask[i:end_idx] = True
    
    return mask

def create_spatial_mask(n_timesteps, n_channels, channels_to_mask, mask_ratio):
    """Create a spatial mask (mask specific channels at selected timesteps)"""
    mask = np.zeros((n_timesteps, n_channels), dtype=bool)
    
    if len(channels_to_mask) > 0:
        n_to_mask = int(n_timesteps * mask_ratio)
        mask_timesteps = np.random.choice(n_timesteps, size=n_to_mask, replace=False)
        
        for channel_idx in channels_to_mask:
            if 0 <= channel_idx < n_channels:
                mask[mask_timesteps, channel_idx] = True
    
    return mask

def impute_data(data, mask, method='forward_fill'):
    """Impute missing values in data according to mask"""
    imputed = data.copy()
    
    if method == 'zero':
        imputed[mask] = 0.0
    elif method == 'mean':
        for ch in range(data.shape[1]):
            ch_mask = mask[:, ch] if mask.ndim == 2 else mask
            if ch_mask.any():
                observed_mean = data[~ch_mask, ch].mean()
                if mask.ndim == 2:
                    imputed[ch_mask, ch] = observed_mean
                else:
                    imputed[ch_mask, ch] = observed_mean
    elif method == 'forward_fill':
        for ch in range(data.shape[1]):
            ch_mask = mask[:, ch] if mask.ndim == 2 else mask
            if ch_mask.any():
                last_valid = None
                for t in range(len(data)):
                    if (mask.ndim == 2 and mask[t, ch]) or (mask.ndim == 1 and mask[t]):
                        if last_valid is not None:
                            imputed[t, ch] = last_valid
                    else:
                        last_valid = data[t, ch]
    elif method == 'interpolate':
        for ch in range(data.shape[1]):
            ch_mask = mask[:, ch] if mask.ndim == 2 else mask
            if ch_mask.any():
                valid_indices = np.where(~ch_mask)[0]
                missing_indices = np.where(ch_mask)[0]
                if len(valid_indices) >= 2:
                    imputed[missing_indices, ch] = np.interp(
                        missing_indices, 
                        valid_indices, 
                        data[valid_indices, ch]
                    )
    
    return imputed

# --------------------------------------------------
# Select and generate system

if SYSTEM not in SYSTEMS:
    raise ValueError(f"Unknown system: {SYSTEM}. Choose from {list(SYSTEMS.keys())}")

system_config = SYSTEMS[SYSTEM]
dt = system_config['dt']
total_steps = system_config['total_steps']
train_len = system_config['train_len']

print(f"\n{'='*60}")
print(f"Running Reservoir Computing on: {system_config['name']} System")
print(f"{'='*60}")
print(f"\nOriginal System Equations:")
for eq in system_config['equations']:
    print(f"  {eq}")
print(f"Parameters: {system_config['params']}")
print(f"Initial condition: {system_config['x0']}")
print(f"Integration: dt={dt}, total_steps={total_steps}, train_len={train_len}")
print(f"{'='*60}\n")

# Generate trajectory
traj = integrate_system(system_config['step_fn'], system_config['x0'], total_steps, dt=dt)

# Add observational noise
if noise_level > 0:
    traj_noisy = traj + np.random.normal(scale=noise_level, size=traj.shape)
else:
    traj_noisy = traj.copy()

if NORM_METHOD == 'none':
    # No normalization - work in original coordinates
    traj_norm = traj_noisy.copy()
    scale_factor = 1.0
    offset = 0.0
elif NORM_METHOD == 'robust_scale':
    # Scale to [-1, 1] range using min/max per dimension
    # This preserves relative structure better than standardization
    train_min = traj_noisy[:train_len].min(axis=0)
    train_max = traj_noisy[:train_len].max(axis=0)
    train_range = train_max - train_min
    scale_factor = 2.0 / (train_range + 1e-8)
    offset = train_min + train_range / 2.0
    traj_norm = (traj_noisy - offset) * scale_factor
elif NORM_METHOD == 'minmax':
    # Simple min-max to [0, 1]
    train_min = traj_noisy[:train_len].min(axis=0)
    train_max = traj_noisy[:train_len].max(axis=0)
    scale_factor = 1.0 / (train_max - train_min + 1e-8)
    offset = train_min
    traj_norm = (traj_noisy - offset) * scale_factor

def denorm(data_norm):
    if NORM_METHOD == 'none':
        return data_norm
    elif NORM_METHOD == 'robust_scale':
        return data_norm / scale_factor + offset
    elif NORM_METHOD == 'minmax':
        return data_norm / scale_factor + offset

# Prepare one-step-ahead forecasting dataset
X = traj_norm[:-1]
y = traj_norm[1:]

# Split train / test
X_train, y_train = X[:train_len], y[:train_len]
X_test, y_test = X[train_len:], y[train_len:]

# --------------------------------------------------
# APPLY MASKING TO TRAINING DATA

print(f"\n{'='*60}")
print("MASKING CONFIGURATION")
print(f"{'='*60}")

# Create combined mask
combined_mask = np.zeros((train_len, 3), dtype=bool)

# Temporal masking
if TEMPORAL_MASK_ENABLED and TEMPORAL_MASK_RATIO > 0:
    temporal_mask = create_temporal_mask(
        train_len, 
        TEMPORAL_MASK_RATIO, 
        TEMPORAL_MASK_TYPE, 
        TEMPORAL_MASK_BLOCK_SIZE
    )
    combined_mask[temporal_mask, :] = True
    n_temporal_masked = temporal_mask.sum()
    print(f"\nTemporal Masking: ENABLED")
    print(f"  Type: {TEMPORAL_MASK_TYPE}")
    print(f"  Ratio: {TEMPORAL_MASK_RATIO:.2%}")
    print(f"  Masked timesteps: {n_temporal_masked}/{train_len} ({100*n_temporal_masked/train_len:.1f}%)")
    if TEMPORAL_MASK_TYPE in ['contiguous', 'periodic']:
        print(f"  Block size: {TEMPORAL_MASK_BLOCK_SIZE}")
else:
    print(f"\nTemporal Masking: DISABLED")

# Spatial masking
if SPATIAL_MASK_ENABLED and len(SPATIAL_MASK_CHANNELS) > 0 and SPATIAL_MASK_RATIO > 0:
    spatial_mask = create_spatial_mask(
        train_len, 
        3, 
        SPATIAL_MASK_CHANNELS, 
        SPATIAL_MASK_RATIO
    )
    combined_mask = combined_mask | spatial_mask
    
    print(f"\nSpatial Masking: ENABLED")
    print(f"  Channels masked: {SPATIAL_MASK_CHANNELS} ({[system_config['var_names'][i] for i in SPATIAL_MASK_CHANNELS if i < 3]})")
    print(f"  Ratio per channel: {SPATIAL_MASK_RATIO:.2%}")
    for ch_idx in SPATIAL_MASK_CHANNELS:
        if ch_idx < 3:
            n_masked = spatial_mask[:, ch_idx].sum()
            print(f"  Channel {ch_idx} ({system_config['var_names'][ch_idx]}): {n_masked}/{train_len} masked ({100*n_masked/train_len:.1f}%)")
else:
    print(f"\nSpatial Masking: DISABLED")

total_masked = combined_mask.sum()
total_elements = combined_mask.size
print(f"\nTotal masked elements: {total_masked}/{total_elements} ({100*total_masked/total_elements:.1f}%)")
print(f"Imputation method: {IMPUTATION_METHOD}")
print(f"{'='*60}\n")

# Apply imputation
X_train_masked = X_train.copy()
y_train_masked = y_train.copy()
X_train_original = X_train.copy()

X_train_masked[combined_mask] = np.nan
X_train_masked = impute_data(X_train_masked, combined_mask, method=IMPUTATION_METHOD)

y_combined_mask = np.zeros_like(combined_mask)
y_combined_mask[:-1] = combined_mask[1:]
y_train_masked[y_combined_mask] = np.nan
y_train_masked = impute_data(y_train_masked, y_combined_mask, method=IMPUTATION_METHOD)

# --------------------------------------------------
# Build and train ESN

try:
    import reservoirpy as rpy
    from reservoirpy.nodes import Reservoir, Ridge
    from reservoirpy.observables import nrmse, rsquare
except Exception as e:
    raise ImportError("This script requires reservoirpy. Install with: pip install reservoirpy" + str(e))

rpy.set_seed(seed)

reservoir = Reservoir(units,
                      input_scaling=input_scaling,
                      sr=spectral_radius,
                      lr=leak_rate,
                      rc_connectivity=connectivity,
                      input_connectivity=input_connectivity,
                      seed=seed)

readout = Ridge(ridge=regularization, output_dim=3)
esn = reservoir >> readout

print("Training ESN on masked data...")
esn = esn.fit(X_train_masked, y_train_masked)

y_train_pred = esn.run(X_train_masked)

unmasked_indices = ~combined_mask.any(axis=1)
if unmasked_indices.any():
    y_train_unmasked = y_train[unmasked_indices]
    y_pred_unmasked = y_train_pred[unmasked_indices]
    r2_unmasked = rsquare(y_train_unmasked, y_pred_unmasked)
    nrmse_unmasked = nrmse(y_train_unmasked, y_pred_unmasked)
    print(f"Training performance on unmasked data: R²={r2_unmasked:.5f}, NRMSE={nrmse_unmasked:.5f}")

# --------------------------------------------------
# Estimate observational noise

val_len = min(50, X_test.shape[0] // 3)
X_val, y_val = X_test[:val_len], y_test[:val_len]

if X_val.shape[0] > 0:
    y_val_pred = esn.run(X_val)
    residuals = y_val - y_val_pred
    sigma_obs = np.std(residuals, axis=0)
else:
    residuals = y_train - y_train_pred
    sigma_obs = np.std(residuals, axis=0)

print("Estimated per-channel sigma_obs:", sigma_obs)
feedback_noise_sigma = feedback_noise_scale * sigma_obs
print("Using feedback_noise_sigma:", feedback_noise_sigma)

# --------------------------------------------------
# Generative MC

warm_start_idx = max(0, train_len - warmup_steps)
warm_inputs = X_train_masked[warm_start_idx:]

print(f"Using warmup from training data: indices {warm_start_idx} to {train_len}")

esn.nodes[0].reset()
warming_out = esn.run(warm_inputs)

mc_all = np.zeros((n_runs, generations, 3))

print(f"Running {n_runs} Monte-Carlo generations (each {generations} steps) ...")
for r in range(n_runs):
    if (r+1) % max(1, n_runs//10) == 0:
        print(f" MC run {r+1}/{n_runs}")
    
    esn.nodes[0].reset()
    warming_out_r = esn.run(warm_inputs)
    last = warming_out_r[-1].ravel()
    
    for t in range(generations):
        noise = np.random.normal(loc=0.0, scale=feedback_noise_sigma, size=last.shape)
        noisy_input = last + noise
        pred = esn(noisy_input)
        pred_vec = np.asarray(pred).ravel()
        mc_all[r, t, :] = pred_vec
        last = pred_vec

print("MC generation finished.")

mc_all_denorm = denorm(mc_all)
mc_mean = mc_all_denorm.mean(axis=0)
mc_std = mc_all_denorm.std(axis=0)

# --------------------------------------------------
# Plotting

plt.rcParams.update({'figure.max_open_warning': 0})

full_time = np.arange(traj_norm.shape[0])
full_denorm = denorm(traj_norm)
traj_train_denorm = denorm(y_train_pred)
pred_start_idx = train_len
gen_time = np.arange(pred_start_idx, pred_start_idx + generations)

# Time series plot
fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
coords = system_config['var_names']

for i, ax in enumerate(axs):
    ax.plot(full_time, full_denorm[:, i], lw=1.0, label='Original (noisy)', alpha=0.7, color='gray')
    ax.plot(np.arange(train_len), traj_train_denorm[:, i], lw=2.0, label='Model fit (training)', color='blue')
    
    masked_times = np.where(combined_mask[:, i])[0]
    if len(masked_times) > 0:
        ax.scatter(masked_times, denorm(X_train_original)[masked_times, i], 
                  c='red', s=20, alpha=0.6, label='Masked data', zorder=5)
    
    ax.plot(gen_time, mc_mean[:, i], lw=2.2, color='red', label='MC mean')
    ax.fill_between(gen_time,
                    mc_mean[:, i] - mc_std[:, i],
                    mc_mean[:, i] + mc_std[:, i],
                    alpha=0.25, label='±1 std (MC)', color='red')
    
    ax.axvline(pred_start_idx, color='k', linestyle='--', alpha=0.5)
    ax.set_ylabel(coords[i])
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

axs[-1].set_xlabel('Timestep')
plt.suptitle(f'{system_config["name"]} System: Training with Masking + MC Predictions')
plt.tight_layout(rect=[0, 0.03, 1, 0.96])

# 3D phase-space
fig2 = plt.figure(figsize=(14, 6))
ax3d = fig2.add_subplot(121, projection='3d')

slice_start = max(0, pred_start_idx - 200)
slice_end = pred_start_idx + generations
ax3d.plot(full_denorm[slice_start:slice_end, 0],
          full_denorm[slice_start:slice_end, 1],
          full_denorm[slice_start:slice_end, 2], 
          lw=0.8, label='Original', alpha=0.7)

plot_members = min(60, n_runs)
idxs = np.linspace(0, n_runs-1, plot_members, dtype=int)
for ix in idxs:
    member = mc_all_denorm[ix]
    ax3d.plot(member[:, 0], member[:, 1], member[:, 2], alpha=0.1, color='gray')

ax3d.plot(mc_mean[:, 0], mc_mean[:, 1], mc_mean[:, 2], 
         lw=2.5, color='red', label='MC mean')
ax3d.set_title(f'{system_config["name"]}: 3D phase space')
ax3d.set_xlabel(coords[0])
ax3d.set_ylabel(coords[1])
ax3d.set_zlabel(coords[2])
ax3d.legend()

ax3d2 = fig2.add_subplot(122, projection='3d')
warm_denorm = denorm(warming_out)
ax3d2.plot(warm_denorm[:, 0], warm_denorm[:, 1], warm_denorm[:, 2], 
           lw=2.0, linestyle='--', label='Warmup (training data)', color='blue')
for ix in idxs:
    member = mc_all_denorm[ix]
    ax3d2.plot(member[:, 0], member[:, 1], member[:, 2], alpha=0.1, color='gray')
ax3d2.plot(mc_mean[:, 0], mc_mean[:, 1], mc_mean[:, 2], 
          lw=2.5, color='red', label='MC mean')
ax3d2.scatter(warm_denorm[-1, 0], warm_denorm[-1, 1], warm_denorm[-1, 2], 
              s=100, c='green', marker='o', label='Generation start', 
              edgecolors='black', linewidths=2)
ax3d2.set_title('Warmup → Generative continuation')
ax3d2.set_xlabel(coords[0])
ax3d2.set_ylabel(coords[1])
ax3d2.set_zlabel(coords[2])
ax3d2.legend()

plt.tight_layout()

# Masking pattern visualization
fig3, axs3 = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
for i, ax in enumerate(axs3):
    train_time = np.arange(train_len)
    
    ax.plot(train_time, denorm(X_train_original)[:, i], 
            'o-', markersize=3, alpha=0.3, label='Original', color='gray')
    
    ax.plot(train_time, denorm(X_train_masked)[:, i], 
            'o-', markersize=2, alpha=0.8, label='After imputation', color='blue')
    
    masked_idx = np.where(combined_mask[:, i])[0]
    if len(masked_idx) > 0:
        ax.scatter(masked_idx, denorm(X_train_original)[masked_idx, i],
                  c='red', s=40, marker='x', linewidths=2, 
                  label=f'Masked ({len(masked_idx)} pts)', zorder=10)
    
    ax.set_ylabel(coords[i])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Channel {i} ({coords[i]}): Masking Pattern')

axs3[-1].set_xlabel('Training Timestep')
plt.suptitle(f'Masking Pattern Visualization (Method: {IMPUTATION_METHOD})')
plt.tight_layout(rect=[0, 0.03, 1, 0.96])

# Mask statistics heatmap
fig4, (ax_temp, ax_spatial) = plt.subplots(1, 2, figsize=(14, 5))

if TEMPORAL_MASK_ENABLED and TEMPORAL_MASK_RATIO > 0:
    temporal_mask_viz = temporal_mask.astype(float).reshape(-1, 1)
    im1 = ax_temp.imshow(temporal_mask_viz.T, aspect='auto', cmap='RdYlGn_r', 
                         interpolation='nearest')
    ax_temp.set_title(f'Temporal Mask Pattern\n({TEMPORAL_MASK_TYPE}, ratio={TEMPORAL_MASK_RATIO:.2%})')
    ax_temp.set_xlabel('Timestep')
    ax_temp.set_yticks([])
    ax_temp.set_ylabel('All Channels')
    plt.colorbar(im1, ax=ax_temp, label='Masked (1) / Observed (0)')
else:
    ax_temp.text(0.5, 0.5, 'Temporal Masking\nDISABLED', 
                ha='center', va='center', transform=ax_temp.transAxes, fontsize=14)
    ax_temp.set_xticks([])
    ax_temp.set_yticks([])

if SPATIAL_MASK_ENABLED and len(SPATIAL_MASK_CHANNELS) > 0:
    im2 = ax_spatial.imshow(spatial_mask.T, aspect='auto', cmap='RdYlGn_r', 
                           interpolation='nearest')
    ax_spatial.set_title(f'Spatial Mask Pattern\n(Channels {SPATIAL_MASK_CHANNELS}, ratio={SPATIAL_MASK_RATIO:.2%})')
    ax_spatial.set_xlabel('Timestep')
    ax_spatial.set_ylabel('Channel')
    ax_spatial.set_yticks(range(3))
    ax_spatial.set_yticklabels(coords)
    plt.colorbar(im2, ax=ax_spatial, label='Masked (1) / Observed (0)')
else:
    ax_spatial.text(0.5, 0.5, 'Spatial Masking\nDISABLED', 
                   ha='center', va='center', transform=ax_spatial.transAxes, fontsize=14)
    ax_spatial.set_xticks([])
    ax_spatial.set_yticks([])

plt.tight_layout()
plt.show()

print('\nScript finished.')

def compute_r2(y_true, y_pred):
    """Compute R² score"""
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean(axis=0))**2)
    return 1 - ss_res / (ss_tot + 1e-12)

def compute_nrmse(y_true, y_pred):
    """Compute normalized RMSE"""
    mse = np.mean((y_true - y_pred)**2)
    return np.sqrt(mse) / (np.std(y_true) + 1e-12)


# --------------------------------------------------
# Metrics and evaluation

X_true_future = None
if y_test.shape[0] >= generations:
    X_true_future = denorm(y_test[:generations])
    
    mean_gen_norm = mc_mean  # Already denormalized
    X_true_norm = denorm(y_test[:generations])
    
    r2 = compute_r2(X_true_norm, mean_gen_norm)
    nrm = compute_nrmse(X_true_norm, mean_gen_norm)
    
    print(f"\n{'='*60}")
    print("PREDICTION METRICS")
    print(f"{'='*60}")
    print(f"MC generative metrics (mean vs truth):")
    print(f"  R² = {r2:.5f}")
    print(f"  NRMSE = {nrm:.5f}")
    
    # Empirical coverage
    z95 = 1.96
    lower = mc_mean - z95 * mc_std
    upper = mc_mean + z95 * mc_std
    in_band = (X_true_future >= lower) & (X_true_future <= upper)
    coverage_per_channel = in_band.mean(axis=0)
    coverage_all_three = in_band.all(axis=1).mean()
    
    print(f"\nEmpirical coverage (95% confidence band):")
    for i, ch in enumerate(coords):
        print(f"  {ch}: {coverage_per_channel[i]:.2%}")
    print(f"  All channels: {coverage_all_three:.2%}")
    
    # Per-channel metrics
    print(f"\nPer-channel metrics:")
    for i, ch in enumerate(coords):
        r2_ch = compute_r2(X_true_norm[:, i:i+1], mean_gen_norm[:, i:i+1])
        nrmse_ch = compute_nrmse(X_true_norm[:, i:i+1], mean_gen_norm[:, i:i+1])
        print(f"  {ch}: R²={r2_ch:.5f}, NRMSE={nrmse_ch:.5f}")
    
    print(f"{'='*60}\n")

# ---------- SINDy integration ----------
try:
    import pysindy as ps
except Exception as e:
    print("\nSkipping SINDy analysis (pysindy not installed).")
    print("Install with: pip install pysindy")
    import sys
    sys.exit(0)

print("\n" + "="*60)
print("SINDy Model Discovery")
print("="*60)

# Baseline SINDy: train only on original data
print("\n=== Baseline SINDy (Original Data Only) ===")

t_orig = np.arange(train_len + 1) * dt
orig_data_denorm = full_denorm[:train_len + 1]

# System-specific SINDy configuration
sindy_threshold = system_config['sindy_threshold']
sindy_degree = system_config['sindy_degree']

print(f"SINDy parameters: threshold={sindy_threshold}, polynomial_degree={sindy_degree}")

# Use more robust differentiation for noisy data
differentiation_method = ps.SmoothedFiniteDifference(
    smoother_kws={'window_length': 11, 'polyorder': 5}
)

# Adjust feature library based on system
if sindy_degree >= 5:
    # For Thomas attractor (needs trig functions)
    feature_library = ps.FourierLibrary(n_frequencies=2)
    print("Using Fourier library for trigonometric terms")
else:
    feature_library = ps.PolynomialLibrary(
        degree=sindy_degree, 
        include_interaction=True, 
        include_bias=True  # Include constant terms for systems with forcing
    )

optimizer = ps.STLSQ(threshold=sindy_threshold, alpha=0.01, max_iter=100)

model_orig = ps.SINDy(
    feature_library=feature_library,
    optimizer=optimizer,
    differentiation_method=differentiation_method,
)

try:
    model_orig.fit(orig_data_denorm, t=t_orig, feature_names=coords)
    
    print("\nDiscovered baseline SINDy model (original data only):")
    model_orig.print()
    
    coefs_orig = model_orig.coefficients()
    n_total_orig = coefs_orig.size
    n_nonzero_orig = np.count_nonzero(np.abs(coefs_orig) > 1e-10)
    sparsity_orig = 1.0 - (n_nonzero_orig / n_total_orig)
    print(f"\nSINDy statistics (original):")
    print(f"  Total coefficients: {n_total_orig}")
    print(f"  Nonzero coefficients: {n_nonzero_orig}")
    print(f"  Sparsity: {sparsity_orig:.3f}")
    
    # Validate on training data
    x_dot_orig = model_orig.differentiate(orig_data_denorm, t=t_orig)
    x_dot_pred_orig = model_orig.predict(orig_data_denorm)
    
    r2_deriv_orig = compute_r2(x_dot_orig, x_dot_pred_orig)
    print(f"  R² (derivative fit): {r2_deriv_orig:.5f}")
    
except Exception as e:
    print(f"Error fitting baseline SINDy model: {e}")
    model_orig = None

# Extended SINDy with MC predictions
print("\n=== Extended SINDy (Original + MC Predictions) ===")

# Adaptive uncertainty threshold
THRESHOLD_MODE = 'adaptive_growth'

if THRESHOLD_MODE == 'fixed':
    uncertainty_threshold = 0.5
    print(f"Using FIXED threshold: {uncertainty_threshold}")
elif THRESHOLD_MODE == 'percentile':
    percentile_value = 50
    all_stds = mc_std.flatten()
    uncertainty_threshold = np.percentile(all_stds, percentile_value)
    print(f"Using PERCENTILE threshold ({percentile_value}th): {uncertainty_threshold:.4f}")
elif THRESHOLD_MODE == 'baseline_multiple':
    multiplier = 1.5
    uncertainty_threshold = multiplier * np.mean(sigma_obs)
    print(f"Using BASELINE MULTIPLE threshold ({multiplier}x): {uncertainty_threshold:.4f}")
elif THRESHOLD_MODE == 'adaptive_growth':
    initial_window = min(20, generations // 10)
    initial_std = mc_std[:initial_window].mean()
    growth_factor = 2.0
    uncertainty_threshold = growth_factor * initial_std
    print(f"Using ADAPTIVE GROWTH threshold ({growth_factor}x initial): {uncertainty_threshold:.4f}")
    print(f"  Initial std: {initial_std:.4f}")
elif THRESHOLD_MODE == 'snr':
    signal_magnitude = np.mean(np.abs(full_denorm[:train_len]))
    target_snr = 10.0
    uncertainty_threshold = signal_magnitude / target_snr
    print(f"Using SNR-based threshold (SNR={target_snr}): {uncertainty_threshold:.4f}")

# Find acceptance cutoff
exceed_idx = np.where((mc_std > uncertainty_threshold).any(axis=1))[0]
if exceed_idx.size > 0:
    accepted_len = int(exceed_idx[0])
    print(f"\nUncertainty exceeds threshold at step {accepted_len}/{generations}")
    print(f"  Accepting {accepted_len} generated timesteps ({100*accepted_len/generations:.1f}%)")
else:
    accepted_len = generations
    print(f"\nAll {generations} steps accepted (max std: {mc_std.max():.4f})")

if accepted_len <= 10:
    print("Too few timesteps accepted - using minimum of 50 or all available")
    accepted_len = min(50, generations)

# Combine datasets
orig_data_denorm = full_denorm[:train_len+1]
appended = mc_mean[:accepted_len]
combined_data_denorm = np.vstack([orig_data_denorm, appended])
t_combined = np.arange(combined_data_denorm.shape[0]) * dt

print(f"Combined dataset: {combined_data_denorm.shape[0]} timesteps")
print(f"  Original: {orig_data_denorm.shape[0]}")
print(f"  Appended: {appended.shape[0]}")

# Fit extended model
model_extended = ps.SINDy(
    feature_library=feature_library,
    optimizer=optimizer,
    differentiation_method=differentiation_method
)

try:
    model_extended.fit(combined_data_denorm, t=t_combined, feature_names=coords)
    
    print("\nDiscovered extended SINDy model (with MC predictions):")
    model_extended.print()
    
    coefs_ext = model_extended.coefficients()
    n_total_ext = coefs_ext.size
    n_nonzero_ext = np.count_nonzero(np.abs(coefs_ext) > 1e-10)
    sparsity_ext = 1.0 - (n_nonzero_ext / n_total_ext)
    print(f"\nSINDy statistics (extended):")
    print(f"  Total coefficients: {n_total_ext}")
    print(f"  Nonzero coefficients: {n_nonzero_ext}")
    print(f"  Sparsity: {sparsity_ext:.3f}")
    
    # Derivative fit
    x_dot_ext = model_extended.differentiate(combined_data_denorm, t=t_combined)
    x_dot_pred_ext = model_extended.predict(combined_data_denorm)
    r2_deriv_ext = compute_r2(x_dot_ext, x_dot_pred_ext)
    print(f"  R² (derivative fit): {r2_deriv_ext:.5f}")
    
    # Simulation validation
    if X_true_future is not None:
        gen_start_idx = train_len
        init_state = full_denorm[gen_start_idx - 1]
        sim_horizon = min(accepted_len, X_true_future.shape[0])
        t_sim = np.arange(sim_horizon) * dt
        
        try:
            sindy_sim = model_extended.simulate(init_state, t_sim)
            if isinstance(sindy_sim, list):
                sindy_sim = np.array(sindy_sim)
            elif hasattr(sindy_sim, 'values'):
                sindy_sim = sindy_sim.values
            sindy_sim = np.asarray(sindy_sim)
            
            if sim_horizon > 0:
                truth = X_true_future[:sim_horizon]
                if truth.shape[0] == sindy_sim.shape[0]:
                    r2_sim = compute_r2(truth, sindy_sim)
                    nrmse_sim = compute_nrmse(truth, sindy_sim)
                    
                    print(f"\nSimulation validation (horizon={sim_horizon} steps):")
                    print(f"  R²: {r2_sim:.5f}")
                    print(f"  NRMSE: {nrmse_sim:.5f}")
                    
                    # Per-channel
                    for i, ch in enumerate(coords):
                        r2_ch = compute_r2(truth[:, i:i+1], sindy_sim[:, i:i+1])
                        print(f"  {ch} R²: {r2_ch:.5f}")
        
        except Exception as e:
            print(f"Simulation validation failed: {e}")
    
    # Compare models
    if model_orig is not None:
        print(f"\n{'='*60}")
        print("MODEL COMPARISON")
        print(f"{'='*60}")
        print(f"Baseline model: {n_nonzero_orig} terms, sparsity={sparsity_orig:.3f}")
        print(f"Extended model: {n_nonzero_ext} terms, sparsity={sparsity_ext:.3f}")
        print(f"Change in sparsity: {sparsity_ext - sparsity_orig:+.3f}")
    
except Exception as e:
    print(f"Error fitting extended SINDy model: {e}")

print("\n" + "="*60)
print("Analysis Complete")
print("="*60)