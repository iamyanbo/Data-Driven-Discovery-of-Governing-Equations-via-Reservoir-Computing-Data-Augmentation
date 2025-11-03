# Reservoir Computing — Lorenz generative demo (Monte‑Carlo noise injection)
# Save and run in an environment with ReservoirPy installed (pip install reservoirpy).
# This script trains a single ESN, estimates observational noise from one-step residuals,
# then performs Monte‑Carlo closed-loop generations by injecting feedback noise.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------------- CONFIG --------------------------------
noise_level = 0.05   # observational noise added to Lorenz data (sigma)
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
n_runs = 300                  # number of Monte-Carlo runs (increase for smoother uncertainty)
feedback_noise_scale = 0.8    # multiply sigma_obs by this to tune band width

# Lorenz system params
sigma_l = 10.0
rho = 28.0
beta = 8.0 / 3.0
dt = 0.01
total_steps = 500

# training split
train_len = 200

# --------------------------------------------------
# Build Lorenz time series (RK4 integrator)

def lorenz_step(x, sigma_l=10.0, rho=28.0, beta=8.0/3.0):
    x0, y0, z0 = x
    dx = sigma_l * (y0 - x0)
    dy = x0 * (rho - z0) - y0
    dz = x0 * y0 - beta * z0
    return np.array([dx, dy, dz])

def integrate_lorenz(x0, steps, dt=0.01):
    traj = np.zeros((steps, 3), dtype=float)
    x = np.array(x0, dtype=float)
    for i in range(steps):
        k1 = lorenz_step(x)
        k2 = lorenz_step(x + 0.5 * dt * k1)
        k3 = lorenz_step(x + 0.5 * dt * k2)
        k4 = lorenz_step(x + dt * k3)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        traj[i] = x
    return traj

# initial condition
x0 = [1.0, 1.0, 1.0]
traj = integrate_lorenz(x0, total_steps, dt=dt)

# add configurable observational noise
if noise_level > 0:
    traj_noisy = traj + np.random.normal(scale=noise_level, size=traj.shape)
else:
    traj_noisy = traj.copy()

# normalize data to [-1, 1] per channel
mins = traj_noisy.min(axis=0)
maxs = traj_noisy.max(axis=0)
traj_norm = 2 * (traj_noisy - mins) / (maxs - mins) - 1

# helpers
def denorm(data_norm):
    return (data_norm + 1) / 2 * (maxs - mins) + mins

# prepare one-step-ahead forecasting dataset
X = traj_norm[:-1]
y = traj_norm[1:]

# split train / test / val
X_train, y_train = X[:train_len], y[:train_len]
X_test, y_test = X[train_len:], y[train_len:]

# choose a small validation slice from the test set to estimate residuals
val_len = min(500, X_test.shape[0] // 3)
X_val, y_val = X_test[:val_len], y_test[:val_len]

# ----------------- Build and train ESN -----------------
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

# fit one-step-ahead (offline)
esn = esn.fit(X_train, y_train)

# model fit on training period (predictions)
y_train_pred = esn.run(X_train)

# ---------------- Estimate observational noise sigma_obs ----------------
# Use residuals on validation (or on training if no val available)
if X_val.shape[0] > 0:
    y_val_pred = esn.run(X_val)
    residuals = y_val - y_val_pred
    sigma_obs = np.std(residuals, axis=0)
else:
    # fallback: use residuals on training
    residuals = y_train - y_train_pred
    sigma_obs = np.std(residuals, axis=0)

print("Estimated per-channel sigma_obs:", sigma_obs)
feedback_noise_sigma = feedback_noise_scale * sigma_obs
print("Using feedback_noise_sigma:", feedback_noise_sigma)

# ----------------- Generative MC (noise injection) --------------------
# MODIFIED: Use the last warmup_steps from training data instead of test data
warm_start_idx = max(0, train_len - warmup_steps)
warm_inputs = X_train[warm_start_idx:]  # last warmup_steps of training data

print(f"Using warmup from training data: indices {warm_start_idx} to {train_len}")

# warmup with trained ESN to obtain starting state
esn.nodes[0].reset()
warming_out = esn.run(warm_inputs)  # shape (warmup_steps, 3)

# prepare storage for MC runs (in normalized space)
mc_all = np.zeros((n_runs, generations, 3))

print(f"Running {n_runs} Monte-Carlo generations (each {generations} steps) ...")
for r in range(n_runs):
    if (r+1) % max(1, n_runs//10) == 0:
        print(f" MC run {r+1}/{n_runs}")
    # reset reservoir to the warmed-up state by replaying warm_inputs into a fresh reservoir instance
    # simpler: reset states and re-run warming inputs to reproduce same internal state
    esn.nodes[0].reset()
    warming_out_r = esn.run(warm_inputs)
    last = warming_out_r[-1].ravel()  # 1D
    for t in range(generations):
        # inject Gaussian noise proportional to sigma_obs
        noise = np.random.normal(loc=0.0, scale=feedback_noise_sigma, size=last.shape)
        noisy_input = last + noise
        pred = esn(noisy_input)  # expects 1D vector
        pred_vec = np.asarray(pred).ravel()
        mc_all[r, t, :] = pred_vec
        last = pred_vec

print("MC generation finished.")

# denormalize MC ensemble members
mc_all_denorm = denorm(mc_all)  # shape (n_runs, generations, 3)

# compute mean and std across runs
mc_mean = mc_all_denorm.mean(axis=0)
mc_std  = mc_all_denorm.std(axis=0)

# prepare plotting pieces
full_time = np.arange(traj_norm.shape[0])
full_denorm = denorm(traj_norm)

# compute training fit denormalized
traj_train_denorm = denorm(y_train_pred)

# MODIFIED: prediction starts right after training period (no test warmup offset)
pred_start_idx = train_len
gen_time = np.arange(pred_start_idx, pred_start_idx + generations)

# ---------------- Metrics: compare mean to ground truth where available ----------------
# MODIFIED: true future starts from train_len (no warmup offset in test)
if y_test.shape[0] >= generations:
    X_true_future = denorm(y_test[:generations])
else:
    X_true_future = None

# compute simple metrics on normalized space for consistency
if X_true_future is not None:
    # compute metrics using reservoirpy observables on normalized values
    # regenerate normalized mean for metrics (inverse of denorm)
    mean_gen_norm = (mc_mean - mins) / (maxs - mins) * 2.0 - 1.0
    X_true_norm = y_test[:generations]
    r2 = rsquare(X_true_norm, mean_gen_norm)
    nrm = nrmse(X_true_norm, mean_gen_norm)
    print(f"MC generative metrics (mean vs truth): R2 = {r2:.5f}, NRMSE = {nrm:.5f}")
else:
    print("Not enough ground truth to evaluate full generated segment.")

# empirical coverage for 95% band
z95 = 1.96
lower = mc_mean - z95 * mc_std
upper = mc_mean + z95 * mc_std
if X_true_future is not None:
    in_band = (X_true_future >= lower) & (X_true_future <= upper)  # shape (generations, 3)
    coverage_per_channel = in_band.mean(axis=0)
    coverage_all_three = in_band.all(axis=1).mean()
    print(f"Empirical coverage per channel (95% band): {coverage_per_channel}")
    print(f"Fraction of timesteps where all 3 channels inside 95% band: {coverage_all_three:.3f}")

# ----------------- Plotting ---------------------------
plt.rcParams.update({'figure.max_open_warning': 0})

# 1) Time series per channel (X,Y,Z) with MC mean and shaded ±1 std
fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
coords = ['X', 'Y', 'Z']

for i, ax in enumerate(axs):
    ax.plot(full_time, full_denorm[:, i], lw=1.0, label='Original (noisy)')
    ax.plot(np.arange(train_len), traj_train_denorm[:, i], lw=2.0, label='Model fit (training)')
    ax.plot(gen_time, mc_mean[:, i], lw=2.2, color='red', label='MC mean')
    ax.fill_between(gen_time,
                    mc_mean[:, i] - mc_std[:, i],
                    mc_mean[:, i] + mc_std[:, i],
                    alpha=0.25, label='±1 std (MC)')
    ax.axvline(pred_start_idx, color='k', linestyle='--')
    ax.set_ylabel(coords[i])
    ax.legend(loc='upper right')

axs[-1].set_xlabel('Timestep')
plt.suptitle('Lorenz: original, training fit, and MC generative prediction (mean ± std)')
plt.tight_layout(rect=[0, 0.03, 1, 0.96])

# 2) 3D phase-space: original vs MC ensemble
fig2 = plt.figure(figsize=(12, 6))
ax3d = fig2.add_subplot(121, projection='3d')

slice_start = max(0, pred_start_idx - 200)
slice_end = pred_start_idx + generations
ax3d.plot(full_denorm[slice_start:slice_end, 0],
          full_denorm[slice_start:slice_end, 1],
          full_denorm[slice_start:slice_end, 2], lw=0.8, label='Original (around prediction)')

# plot a subsample of MC members faintly to avoid overcrowding
plot_members = min(60, n_runs)
idxs = np.linspace(0, n_runs-1, plot_members, dtype=int)
for ix in idxs:
    member = mc_all_denorm[ix]
    ax3d.plot(member[:, 0], member[:, 1], member[:, 2], alpha=0.12)

# plot MC mean
ax3d.plot(mc_mean[:, 0], mc_mean[:, 1], mc_mean[:, 2], lw=2.5, color='red', label='MC mean')
ax3d.set_title('3D phase space: MC ensemble + mean')
ax3d.set_xlabel('X')
ax3d.set_ylabel('Y')
ax3d.set_zlabel('Z')
ax3d.legend()

# 3) warmup -> generation zoom
ax3d2 = fig2.add_subplot(122, projection='3d')
warm_denorm = denorm(warming_out)
ax3d2.plot(warm_denorm[:, 0], warm_denorm[:, 1], warm_denorm[:, 2], lw=1.5, linestyle='--', label='Warmup (training data)')
for ix in idxs:
    member = mc_all_denorm[ix]
    ax3d2.plot(member[:, 0], member[:, 1], member[:, 2], alpha=0.12)
ax3d2.plot(mc_mean[:, 0], mc_mean[:, 1], mc_mean[:, 2], lw=2.5, color='red', label='MC mean')
ax3d2.scatter(warm_denorm[-1, 0], warm_denorm[-1, 1], warm_denorm[-1, 2], s=80, c='k', label='Generation start')
ax3d2.set_title('Warmup -> Generative continuation (3D)')
ax3d2.set_xlabel('X')
ax3d2.set_ylabel('Y')
ax3d2.set_zlabel('Z')
ax3d2.legend()

plt.tight_layout()
plt.show()

# Done
print('Script finished.')

# ---------- SINDy integration: append predicted data while uncertainty <= threshold ----------
# Requires: pip install pysindy
import numpy as np

try:
    import pysindy as ps
except Exception as e:
    raise ImportError("Install pysindy first: pip install pysindy\n" + str(e))

# ---------- Baseline SINDy: train only on original noisy data (no extensions) ----------
print("\n=== Baseline SINDy on Original Data ===")

# build time vector for the original training data only
t_orig = np.arange(train_len + 1) * dt
orig_data_denorm = denorm(traj_norm[:train_len + 1])  # same as used in combined set, denormalized

# configure same differentiation + library + optimizer as extended SINDy
differentiation_method = ps.SmoothedFiniteDifference(smoother_kws={'window_length': 9, 'polyorder': 3})
feature_library = ps.PolynomialLibrary(degree=3, include_interaction=True, include_bias=False)
optimizer = ps.STLSQ(threshold=0.08, alpha=0.0)

# build and fit model
model_orig = ps.SINDy(
    feature_library=feature_library,
    optimizer=optimizer,
    differentiation_method=differentiation_method,
)
model_orig.fit(orig_data_denorm, t=t_orig, feature_names=['x', 'y', 'z'])

# print discovered equations
print("\nDiscovered baseline SINDy model (original data only):")
model_orig.print()

# compute sparsity metrics
coefs_orig = model_orig.coefficients()
n_total_orig = coefs_orig.size
n_nonzero_orig = np.count_nonzero(coefs_orig)
sparsity_orig = 1.0 - (n_nonzero_orig / n_total_orig)
print(f"SINDy coefficients (original): total={n_total_orig}, nonzero={n_nonzero_orig}, sparsity={sparsity_orig:.3f}")

print("=== End Baseline SINDy ===\n")


# Config (tune these)
uncertainty_threshold = 1   # default: 3 * observational noise (denorm units)
# NOTE: mc_mean, mc_std are currently DENORMALIZED in your script (mc_all_denorm -> mc_mean, mc_std)
# If mc_std was computed in normalized units change threshold accordingly.

# If your mc_std is normalized (it was computed on denorm arrays above),
# ensure the threshold is in the same units; here we assume denormalized units.

# 1) Find accepted generation length: stop at first t where ANY channel std > threshold
# mc_std shape: (generations, 3)
exceed_idx = np.where((mc_std > uncertainty_threshold).any(axis=1))[0]
if exceed_idx.size > 0:
    accepted_len = int(exceed_idx[0])  # do NOT include this timestep (stop just before first exceed)
    print(f"SINDy: stopping appended predictions at generation index {accepted_len} (first exceed at {exceed_idx[0]})")
else:
    accepted_len = generations
    print(f"SINDy: all {generations} steps under threshold -> accepted_len = {accepted_len}")

if accepted_len <= 0:
    raise RuntimeError("No generated timesteps accepted under the given uncertainty_threshold. "
                       "Lower threshold or increase MC smoothing / reduce noise.")

# 2) Build combined dataset: original noisy data up to training period, then appended accepted MC predictions
# The user requested: "original noisy data up to the training period" -> use traj_noisy up to train_len (one-step indices)
# Align shapes carefully. SINDy expects data as (n_samples, n_states).
# Use denormalized data for SINDy (derivative estimators will expect physical units).
orig_data_denorm = full_denorm[:train_len+1]  # +1 because X/y shift; contains states up to train end
# mc_mean is of shape (generations, 3) and denormalized already (mc_mean variable from your script)
appended = mc_mean[:accepted_len]  # accepted part of generative mean in denorm units

# combine (concatenate along time)
combined_data_denorm = np.vstack([orig_data_denorm, appended])

# create time vector for combined data
t_combined = np.arange(combined_data_denorm.shape[0]) * dt  # simple uniform sampling

print(f"Combined dataset length for SINDy: {combined_data_denorm.shape[0]} timesteps "
      f"({orig_data_denorm.shape[0]} original + {appended.shape[0]} appended).")

# 3) Configure PySINDy model (Polynomial library + Smoothed finite differences + STLSQ)
differentiation_method = ps.SmoothedFiniteDifference(smoother_kws={'window_length': 9, 'polyorder': 3})
# for noisy data you can also try total_variation regularization:
# differentiation_method = ps.TotalVariationRegularizedDifferentiation(differentiation_method='SAV', alpha=1e-4)

feature_library = ps.PolynomialLibrary(degree=3, include_interaction=True, include_bias=False)
optimizer = ps.STLSQ(threshold=0.08, alpha=0.0)  # threshold tuned for sparsity; adjust if needed

model = ps.SINDy(feature_library=feature_library,
                 optimizer=optimizer,
                 differentiation_method=differentiation_method,
                 )

# 4) Fit SINDy on combined dataset
model.fit(combined_data_denorm, t=t_combined, feature_names=['x', 'y', 'z'])
print("\nDiscovered SINDy model:")
model.print()

# 5) Sparsity: fraction nonzero coefficients
coefs = model.coefficients()  # shape (n_eqns, n_features)
n_total = coefs.size
n_nonzero = np.count_nonzero(coefs)
sparsity = 1.0 - (n_nonzero / n_total)
print(f"SINDy coefficients: total={n_total}, nonzero={n_nonzero}, sparsity={sparsity:.3f}")

# 6) Validation: simulate SINDy forward from generation start and compare to ground truth (if available)
# MODIFIED: choose initial condition at the end of training (start of generation)
gen_start_idx = train_len
init_state = full_denorm[gen_start_idx - 1]  # last state of training period

# simulation horizon: compare on accepted_len steps or on true future if present
sim_horizon = accepted_len
t_sim = np.arange(sim_horizon) * dt

try:
    sindy_sim = model.simulate(init_state, t_sim)  # shape (sim_horizon, 3)
except Exception as e:
    # fallback: use predict step-by-step
    sindy_sim = np.zeros((sim_horizon, 3))
    s = init_state.copy()
    for i in range(sim_horizon):
        ds = model.predict(s.reshape(1, -1))  # one-step derivative estimate via model.predict (if available)
        # predict expects state -> returns derivative if SINDy is a continuous model; we'll do Euler step
        s = s + ds.ravel() * dt
        sindy_sim[i] = s

# compare to true future if available
if X_true_future is not None:
    # X_true_future is denorm true future in your script; ensure we compare same length
    truth = X_true_future[:sim_horizon]
    # compute simple metrics
    def nrmse(a, b):
        return np.sqrt(np.mean((a - b)**2)) / (np.sqrt(np.mean((a)**2)) + 1e-12)
    # channel-wise
    nrmse_vals = [nrmse(truth[:, i], sindy_sim[:, i]) for i in range(3)]
    # R^2 (basic)
    ss_res = np.sum((truth - sindy_sim)**2, axis=0)
    ss_tot = np.sum((truth - truth.mean(axis=0))**2, axis=0)
    r2_vals = 1 - ss_res / (ss_tot + 1e-12)
    print(f"SINDy validation (sim horizon {sim_horizon}): NRMSE per channel = {nrmse_vals}, R2 per channel = {r2_vals}")
else:
    print("No ground truth available for the full simulated horizon — only qualitative comparison possible.")

# 7) (Optional) Ensemble SINDy to stabilize: bootstrap and aggregate coefficients
def ensemble_sindy(data, t, n_ensembles=40, sample_frac=0.8, seed=seed):
    rng = np.random.RandomState(seed)
    coef_list = []
    for k in range(n_ensembles):
        # bootstrap: sample contiguous blocks might be better for time-series; here we sample indices
        n = data.shape[0]
        # sample contiguous block start
        if n == 0:
            continue
        start = rng.randint(0, max(1, n - int(sample_frac*n)))
        end = start + int(sample_frac*n)
        subset = data[start:end]
        t_sub = (np.arange(subset.shape[0]) * dt)
        m = ps.SINDy(feature_library=feature_library,
                     optimizer=ps.STLSQ(threshold=optimizer.threshold),
                     differentiation_method=differentiation_method,
                     )
        try:
            m.fit(subset, t=t_sub, feature_names=['x', 'y', 'z'])
            coef_list.append(m.coefficients())
        except Exception:
            continue
    if len(coef_list) == 0:
        return None
    coefs_stack = np.stack(coef_list, axis=0)
    median_coefs = np.median(coefs_stack, axis=0)
    return median_coefs

# run ensemble if you want to stabilize (this is optional and may take time)
run_ensemble = True
if run_ensemble:
    print("Running Ensemble-SINDy (bootstrap aggregation)...")
    median_coefs = ensemble_sindy(combined_data_denorm, t_combined, n_ensembles=40, sample_frac=0.75, seed=seed)
    if median_coefs is not None:
        print("Median coefficient matrix from ensemble (shape):", median_coefs.shape)
        # Build a new model using the median coefficients
        model_median = ps.SINDy(feature_library=feature_library,
                                optimizer=optimizer,
                                differentiation_method=differentiation_method)
        model_median.fit(combined_data_denorm, t=t_combined, feature_names=['x', 'y', 'z'])
        # overwrite its coefficients with the median
        model_median.coefficients = lambda: median_coefs
        print("\nMedian (Ensemble) SINDy model:")
        model_median.print()
    else:
        print("Ensemble-SINDy produced no valid models.")