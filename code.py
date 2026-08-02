import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma
from scipy.linalg import cholesky
from numpy.linalg import solve

# === Parameters ===
T_total = 15.0       # simulate P(t), I(t) on [0, 15] s
T_obs = 10.0         # we only "observe" up to 10 s
dt = 0.001           # fine time step for continuous path (1000 Hz)
t = np.arange(0, T_total, dt)
N = len(t)

fs_sample = 100.0     # sampling of |I|^2 (0.1 kHz)
sample_interval = 1.0 / fs_sample  # 0.1 s
# indices of sampled points up to T_obs
indices = np.arange(0, int(T_obs / dt), int(sample_interval / dt))
t_sampled = t[indices]

# Gamma marginal for P(t): mean=1, var=1/2 -> Gamma(k=2, theta=0.5)
k_shape = 2.0
theta_scale = 0.5
mean_P = 1.0

# === 1. Construct latent Gaussian process for P(t) with covariance C(τ) = 0.5 * exp(-3 τ^2) ===
tau_full = t[:, None] - t[None, :]
k_emp = 0.5
D_emp = 0.1352
C_full = k_emp * np.exp(-D_emp * tau_full**2)

# numerical jitter for Cholesky
jitter = 1e-10
C_full += jitter * np.eye(N)

L = cholesky(C_full, lower=True)

# latent Gaussian process
z_latent = L @ np.random.randn(N)

# Gaussian copula to get Gamma process P(t)
u = norm.cdf(z_latent)
P = gamma.ppf(u, a=k_shape, scale=theta_scale)

# === 2. Generate I(t) = X(t)*sqrt(P(t)) ===
X = np.random.randn(N)
I = X * np.sqrt(P)

# === 3. Sample I and form |I|^2 on [0, T_obs] at 0.1 kHz ===
I_sampled = I[indices]
power_sampled = np.abs(I_sampled)**2   # z_i = |I(t_i)|^2

# === 4. 300-sample moving average on sampled powers (0..T_obs) ===
window_size = 300
kernel = np.ones(window_size) / window_size
power_ma = np.convolve(power_sampled, kernel, mode='valid')

# time axis for centered MA
if window_size > 1:
    t_ma = t_sampled[(window_size-1)//2 : -(window_size//2)]
else:
    t_ma = t_sampled

# Simple MA "forecast": extend last MA value flat from last t_ma to T_total
t_ma_forecast = np.arange(t_ma[-1], T_total + 1e-9, sample_interval)  # 0.1 kHz grid to end
power_ma_forecast = np.full_like(t_ma_forecast, power_ma[-1])

# Approximate noise variance for |I|^2
# Var(|I|^2 | P=p) = 2 p^2, E[P^2]=1.5 => average Var ≈ 3
noise_var = 3.0

# Variance of 300-sample MA under iid noise with var=noise_var
ma_var = noise_var / window_size
ma_std = np.sqrt(ma_var)

# === 5. GPR prediction of P(t) on [0, 15] from noisy power samples on [0, 10] ===
# GP prior: P ~ GP(mean=mean_P, cov(t,s) = 0.5*exp(-3(t-s)^2))
# Observations: z_i = |I(t_i)|^2 ≈ P(t_i) + N(0, noise_var)

# training times (sampled up to T_obs)
t_train = t_sampled
z_train = power_sampled

# Covariance among training points
tau_ss = t_train[:, None] - t_train[None, :]
k_prior = 0.5
D_prior = 0.1352
C_ss = k_prior * np.exp(-D_prior * tau_ss**2 )
# Covariance between all times (0..T_total) and training points
tau_fs = t[:, None] - t_train[None, :]
C_fs = k_prior * np.exp(-D_prior * tau_fs**2 )


# Add observation noise to training covariance
C_ss_noisy = C_ss + noise_var * np.eye(len(t_train))

# Posterior mean: m_post(t) = mean_P + C_fs * K^{-1} * (z - mean_P)
z_centered = z_train - mean_P
alpha = solve(C_ss_noisy, z_centered)
P_gp_mean = mean_P + C_fs @ alpha

# Posterior variance: diag(C_ff - C_fs K^{-1} C_sf)
C_ff_diag = np.full(N, 0.5)  # since C_ff(t,t) = 0.5 for all t
V = solve(C_ss_noisy, C_fs.T)  # shape: (n_train, N)
C_fs_V = np.sum(C_fs * V.T, axis=1)
P_gp_var = C_ff_diag - C_fs_V
P_gp_std = np.sqrt(np.maximum(P_gp_var, 0.0))

# === 6. Plot ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# ---------- Subplot (a): P(t), GPR forecast, MA forecast ----------
# True P(t) over [0, 15]
ax1.plot(t, P, label='True signal power P(t)', color='C0', alpha=0.7)

# GPR mean and ±2σ over [0, 15]
ax1.plot(t, P_gp_mean, label='GPR prediction of P(t)', color='C1', linewidth=2)
# ax1.fill_between(t,
#                  P_gp_mean - 2*P_gp_std,
#                  P_gp_mean + 2*P_gp_std,
#                  color='C1', alpha=0.2, label='GPR ±2σ')

# 300-sample MA on [0, T_obs]
ax1.plot(t_ma, power_ma, 'r-', label='MA of |I|² (0–10 s)')
# ax1.fill_between(t_ma,
#                  power_ma - 2*ma_std,
#                  power_ma + 2*ma_std,
#                  color='r', alpha=0.15, label='MA ±2σ (approx, 0–10 s)')

# Flat MA forecast beyond last MA point
ax1.plot(t_ma_forecast, power_ma_forecast, 'r--', label='MA forecast (flat)')

# Raw power samples (0–10 s)
# ax1.plot(t_sampled, power_sampled, 'kx', label='Samples |I(t)|² (0.1 kHz)', markersize=6)

# Mark observation horizon
ax1.axvline(T_obs, color='k', linestyle=':', alpha=0.5)
ax1.text(T_obs+0.05, ax1.get_ylim()[1]*0.9, 'Forecast region', fontsize=12)

ax1.set_ylabel('Signal power', fontsize=12)
ax1.set_title('Fig 1(a): True P(t), GPR estimate & forecast, and 300-sample MA & forecast', fontsize=12)
ax1.grid(True)
ax1.legend(fontsize=12)

# ---------- Subplot (b): I(t) and samples ----------
ax2.plot(t, I, label='I(t) = X(t)√P(t)', color='tab:orange')
ax2.plot(t_sampled, I_sampled, 'ko', label='Samples of I(t) (0–10 s, 0.1 kHz)', markersize=1)
ax2.axvline(T_obs, color='k', linestyle=':', alpha=0.5)
ax2.set_xlabel('t [s]')
ax2.set_ylabel('Amplitude', fontsize=12)
ax2.set_title('Fig 1(b): Non-stationary Gaussian noise I(t) with sampled observations', fontsize=12)
ax2.grid(True)
ax2.legend(fontsize=12)

plt.tight_layout()
plt.show()
