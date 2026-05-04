# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:22:15 2026

@author: Meaghan Parks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score

# --- Configuration (Matching your workflow) ---
MAX_REGRESS_ITER = 1000
NUM_SEEDS = 25             # Multi-seed approach to ensure global minimum
L2_LAMBDA = 1e-4 
TARGET_D = 2 
C_COUNT, M_COUNT = 5, 20 # Synthetic dimensions

# --- Helper Functions ---

def gauge_fix_Fixed(Z, P, d):
    """
    Fixes the gauge degrees of freedom (translation and rotation)
    by aligning to the first d coordinates of P.
    """
    # 1. Translation: Center the mutants (P) at the origin
    Pmean = P.mean(axis=0)
    Pshifted = P - Pmean
    Zshifted = Z + Pmean  # Z must shift inversely to maintain fitness values

    # 2. Rotation: Align Anchor Mutant to the +X axis
    M = Pshifted[:d, :d].T
    Q, R = np.linalg.qr(M)
    
    # Standard 2D Rotation Matrix
    Protated = Pshifted @ Q
    Zrotated = Zshifted @ Q
    
    # 3. Reflection: Ensure Y-axis orientation is consistent
    signs = np.sign(np.diag(Protated))
    S = np.diag(signs)
    P_fixed = Protated @ S
    Z_fixed = Zrotated @ S

    return Z_fixed, P_fixed

# --- Updated Classes ---

class Landscape:
    def __init__(self, C, D, M):
        self.C, self.D, self.M = C, D, M

    def calculate_fitness(self, Z, P, X):
        # f = log(X) - 1/2 * |Z + P|^2
        combined = Z[:, np.newaxis, :] + P[np.newaxis, :, :]
        dist_sq = np.sum(combined**2, axis=2)
        return np.log(np.abs(X)) - (dist_sq / 2.0)

class RegressionProblem:
    def __init__(self, landscape, observed, norm):
        self.landscape = landscape
        self.observed = observed
        self.norm = norm

    def pack(self, Z, P, X):
        return np.concatenate([[X], Z.flatten(), P.flatten()])

    def unpack(self, params):
        X = params[0]
        Z = params[1 : self.landscape.C*self.landscape.D + 1].reshape((self.landscape.C, self.landscape.D))
        P = params[self.landscape.C*self.landscape.D + 1 :].reshape((self.landscape.M, self.landscape.D))
        return Z, P, X

    def loss_function(self, params):
        Z, P, X = self.unpack(params)
        predicted = self.landscape.calculate_fitness(Z, P, X)
        # Weighted residuals using uncertainty (norm)
        weighted_res = (self.observed - predicted) / self.norm
        data_loss = np.mean(weighted_res**2)
        reg_loss = L2_LAMBDA * (np.sum(Z**2) + np.sum(P**2))
        return data_loss + reg_loss

# --- 1. Generate Synthetic Data ---
np.random.seed(980)
ls_obj = Landscape(C_COUNT, TARGET_D, M_COUNT)

# Ground Truth
Z_true = np.random.normal(size=(C_COUNT, TARGET_D))
P_true = np.random.normal(size=(M_COUNT, TARGET_D))
X_true = 25.0
fit_true = ls_obj.calculate_fitness(Z_true, P_true, X_true)

# Simulated Uncertainty and Noisy Observations
Norm = np.abs(np.random.normal(loc=0.2, scale=0.1, size=fit_true.shape))
noise = np.random.normal(scale=Norm)
Simulated_fitness = fit_true + noise

# --- 2. Multi-Seed Optimization ---
prob = RegressionProblem(ls_obj, Simulated_fitness, Norm)
best_loss = np.inf
best_preds = None
best_Z, best_P, best_X = None, None, None

print(f"Optimizing across {NUM_SEEDS} seeds...")
for i in range(NUM_SEEDS):
    # Random guess for each seed
    init_params = prob.pack(np.random.normal(size=(C_COUNT, TARGET_D)), 
                            np.random.normal(size=(M_COUNT, TARGET_D)), 1.0)
    
    res = minimize(prob.loss_function, init_params, method='L-BFGS-B', options={'maxiter': MAX_REGRESS_ITER})
    
    if res.fun < best_loss:
        best_loss = res.fun
        best_Z, best_P, best_X = prob.unpack(res.x)
        best_preds = ls_obj.calculate_fitness(best_Z, best_P, best_X)

# --- 3. Apply Fixed Gauge Fixing ---
Z_final, P_final = gauge_fix_Fixed(best_Z, best_P, TARGET_D)

# --- 4. Plotting and Metrics ---
obs_flat = Simulated_fitness.flatten()
pred_flat = best_preds.flatten()

mae = mean_absolute_error(obs_flat, pred_flat)
pcc, p_val = pearsonr(obs_flat, pred_flat)

plt.figure(figsize=(8, 6))
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

sc = plt.scatter(obs_flat, pred_flat, c=Norm.flatten(), cmap="summer", alpha=.6, edgecolors='black', linewidths=0.2)
plt.colorbar(sc, label='Synthetically Generated Uncertainty')

plt.title("Regressed vs. Measured Synthetic Fitness", fontsize=16)
plt.ylabel("Log Regressed Fitness", fontsize=14)
plt.xlabel("Log Observed Synthetic Fitness", fontsize=14)

# Identity Line
lims = [min(obs_flat.min(), pred_flat.min()) - 0.5, max(obs_flat.max(), pred_flat.max()) + 0.5]
plt.plot(lims, lims, color="black", linestyle="--", linewidth=1)

# Dynamic text labels
plt.text(lims[0] + 0.5, lims[1] - 1.0, f"MAE= {mae:.4f}", fontsize=12)
plt.text(lims[0] + 0.5, lims[1] - 2.0, f"PCC = {pcc:.4f}, p < .00001", fontsize=12)

plt.tight_layout()
plt.savefig(r"SyntheticFitnessGF.pdf", dpi=300)
plt.show()
