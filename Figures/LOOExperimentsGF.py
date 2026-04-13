# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 16:49:37 2026

@author: Meaghan Parks
"""


import jax
import jax.numpy as jnp
from jax import random
import jaxopt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.direction': 'in',
    'ytick.direction': 'in'
})

# --- Configuration ---
MAX_REGRESS_ITER = 50000 
NUM_SEEDS = 500
BASE_SEED = 404
TARGET_D = 2  
L2_LAMBDA = 1e-4 

# --- Helper Functions ---

def calculate_r_squared(observed, predicted, weights=None):
    """
    Calculates the Coefficient of Determination (R^2).
    If weights (1/norm) are provided, calculates Weighted R^2.
    """
    obs_flat = jnp.ravel(observed)
    pred_flat = jnp.ravel(predicted)
    
    if weights is not None:
        w = jnp.ravel(weights)
        weighted_mean = jnp.sum(w * obs_flat) / jnp.sum(w)
        ss_tot = jnp.sum(w * jnp.square(obs_flat - weighted_mean))
        ss_res = jnp.sum(w * jnp.square(obs_flat - pred_flat))
    else:
        mean_obs = jnp.mean(obs_flat)
        ss_tot = jnp.sum(jnp.square(obs_flat - mean_obs))
        ss_res = jnp.sum(jnp.square(obs_flat - pred_flat))
    
    return 1.0 - (ss_res / ss_tot)

def gauge_fix_posthoc(Z, P, anchor_idx, second_anchor_idx=None):
    # 1. Translation: Center the mutants (P) at the origin
    P_mean = np.mean(P, axis=0)
    P_centered = P - P_mean
    Z_shifted = Z + P_mean  # Z must shift inversely to maintain fitness values

    # 2. Rotation: Align Anchor Mutant to the +X axis
    anchor = P_centered[anchor_idx]
    angle = np.arctan2(anchor[1], anchor[0])
    
    # Standard 2D Rotation Matrix
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    P_fixed = P_centered @ R
    Z_fixed = Z_shifted @ R

    # 3. Reflection: Ensure Y-axis orientation is consistent
    if second_anchor_idx is None:
        # Default to the point with the largest Y-magnitude if not specified
        second_anchor_idx = np.argmax(np.abs(P_fixed[:, 1]))
    
    if P_fixed[second_anchor_idx, 1] < 0:
        P_fixed[:, 1] = -P_fixed[:, 1]
        Z_fixed[:, 1] = -Z_fixed[:, 1]

    return Z_fixed, P_fixed
def calculate_intermutant_distances(P):
    """Returns an (M, M) matrix of pairwise Euclidean distances."""
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))
# --- Classes ---

class Landscape:
    def __init__(self, C=3, D=2, M=28, CONSTRAIN_ROTATION=True):
        self.C = C
        self.D = D
        self.M = M
        self.CONSTRAIN_ROTATION = CONSTRAIN_ROTATION

    def generate_initial_guess(self, key):
        key, kZ, kP, kX = random.split(key, 4)
        Z = random.normal(kZ, (self.C, self.D))
        P = random.normal(kP, (self.M, self.D))
        X = 1.0  
        return key, Z, P, X

    def calculate_fitness(self, Z, P, X):
        combined_phenotype = Z[:, jnp.newaxis, :] + P[jnp.newaxis, :, :]
        dist_sq = jnp.sum(jnp.square(combined_phenotype), axis=2)
        return jnp.log(jnp.abs(X)) - (dist_sq / 2.0)

class RegressionProblem:
    def __init__(self, landscape_obj, observed_fitnesses, norm):
        self.landscape = landscape_obj
        self.observed_fitnesses = observed_fitnesses
        self.norm = norm
        self.C = landscape_obj.C
        self.D = landscape_obj.D
        self.M = landscape_obj.M
        self.CONSTRAIN_ROTATION = landscape_obj.CONSTRAIN_ROTATION

    def get_parameter_vector(self, Z, P, X):
        Z_flat = jnp.ravel(Z)
        X_val = jnp.array([X])
        if self.CONSTRAIN_ROTATION:
            indices = jnp.tril_indices(self.M, k=0, m=self.D)
            P_flat = P[indices]
        else:
            P_flat = jnp.ravel(P)
        return jnp.concatenate([X_val, Z_flat, P_flat])

    def reconstruct_ZP(self, parameter_vector):
        X = parameter_vector[0]
        Z = parameter_vector[1 : self.C * self.D + 1].reshape((self.C, self.D))
        remaining = parameter_vector[self.C * self.D + 1:]
        
        P = jnp.zeros((self.M, self.D))
        if self.CONSTRAIN_ROTATION:
            indices = jnp.tril_indices(self.M, k=0, m=self.D)
            P = P.at[indices].set(remaining)
        else:
            P = remaining.reshape((self.M, self.D))
        return Z, P, X

    # 1. Update signature to match solver.run() arguments
    def loss_function(self, params, observed, norm, l2_lambda, mask):
        Z, P, X = self.reconstruct_ZP(params)
        predicted = self.landscape.calculate_fitness(Z, P, X)
        
        # 2. Fix naming: use the 'observed' passed into the function
        # 3. Manual zeroing logic (at[2,0] is EGFR mutant 0)
        target = observed
        # 4. Use jnp (JAX NumPy) for all math operations
        weighted_res = (target - predicted) / norm
        
        # 5. Apply the mask to ensure we are actually leaving the point out
        # instead of just forcing the model to fit a zero.
        sq_residuals = jnp.square(weighted_res)
        sq_residuals = sq_residuals
        data_loss = jnp.sum(sq_residuals * mask) / jnp.sum(mask)
        
        reg_loss = l2_lambda * (jnp.sum(jnp.square(Z)) + jnp.sum(jnp.square(P)))
        return data_loss+ reg_loss
         



ls_obj = Landscape(C=3, D=TARGET_D, M=28, CONSTRAIN_ROTATION=False)
key = random.PRNGKey(BASE_SEED)

    # --- Data Loading ---
MiceG12C = pd.read_csv(r"C:\Users\Meaghan Parks\Documents\McFarlandLabProjects\Figure5A.csv")
MiceG12D = pd.read_csv(r"C:\Users\Meaghan Parks\Documents\McFarlandLabProjects\Figure5B.csv")
MiceEGFR = pd.read_csv(r"C:\Users\Meaghan Parks\Documents\McFarlandLabProjects\Figure5D.csv")


merged = pd.merge(MiceG12C, MiceG12D, on='gene', how='inner')
AllMice = pd.merge(merged, MiceEGFR, on="gene", how='inner')

obs_vals = AllMice[['tumor_enrichment_x', 'tumor_enrichment_y', 'tumor_enrichment']].values
observed = jnp.log(jnp.transpose(obs_vals))

dy_x = (AllMice['CI_upper_x'] - AllMice['CI_lower_x']) / 2
dy_y = (AllMice['CI_upper_y'] - AllMice['CI_lower_y']) / 2
dy_egfr = (AllMice['CI_upper'] - AllMice['CI_lower']) / 2
norm = jnp.stack([(dy_x/AllMice['tumor_enrichment_x']).values, 
                  (dy_y/AllMice['tumor_enrichment_y']).values, 
                  (dy_egfr/AllMice['tumor_enrichment']).values])

prob_obj = RegressionProblem(ls_obj, observed, norm)
solver = jaxopt.ScipyMinimize(method="L-BFGS-B", fun=prob_obj.loss_function, maxiter=MAX_REGRESS_ITER)
results_list, Z_runs, P_runs = [], [], []

# Flatten observed fitness and norm for easier indexing
flat_obs = observed.ravel()
flat_norm = norm.ravel()
num_points = len(flat_obs)

print("Running global fit for warm-start...")
full_mask = jnp.ones_like(observed)
key, Z_init, P_init, X_init = ls_obj.generate_initial_guess(key)
init_pv = prob_obj.get_parameter_vector(Z_init, P_init, X_init)
global_res = solver.run(init_pv, observed, norm, L2_LAMBDA, full_mask)
best_params = global_res.params # Use this as the starting point for all LOO folds


Pred_Z, Pred_P, Pred_X=prob_obj.reconstruct_ZP(best_params)
pred_fit=ls_obj.calculate_fitness(Pred_Z, Pred_P, Pred_X)

resis=observed-pred_fit

global_mae = jnp.mean(jnp.abs(resis))

# 2. Per-condition MAE (G12C, G12D, EGFR)
# Axis 1 averages across the mutants for each row (condition)
condition_maes = jnp.mean(jnp.abs(resis), axis=1)
condition_names = ['G12C', 'G12D', 'EGFR']
gene_names = AllMice['gene'].values

print("-" * 30)
print(f"MAE Analysis Results:")
print(f"Global MAE: {global_mae:.4f}")
print("-" * 30)

# condition_names should be defined as ['G12C', 'G12D', 'EGFR']
for name, mae in zip(condition_names, condition_maes):
    print(f"{name} Model MAE: {mae:.4f}")
print("-" * 30)

import seaborn as sns
from scipy.stats import pearsonr

# --- 1. Define Metadata ---
# The order here must match the order in 'observed' (G12C, G12D, EGFR)

# --- 2. Plot and Save the Residual Heatmap ---
plt.figure(figsize=(15, 6))
res_np = np.array(resis) # Global residuals from your script

sns.heatmap(res_np, annot=False, cmap='RdBu_r', center=0,
            xticklabels=gene_names, yticklabels=condition_names,
            cbar_kws={'label': 'Residual (Log Fitness Error)'})

plt.title('LOOCV Residual Heatmap: Observed - Predicted')
plt.xlabel('Mutants (Genes)')
plt.ylabel('Conditions (Mice Models)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('LOOCV_Residual_Heatmap.pdf', dpi=300)
plt.show()

# --- 3. 2-Fold Cross-Validation Logic ---
def run_2fold_cv(prob_obj, solver, observed, norm, l2_lambda, initial_params):
    n_conditions, n_mutants = observed.shape
    mid = n_mutants // 2
    
    # Fold 1: Train on first half (0 to mid), Predict second half (mid to end)
    mask1 = jnp.zeros_like(observed).at[:, :mid].set(1.0)
    res_fold1 = solver.run(initial_params, observed, norm, l2_lambda, mask1)
    Z1, P1, X1 = prob_obj.reconstruct_ZP(res_fold1.params)
    pred_from_fold1 = ls_obj.calculate_fitness(Z1, P1, X1)
    
    # Fold 2: Train on second half (mid to end), Predict first half (0 to mid)
    mask2 = jnp.zeros_like(observed).at[:, mid:].set(1.0)
    res_fold2 = solver.run(initial_params, observed, norm, l2_lambda, mask2)
    Z2, P2, X2 = prob_obj.reconstruct_ZP(res_fold2.params)
    pred_from_fold2 = ls_obj.calculate_fitness(Z2, P2, X2)
    
    return pred_from_fold1, pred_from_fold2, mid

# Run CV using the global 'best_params' as a warm start
pred_f1, pred_f2, mid_idx = run_2fold_cv(prob_obj, solver, observed, norm, L2_LAMBDA, best_params)

# --- 4. Plot and Save CV Scatter Plots ---
for i, name in enumerate(condition_names):
    plt.figure(figsize=(6, 6))
    
    # Extract training and test sets for this specific condition
    obs_test_h1, pred_test_h1 = observed[i, mid_idx:], pred_f1[i, mid_idx:]
    obs_test_h2, pred_test_h2 = observed[i, :mid_idx], pred_f2[i, :mid_idx]
    
    plt.scatter(obs_test_h1, pred_test_h1, label='First Half Test', marker='s', color='#1f77b4', alpha=0.7)
    plt.scatter(obs_test_h2, pred_test_h2, label='Second Half Test', marker='^', color='#ff7f0e', alpha=0.7)
    
    # Statistics
    all_obs = np.concatenate([obs_test_h1, obs_test_h2])
    all_pred = np.concatenate([pred_test_h1, pred_test_h2])
    r, p_val = pearsonr(all_obs, all_pred)
    
    # Plot Identity Line
    mn, mx = min(all_obs.min(), all_pred.min()), max(all_obs.max(), all_pred.max())
    plt.plot([mn, mx], [mn, mx], 'k--', alpha=0.5, label='Perfect Prediction')
    
    plt.title(f'2-Fold Cross-Validation: {name} Mutants')
    plt.xlabel('Measured Log Fitness')
    plt.ylabel('Predicted Log Fitness')
    plt.legend()
    
    # Format p-value and PCC text
    p_text = f"p < 0.0001" if p_val < 0.0001 else f"p = {p_val:.4f}"
    plt.text(0.035, 0.7, f'Combined Pearson r: {r:.4f}\n{p_text}', 
             transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{name}2FoldCrossVal.pdf', dpi=300)
    plt.show()