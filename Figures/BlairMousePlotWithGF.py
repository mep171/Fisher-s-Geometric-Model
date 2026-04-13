# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 13:30:04 2026

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
from sklearn.metrics import mean_absolute_error, r2_score
import scipy.stats as stats
# --- Configuration ---
MAX_REGRESS_ITER = 50000 
NUM_SEEDS = 500
BASE_SEED = 980
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

    def loss_function(self, parameter_vector, observed_fitness, norm, l2_lambda):
        Z, P, X = self.reconstruct_ZP(parameter_vector)
        predicted_fitness = self.landscape.calculate_fitness(Z, P, X)
        weighted_residuals = (observed_fitness - predicted_fitness) / norm
        data_loss = jnp.mean(jnp.square(weighted_residuals))
        reg_loss = l2_lambda * (jnp.sum(jnp.square(Z)) + jnp.sum(jnp.square(P)))
        return data_loss + reg_loss

def align_all(ps_runs, zs_runs, max_iter=10):
    mean_P = np.mean(ps_runs, axis=0)
    for _ in range(max_iter):
        aligned_ps, aligned_zs = [], []
        for i in range(len(ps_runs)):
            P, Z = ps_runs[i], zs_runs[i]
            P_mean = np.mean(P, axis=0)
            P_centered = P - P_mean
            scale = np.linalg.norm(P_centered)
            P_scaled = P_centered / scale
            mean_centered = mean_P - np.mean(mean_P, axis=0)
            mean_scaled = mean_centered / np.linalg.norm(mean_centered)
            R, _ = orthogonal_procrustes(P_scaled, mean_scaled)
            aligned_ps.append(P_scaled @ R)
            aligned_zs.append(((Z + P_mean) / scale) @ R)
        aligned_ps, aligned_zs = np.array(aligned_ps), np.array(aligned_zs)
        new_mean = np.mean(aligned_ps, axis=0)
        if np.linalg.norm(new_mean - mean_P) < 1e-6: break
        mean_P = new_mean
    return aligned_ps, aligned_zs

def main():
    # --- Setup & Data Loading ---
    ls_obj = Landscape(C=3, D=TARGET_D, M=28, CONSTRAIN_ROTATION=False)
    key = random.PRNGKey(BASE_SEED)

    try:
        MiceG12C = pd.read_csv(r"C:\Users\Meaghan Parks\Documents\McFarlandLabProjects\Figure5A.csv")
        MiceG12D = pd.read_csv(r"C:\Users\Meaghan Parks\Documents\McFarlandLabProjects\Figure5B.csv")
        MiceEGFR = pd.read_csv(r"C:\Users\Meaghan Parks\Documents\McFarlandLabProjects\Figure5D.csv")
    except FileNotFoundError:
        print("Error: CSV files not found.")
        return

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

    results_list, Z_runs, P_runs, Pred_fits = [], [], [], []
    
    # --- Optimization Loop ---
    for i in range(NUM_SEEDS):
        key, Z_init, P_init, X_init = ls_obj.generate_initial_guess(key)
        init_pv = prob_obj.get_parameter_vector(Z_init, P_init, X_init)
        try:
            res = solver.run(init_pv, observed, norm, L2_LAMBDA)
            fZ, fP, fX = prob_obj.reconstruct_ZP(res.params)
            pred_fit = ls_obj.calculate_fitness(fZ, fP, fX)
            
            Z_runs.append(np.array(fZ))
            P_runs.append(np.array(fP))
            Pred_fits.append(np.array(pred_fit))
            results_list.append({"seed": i, "loss": float(res.state.fun_val)})
        except Exception as e:
            continue

    # --- Identify Best Seed for Plotting ---
    losses = np.array([r['loss'] for r in results_list])
    best_idx = np.argmin(losses)
    best_seed = results_list[best_idx]['seed']
    print(f"Best Seed found: {best_seed}")
    # Map your requested variables to the best result
    mice_fitness = jnp.ravel(observed)
    MicePredFitBest = jnp.ravel(Pred_fits[best_idx])
    uncertainty = norm.flatten()

    # --- Calculations ---
    r2 = r2_score(mice_fitness, MicePredFitBest)
    mae = mean_absolute_error(mice_fitness, MicePredFitBest)
    pcc, p_val = stats.pearsonr(mice_fitness, MicePredFitBest)

    print(f"Best Seed Stats: R2={r2:.4f}, MAE={mae:.4f}, PCC={pcc:.4f}")

    # --- Plotting: Regressed vs Measured ---
    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 12})
    
    plt.scatter(mice_fitness, MicePredFitBest, c=uncertainty, cmap="summer", marker="o", alpha=.5)
    
    plt.title("Regressed vs. Measured Fitness", fontsize=16)
    plt.ylabel("Log Regressed Fitness", fontsize=14)
    plt.xlabel("Log Observed Fitness", fontsize=14)
    
    # Dynamic text placement based on metrics
    plt.text(np.min(mice_fitness), np.max(MicePredFitBest)-0.3, f"MAE= {mae:.4f}", fontsize=12)
    plt.text(np.min(mice_fitness), np.max(MicePredFitBest)-0.6, f"PCC = {pcc:.4f}, p < .00001", fontsize=12)
    
    plt.colorbar(label='Experimental Uncertainty')
    
    # Identity Line
    lims = [np.min([mice_fitness, MicePredFitBest]), np.max([mice_fitness, MicePredFitBest])]
    plt.plot(lims, lims, color="black", linestyle="--")
    
    plt.tight_layout()
    plt.savefig("Reg_vs_Real_Mouse.pdf", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()