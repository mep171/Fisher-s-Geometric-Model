# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 14:25:45 2026

@author: Meaghan Parks
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 14:25:45 2026
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
    Z_shifted = Z + P_mean 

    # 2. Rotation: Align Anchor Mutant to the +X axis
    anchor = P_centered[anchor_idx]
    angle = np.arctan2(anchor[1], anchor[0])
    
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    P_fixed = P_centered @ R
    Z_fixed = Z_shifted @ R

    # 3. Reflection: Ensure Y-axis orientation is consistent
    if second_anchor_idx is None:
        second_anchor_idx = np.argmax(np.abs(P_fixed[:, 1]))
    
    if P_fixed[second_anchor_idx, 1] < 0:
        P_fixed[:, 1] = -P_fixed[:, 1]
        Z_fixed[:, 1] = -Z_fixed[:, 1]

    return Z_fixed, P_fixed

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

def main():
    # --- Setup & Data Loading ---
    ls_obj = Landscape(C=3, D=TARGET_D, M=28, CONSTRAIN_ROTATION=False)
    key = random.PRNGKey(BASE_SEED)

    # Note: Ensure these paths are correct for your local machine
    try:
        MiceG12C = pd.read_csv(r"C:\Users\Meaghan Parks\Documents\McFarlandLabProjects\Figure5A.csv")
        MiceG12D = pd.read_csv(r"C:\Users\Meaghan Parks\Documents\McFarlandLabProjects\Figure5B.csv")
        MiceEGFR = pd.read_csv(r"C:\Users\Meaghan Parks\Documents\McFarlandLabProjects\Figure5D.csv")
    except FileNotFoundError:
        print("Error: CSV files not found. Please check your file paths.")
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

    results_list, P_best_runs, Z_best_runs = [], [], []
    best_loss = float('inf')
    fP_best = None
    fZ_best = None
    fX_best = None
    
    # --- Optimization Loop ---
    print(f"Starting optimization for {NUM_SEEDS} seeds...")
    for i in range(NUM_SEEDS):
        key, Z_init, P_init, X_init = ls_obj.generate_initial_guess(key)
        init_pv = prob_obj.get_parameter_vector(Z_init, P_init, X_init)
        try:
            res = solver.run(init_pv, observed, norm, L2_LAMBDA)
            current_loss = float(res.state.fun_val)
            
            # Reconstruct parameters
            fZ, fP, fX = prob_obj.reconstruct_ZP(res.params)
            
            if current_loss < best_loss:
                best_loss = current_loss
                fP_best, fZ_best, fX_best = fP, fZ, fX
                
            results_list.append({"seed": i, "loss": current_loss})
        except Exception as e:
            continue

    print(f"Optimization complete. Best Loss: {best_loss:.4f}")

    # --- Analysis of Dimension Differences ---
    # We use fP_best which is the mutant position matrix (M x 2)
    
    # 1. Absolute magnitude of effects per mutant for each dimension
    dim1_abs = np.abs(fP_best[:, 0])
    dim2_abs = np.abs(fP_best[:, 1])

    # 2. Absolute difference between the effect in Dimension 1 and Dimension 2
    diff_between_dims = np.abs(dim1_abs - dim2_abs)
    # Create a Summary DataFrame
    summary_df = pd.DataFrame({
        'Gene': AllMice['gene'],
        'Dim1_Effect': dim1_abs,
        'Dim2_Effect': dim2_abs,
        'Abs_Difference': diff_between_dims
    })

    print("\n--- Summary of Mutational Effects ---")
    print(summary_df)
    print(f"\nAverage Absolute Difference across all mutations: {np.mean(diff_between_dims):.4f}")
    
    # Statistics for Best Fit
    MicePredFitBest = ls_obj.calculate_fitness(fZ_best, fP_best, fX_best)
    r2 = r2_score(np.ravel(observed), np.ravel(MicePredFitBest))
    print(f"Model R^2: {r2:.4f}")

    # --- Optional Visualization ---
    plt.figure(figsize=(10, 5))
    plt.bar(summary_df['Gene'][:15], summary_df['Abs_Difference'][:15])
    plt.xticks(rotation=45)
    plt.ylabel('Abs Difference (Dim 1 vs Dim 2)')
    plt.title('Top 15 Mutations: Difference in Effect Between Dimension 1 and 2')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()