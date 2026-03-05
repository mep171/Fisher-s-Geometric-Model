# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 12:24:34 2026

@author: Meaghan Parks
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
import numpy as np
from jax import random
import jaxopt 
import pandas as pd

# --- Configuration ---
MAX_REGRESS_ITER = 50000
seed = 15
key = random.PRNGKey(seed)

# Plotting Settings
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

# --- Classes ---

class landscape:
    def __init__(self, C=3, D=1, M=28, scale=1, CONSTRAIN_ROTATION=True):
        self.C = C
        self.D = D
        self.M = M
        self.scale = scale
        self.CONSTRAIN_ROTATION = CONSTRAIN_ROTATION

    def simulate_dataset(self, key, noise=0):
        key, key_z = random.split(key)
        Z = random.normal(key_z, (self.C, self.D))
        key, key_p = random.split(key)
        P = random.normal(key_p, (self.M, self.D))
        
        if self.CONSTRAIN_ROTATION:
            P = jnp.tril(P, -1)
            P = P.at[jnp.triu_indices_from(P,-1)].set(jnp.abs(P[jnp.triu_indices_from(P,-1)]))
        return key, Z, P
    
    def calculate_fitness(self, Z, P, X):
        tiledZ = jnp.tile(Z, (self.M, 1))
        repedP = jnp.repeat(P, self.C, axis=0)
        repMutant = tiledZ + repedP
        # Vectorized fitness calculation
        Fitness = X * (jnp.exp(-jnp.einsum('cd,cd->c', repMutant, repMutant) / 2))
        return jnp.log(Fitness)

class RegressionProblem:
    def __init__(self, landscape_obj, C=3, D=1, M=28, CONSTRAIN_ROTATION=True):
        self.landscape = landscape_obj
        self.C = C
        self.D = D
        self.M = M
        self.CONSTRAIN_ROTATION = CONSTRAIN_ROTATION
    
    def get_parameter_vector(self, Z, P, X):
        if self.CONSTRAIN_ROTATION:
            P_flat = P[jnp.tril_indices_from(P, -1)]
        else:
            P_flat = jnp.ravel(P)
        Z_flat = jnp.ravel(Z)
        X_val = jnp.atleast_1d(X)
        return jnp.concatenate([X_val, Z_flat, P_flat])

    def reconstruct_ZP(self, parameter_vector, D):
        X = parameter_vector[0]
        Z = parameter_vector[1 : self.C*self.D + 1].reshape((self.C, self.D))
        
        if self.CONSTRAIN_ROTATION:
            P = jnp.zeros((self.M, self.D))
            tril_indices = jnp.tril_indices_from(P, -1)
            P = P.at[tril_indices].set(parameter_vector[self.C*self.D + 1:])
            # Enforce rotation constraint
            P = P.at[jnp.triu_indices_from(P, -1)].set(jnp.abs(P[jnp.triu_indices_from(P, -1)]))
        else:
            P = parameter_vector[-(self.M * self.D):].reshape((self.M, self.D))
        return Z, P, X

    def loss_function(self, parameter_vector, observed_fitness, norm, mask=None):
        Z, P, X = self.reconstruct_ZP(parameter_vector, self.D)
        predicted_fitness = self.landscape.calculate_fitness(Z, P, X)
        
        obs_flat = observed_fitness.ravel()
        pred_flat = predicted_fitness.ravel()
        norm_flat = norm.ravel()
        
        # LOGARITHMIC WEIGHTING: Scale residuals by the log-error (delta_z)
        # This ensures error bars are symmetric in log-space per the Stuve PDF.
        weighted_residuals = (obs_flat - pred_flat) / norm_flat
        
        loss = jaxopt.loss.huber_loss(weighted_residuals, jnp.zeros_like(weighted_residuals))
        
        if mask is not None:
            loss = loss.at[mask].set(0.0)
            
        return loss.sum()

# --- Functions ---

def regress_LBFGS(regression_obj, fitness_data, norm_data, Z, P, X, masked_indices=None):
    parameter_vector = regression_obj.get_parameter_vector(Z, P, X)
    solver = jaxopt.LBFGS(fun=regression_obj.loss_function, maxiter=MAX_REGRESS_ITER)
    res = solver.run(
        parameter_vector, 
        observed_fitness=fitness_data, 
        norm=norm_data,
        mask=masked_indices
    )
    return res.params

def run_k_fold_cv_full(regression_obj, fitness_data, norm_data, Z_init, P_init, X_init, SplitSeed=42, k=5):
    obs_flat = fitness_data.ravel()
    norm_flat = norm_data.ravel()
    n_samples = obs_flat.size
    
    indices = np.arange(n_samples)
    np.random.seed(SplitSeed)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    
    results = {"fold_errors": [], "parameters": [], "all_predictions": []}
    
    for i in range(k):
        test_indices = folds[i]
        
        final_params = regress_LBFGS(
            regression_obj, obs_flat, norm_flat, 
            Z_init, P_init, X_init, 
            masked_indices=test_indices
        )
        
        Z_f, P_f, X_f = regression_obj.reconstruct_ZP(final_params, regression_obj.D)
        full_pred = regression_obj.landscape.calculate_fitness(Z_f, P_f, X_f)
        
        # Weighted Test MSE (using logarithmic norm)
        actual_test = obs_flat[test_indices]
        pred_test = full_pred.ravel()[test_indices]
        sigma_test = norm_flat[test_indices]
        
        weighted_mse = jnp.mean(((actual_test - pred_test) / sigma_test)**2)
        
        results["fold_errors"].append(weighted_mse)
        results["parameters"].append({"Z": Z_f, "P": P_f, "X": X_f})
        results["all_predictions"].append(full_pred)
        
        print(f"Fold {i+1} completed. Weighted Test MSE: {weighted_mse:.4f}")

    return results

# --- Main Execution ---

# 1. Load and Merge Data
MiceG12C = pd.read_csv(r"C:\Users\Meaghan Parks\Documents\McFarlandLabProjects\Figure5A.csv")
MiceG12D = pd.read_csv(r"C:\Users\Meaghan Parks\Documents\McFarlandLabProjects\Figure5B.csv")
MiceEGFR = pd.read_csv(r"C:\Users\Meaghan Parks\Documents\McFarlandLabProjects\Figure5D.csv")

mergeMiceG12 = pd.merge(MiceG12C, MiceG12D, on='gene', how='inner')
AllMice = pd.merge(mergeMiceG12, MiceEGFR, on="gene", how='inner')

# 2. Prepare Log-Fitness
AllMiceVals = AllMice.loc[:, ['tumor_enrichment_x', 'tumor_enrichment_y', 'tumor_enrichment']]
mice_fitness = jnp.log(jnp.transpose(AllMiceVals.to_numpy()))

# 3. Calculate CORRECT Logarithmic Norms (delta_z = 0.434 * delta_y / y)
# Per the Eric M. Stuve document, this makes error bars symmetric on log plots.
dy_x = (AllMice['CI_upper_x'] - AllMice['CI_lower_x']) / 2
dy_y = (AllMice['CI_upper_y'] - AllMice['CI_lower_y']) / 2
dy_egfr = (AllMice['CI_upper'] - AllMice['CI_lower']) / 2

rel_error_x =(dy_x / AllMice['tumor_enrichment_x'])
rel_error_y = (dy_y / AllMice['tumor_enrichment_y'])
rel_error_egfr =(dy_egfr / AllMice['tumor_enrichment'])

ALLMiceNormNP = jnp.stack([rel_error_x.values, rel_error_y.values, rel_error_egfr.values])

# 4. Initialize Landscape and Problem
landscape_obj = landscape()
regression_obj = RegressionProblem(landscape_obj)

# 5. Initialize Parameters
key, Z_start, P_start = landscape_obj.simulate_dataset(key)
X_start = jnp.array([1.0])



import seaborn as sns

def run_loocv_residuals(regression_obj, fitness_data, norm_data, Z_init, P_init, X_init):
    obs_flat = fitness_data.ravel()
    norm_flat = norm_data.ravel()
    n_samples = obs_flat.size
    
    # Storage for residuals (observed - predicted)
    loocv_residuals = np.zeros(n_samples)
    
    print(f"Starting LOOCV for {n_samples} data points...")
    
    for i in range(n_samples):
        # The mask is just the current index
        test_mask = jnp.array([i])
        
        # Train on everything EXCEPT index i
        final_params = regress_LBFGS(
            regression_obj, fitness_data, norm_data, 
            Z_init, P_init, X_init, 
            masked_indices=test_mask
        )
        
        # Predict the value for the omitted point
        Z_f, P_f, X_f = regression_obj.reconstruct_ZP(final_params, regression_obj.D)
        full_pred = regression_obj.landscape.calculate_fitness(Z_f, P_f, X_f).ravel()
        
        # Calculate residual: (Observed - Predicted) / Sigma
        # We divide by sigma to get 'studentized' residuals (standardized)
        residual = (obs_flat[i] - full_pred[i]) / norm_flat[i]
        loocv_residuals[i] = residual
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{n_samples} points...")

    # Reshape back to the original matrix dimensions (3, 28)
    return loocv_residuals.reshape(fitness_data.shape)

# --- Execute LOOCV ---
# Note: This may take a few minutes depending on MAX_REGRESS_ITER
residuals_matrix = run_loocv_residuals(
    regression_obj, mice_fitness, ALLMiceNormNP, Z_start, P_start, X_start
)

# --- Plotting the Heatmap ---
plt.figure(figsize=(14, 6))
conditions = ['G12C', 'G12D', 'EGFR']
gene_labels = AllMice['gene'].values

sns.heatmap(
    residuals_matrix, 
    annot=False, 
    fmt=".2f", 
    cmap="RdBu_r", 
    center=0,
    xticklabels=gene_labels, 
    yticklabels=conditions,
    cbar_kws={'label': 'Standardized Residual (Log Space)'}
)

plt.title("LOOCV Residuals Heatmap (Leave-One-Out)")
plt.xlabel("Mutant (Gene)")
plt.ylabel("Mouse Model")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
#plt.savefig(r"MouseLOOCI.pdf", dpi=300)
plt.show()