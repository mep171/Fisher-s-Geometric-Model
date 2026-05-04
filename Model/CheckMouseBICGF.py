# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 14:22:09 2026

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

# --- Configuration ---
MAX_REGRESS_ITER = 50000 
NUM_SEEDS = 200 # Reduced for faster comparison; increase for production
BASE_SEED = 980
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

# --- Classes ---

class Landscape:
    def __init__(self, C=3, D=2, M=28):
        self.C = C
        self.D = D
        self.M = M

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

    def get_parameter_vector(self, Z, P, X):
        return jnp.concatenate([jnp.array([X]), jnp.ravel(Z), jnp.ravel(P)])

    def reconstruct_ZP(self, parameter_vector):
        X = parameter_vector[0]
        Z = parameter_vector[1 : self.C * self.D + 1].reshape((self.C, self.D))
        P = parameter_vector[self.C * self.D + 1:].reshape((self.M, self.D))
        return Z, P, X

    def loss_function(self, parameter_vector, observed_fitness, norm, l2_lambda):
        Z, P, X = self.reconstruct_ZP(parameter_vector)
        predicted_fitness = self.landscape.calculate_fitness(Z, P, X)
        weighted_residuals = (observed_fitness - predicted_fitness) / norm
        data_loss = jnp.mean(jnp.square(weighted_residuals))
        reg_loss = l2_lambda * (jnp.sum(jnp.square(Z)) + jnp.sum(jnp.square(P)))
        return data_loss + reg_loss

    def calculate_bic(self, parameter_vector, observed_fitness, norm):
        Z, P, X = self.reconstruct_ZP(parameter_vector)
        predicted_fitness = self.landscape.calculate_fitness(Z, P, X)
        residuals = (observed_fitness - predicted_fitness) / norm
        rss = jnp.sum(jnp.square(residuals))
        n = observed_fitness.size
        k = len(parameter_vector)
        return k * jnp.log(n) + n * jnp.log(rss / n)

def run_inference_for_d(d_val, observed, norm, num_seeds, base_key):
    ls_obj = Landscape(C=3, D=d_val, M=28)
    prob_obj = RegressionProblem(ls_obj, observed, norm)
    solver = jaxopt.ScipyMinimize(method="L-BFGS-B", fun=prob_obj.loss_function, maxiter=MAX_REGRESS_ITER)
    
    best_loss = float('inf')
    best_params = None
    all_results = []
    current_key = base_key

    print(f"\n>>> Starting Inference for D={d_val} ({num_seeds} seeds)...")
    for i in range(num_seeds):
        current_key, Z_init, P_init, X_init = ls_obj.generate_initial_guess(current_key)
        init_pv = prob_obj.get_parameter_vector(Z_init, P_init, X_init)
        
        try:
            res = solver.run(init_pv, observed, norm, L2_LAMBDA)
            if res.state.fun_val < best_loss:
                best_loss = float(res.state.fun_val)
                best_params = res.params
            all_results.append(res)
        except Exception as e:
            continue
        
        if (i + 1) % 50 == 0: 
            print(f"  Completed {i+1}/{num_seeds}")

    bic = prob_obj.calculate_bic(best_params, observed, norm)
    fZ, fP, fX = prob_obj.reconstruct_ZP(best_params)
    r2 = calculate_r_squared(observed, ls_obj.calculate_fitness(fZ, fP, fX))
    
    return {
        "bic": float(bic),
        "loss": best_loss,
        "r2": float(r2),
        "params": best_params,
        "landscape": ls_obj,
        "problem": prob_obj
    }

def main():
    # --- Data Loading ---
    try:
        MiceG12C = pd.read_csv(r"Figure5A.csv")
        MiceG12D = pd.read_csv(r"Figure5B.csv")
        MiceEGFR = pd.read_csv(r"Figure5D.csv")
    except FileNotFoundError:
        print("Error: CSV files not found. Using dummy data for execution testing.")
        MiceG12C = pd.DataFrame({'gene': range(28), 'tumor_enrichment_x': [0.1]*28, 'CI_upper_x': [0.12]*28, 'CI_lower_x': [0.08]*28})
        MiceG12D = pd.DataFrame({'gene': range(28), 'tumor_enrichment_y': [0.1]*28, 'CI_upper_y': [0.12]*28, 'CI_lower_y': [0.08]*28})
        MiceEGFR = pd.DataFrame({'gene': range(28), 'tumor_enrichment': [0.1]*28, 'CI_upper': [0.12]*28, 'CI_lower': [0.08]*28})

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

    # --- Run Comparisons ---
    master_key = random.PRNGKey(BASE_SEED)
    
    results_d1 = run_inference_for_d(1, observed, norm, NUM_SEEDS, master_key)
    results_d2 = run_inference_for_d(2, observed, norm, NUM_SEEDS, master_key)

    # --- Print BIC Report ---
    print("\n" + "="*30)
    print("MODEL COMPARISON SUMMARY")
    print("="*30)
    print(f"D=1: BIC = {results_d1['bic']:.2f} | R^2 = {results_d1['r2']:.4f}")
    print(f"D=2: BIC = {results_d2['bic']:.2f} | R^2 = {results_d2['r2']:.4f}")
    
    delta_bic = results_d1['bic'] - results_d2['bic']
    if delta_bic > 10:
        print(f"\nResult: Strong evidence for D=2 (ΔBIC = {delta_bic:.2f})")
        winning_res = results_d2
    elif delta_bic > 0:
        print(f"\nResult: Weak evidence for D=2 (ΔBIC = {delta_bic:.2f})")
        winning_res = results_d2
    else:
        print(f"\nResult: D=1 preferred. D=2 is likely overfitting (ΔBIC = {delta_bic:.2f})")
        winning_res = results_d1



if __name__ == "__main__":
    main()
