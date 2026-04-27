# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 11:06:46 2026

@author: Meaghan Parks
"""

# -*- coding: utf-8 -*-

import jax
import jax.numpy as jnp
from jax import random
import jaxopt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns

# --- Configuration ---
MAX_REGRESS_ITER = 50000 
BASE_SEED = 404
TARGET_D = 2  
L2_LAMBDA = 1e-4 

# --- Robust Helper Functions ---

def robust_metrics(observed, predicted):
    """Calculates R^2 and PCC while safely ignoring non-finite values."""
    obs_flat = np.ravel(observed)
    pred_flat = np.ravel(predicted)
    
    mask = np.isfinite(obs_flat) & np.isfinite(pred_flat)
    
    if not np.any(mask):
        return np.nan, np.nan
        
    y_true = obs_flat[mask]
    y_pred = pred_flat[mask]
    
    # 1. Pearson Correlation
    pcc, _ = pearsonr(y_true, y_pred)
    
    # 2. R-squared
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot)
    
    return r2, pcc

class Landscape:
    def __init__(self, C=3, D=2, M=28, CONSTRAIN_ROTATION=True):
        self.C = C
        self.D = D
        self.M = M
        self.CONSTRAIN_ROTATION = CONSTRAIN_ROTATION

    def calculate_fitness(self, Z, P, X):
        # Ensure X doesn't become zero or negative to avoid NaN in log
        X_safe = jnp.where(jnp.abs(X) < 1e-6, 1e-6, jnp.abs(X))
        combined_phenotype = Z[:, jnp.newaxis, :] + P[jnp.newaxis, :, :]
        dist_sq = jnp.sum(jnp.square(combined_phenotype), axis=2)
        return jnp.log(X_safe) - (dist_sq / 2.0)

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

    def loss_function(self, params, observed, norm, l2_lambda, mask):
        Z, P, X = self.reconstruct_ZP(params)
        predicted = self.landscape.calculate_fitness(Z, P, X)
        weighted_res = (observed - predicted) / norm
        sq_residuals = jnp.square(weighted_res)
        data_loss = jnp.sum(sq_residuals * mask) / (jnp.sum(mask) + 1e-8)
        reg_loss = l2_lambda * (jnp.sum(jnp.square(Z)) + jnp.sum(jnp.square(P)))
        return data_loss + reg_loss

# --- Data Loading ---
MiceG12C = pd.read_csv(r"Figure5A.csv")
MiceG12D = pd.read_csv(r"Figure5B.csv")
MiceEGFR = pd.read_csv(r"Figure5D.csv")

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

def get_empirical_initial_params(obs_matrix, target_d, key):
    # Mean fitness per Mouse Model (Z) and per Mutant (P)
    z_means = -jnp.mean(obs_matrix, axis=1, keepdims=True)
    p_means = -jnp.mean(obs_matrix, axis=0, keepdims=True)
    
    Z_init = jnp.tile(z_means, (1, target_d))
    P_init = jnp.tile(p_means.T, (1, target_d))
    
    k1, k2 = random.split(key)
    Z_init += random.normal(k1, Z_init.shape) * 0.05
    P_init += random.normal(k2, P_init.shape) * 0.05
    return Z_init, P_init

ls_obj = Landscape(C=3, D=TARGET_D, M=28, CONSTRAIN_ROTATION=False)
prob_obj = RegressionProblem(ls_obj, observed, norm)
solver = jaxopt.ScipyMinimize(method="L-BFGS-B", fun=prob_obj.loss_function, maxiter=MAX_REGRESS_ITER)
key = random.PRNGKey(BASE_SEED)

# --- 1. Global Fit ---
print("Running Global Fit with Empirical Initial Guesses...")
Z_emp, P_emp = get_empirical_initial_params(observed, TARGET_D, key)
init_pv = prob_obj.get_parameter_vector(Z_emp, P_emp, 1.0)
global_res = solver.run(init_pv, observed, norm, L2_LAMBDA, jnp.ones_like(observed))

Pred_Z, Pred_P, Pred_X = prob_obj.reconstruct_ZP(global_res.params)
pred_global = ls_obj.calculate_fitness(Pred_Z, Pred_P, Pred_X)

# Robust Metrics
r2_global, pcc_global = robust_metrics(observed, pred_global)

# --- 2. 5-Fold Cross-Validation ---
def run_5fold_cv_empirical(prob_obj, solver, observed, norm, l2_lambda, k=5):
    n_conditions, n_mutants = observed.shape
    total_points = n_conditions * n_mutants
    indices = np.arange(total_points)
    np.random.seed(BASE_SEED)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    
    cv_predictions = np.zeros_like(observed)
    for i in range(k):
        test_indices = folds[i]
        train_indices = np.concatenate([folds[j] for j in range(k) if j != i])
        
        mask_flat = np.zeros(total_points)
        mask_flat[train_indices] = 1.0
        mask = jnp.array(mask_flat.reshape((n_conditions, n_mutants)))
        
        # Empirical initialization using only training data
        train_data = jnp.where(mask == 1.0, observed, jnp.nan)
        z_fold = -jnp.nanmean(train_data, axis=1, keepdims=True)
        p_fold = -jnp.nanmean(train_data, axis=0, keepdims=True)
        
        z_fold = jnp.nan_to_num(z_fold, nan=0.0)
        p_fold = jnp.nan_to_num(p_fold, nan=0.0)
        
        fold_init_pv = prob_obj.get_parameter_vector(jnp.tile(z_fold, (1, TARGET_D)), 
                                                   jnp.tile(p_fold.T, (1, TARGET_D)), 1.0)
        
        res = solver.run(fold_init_pv, observed, norm, l2_lambda, mask)
        Z_cv, P_cv, X_cv = prob_obj.reconstruct_ZP(res.params)
        pred_full = ls_obj.calculate_fitness(Z_cv, P_cv, X_cv)
        
        test_mask = np.zeros(total_points)
        test_mask[test_indices] = 1.0
        cv_predictions = np.where(test_mask.reshape((n_conditions, n_mutants)) == 1.0, pred_full, cv_predictions)
        print(f"Fold {i+1}/{k} complete.")
    return cv_predictions

print("\nRunning 5-Fold CV...")
pred_cv = run_5fold_cv_empirical(prob_obj, solver, observed, norm, L2_LAMBDA)

# Robust CV Metrics
r2_cv, pcc_cv = robust_metrics(observed, pred_cv)

print("\n" + "="*40)
print("FINAL MODEL PERFORMANCE SUMMARY")
print("="*40)
print(f"Global Fit R^2:  {r2_global:.4f}")
print(f"Global Fit PCC:  {pcc_global:.4f}")
print("-" * 20)
print(f"5-Fold CV R^2:   {r2_cv:.4f}")
print(f"5-Fold CV PCC:   {pcc_cv:.4f}")
print("="*40)

# --- Scatter Plot of Unseen (CV) Data ---

condition_names = ['G12C', 'G12D', 'EGFR']
obs_flat = np.array(observed).flatten()
pred_flat = np.array(pred_cv).flatten()

model_labels = []
for name in condition_names:
    model_labels.extend([name] * observed.shape[1])

df = pd.DataFrame({
    'Observed': obs_flat,
    'Predicted': pred_flat,
    'Model': model_labels
})

# 2. Calculate Statistics
# Global Stats
mask_global = np.isfinite(obs_flat) & np.isfinite(pred_flat)
glob_r, glob_p = pearsonr(obs_flat[mask_global], pred_flat[mask_global])

# Non-EGFR Stats
non_egfr_df = df[df['Model'] != 'EGFR']
mask_non = np.isfinite(non_egfr_df['Observed']) & np.isfinite(non_egfr_df['Predicted'])
non_r, non_p = pearsonr(non_egfr_df['Observed'][mask_non], non_egfr_df['Predicted'][mask_non])

# 3. Plotting
plt.figure(figsize=(10, 9))
sns.set_style("white")


palette = {'G12C': '#72a2c9', 'G12D': '#d96a6a', 'EGFR': '#76bf77'}

scatter = sns.scatterplot(
    data=df, x='Observed', y='Predicted', hue='Model', 
    palette=palette, s=80, alpha=0.8, edgecolor='w', linewidth=0.5
)


lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
plt.plot(lims, lims, color='gray', linestyle='--', alpha=0.8, zorder=0, label='Identity Line')


stats_text = (
    f"Global Pearson r: {glob_r:.4f}\n"
    f"Global p-val: {glob_p:.4e}\n"
    "------------------------\n"
    f"Non-EGFR Pearson r: {non_r:.4f}\n"
    f"Non-EGFR p-val: {non_p:.4e}"
)

plt.text(
    0.05, 0.65, stats_text, 
    transform=plt.gca().transAxes, 
    fontsize=12, verticalalignment='top',
    bbox=dict(boxstyle='square,pad=0.5', facecolor='white', alpha=0.5, edgecolor='gray')
)


plt.title("5-Fold Cross-Validation: Global vs. Non-EGFR Performance", fontsize=15)
plt.xlabel("Measured Log Fitness", fontsize=13)
plt.ylabel("Predicted Log Fitness", fontsize=13)


plt.legend(loc='lower right', frameon=True, fontsize=11)

plt.tight_layout()
plt.savefig("CV5AverageStart.pdf", dpi=300)
plt.show()

def run_half_leave_out_cv(prob_obj, solver, observed, norm, l2_lambda):
    n_conditions, n_mutants = observed.shape
    cv_predictions = np.full_like(observed, np.nan) # Initialize with NaNs
    
    
    n_splits = 2
    indices = np.arange(n_mutants)
    np.random.seed(BASE_SEED)
    np.random.shuffle(indices)
    gene_folds = np.array_split(indices, n_splits)

    for cond_idx in range(n_conditions):
        for fold_idx in range(n_splits):
            
            mask = np.ones((n_conditions, n_mutants))
            
            
            test_gene_indices = gene_folds[fold_idx]
            
            mask[cond_idx, test_gene_indices] = 0.0
            mask_jax = jnp.array(mask)

            # Empirical initialization (ignoring the masked half)
            train_data = jnp.where(mask_jax == 1.0, observed, jnp.nan)
            z_fold = jnp.nan_to_num(-jnp.nanmean(train_data, axis=1, keepdims=True), nan=0.0)
            p_fold = jnp.nan_to_num(-jnp.nanmean(train_data, axis=0, keepdims=True), nan=0.0)
            
            fold_init_pv = prob_obj.get_parameter_vector(
                jnp.tile(z_fold, (1, TARGET_D)), 
                jnp.tile(p_fold.T, (1, TARGET_D)), 
                1.0
            )
            

            res = solver.run(fold_init_pv, observed, norm, l2_lambda, mask_jax)
            Z_cv, P_cv, X_cv = prob_obj.reconstruct_ZP(res.params)
            pred_full = ls_obj.calculate_fitness(Z_cv, P_cv, X_cv)

            cv_predictions[cond_idx, test_gene_indices] = pred_full[cond_idx, test_gene_indices]
            
        print(f"Condition {cond_idx+1}/{n_conditions} halves complete.")
        
    return cv_predictions


pred_half_cv = run_half_leave_out_cv(prob_obj, solver, observed, norm, L2_LAMBDA)


residuals = np.array(observed) - pred_half_cv


plt.figure(figsize=(16, 6))
gene_names = AllMice['gene'].values
condition_names = ['G12C', 'G12D', 'EGFR']


sns.heatmap(residuals, 
            xticklabels=gene_names, 
            yticklabels=condition_names, 
            cmap='RdBu_r', 
            center=0,
            cbar_kws={'label': 'Residual (Log Fitness Error)'})

plt.title("Leave-Half-Out CV Residual Heatmap: Observed - Predicted")
plt.xlabel("Mutants (Genes)")
plt.ylabel("Conditions (Mice Models)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()



print("\n" + "="*40)
print("PER-CONDITION STATISTICAL SUMMARY")
print("="*40)

condition_pccs = []
condition_pvals = []
condition_names = ['G12C', 'G12D', 'EGFR']

for i in range(len(condition_names)):
   
    y_true = np.ravel(observed[i, :])
    y_pred = np.ravel(pred_half_cv[i, :])
    
    
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) > 1:
       
        pcc_val, p_val = pearsonr(y_true_clean, y_pred_clean)
        
        condition_pccs.append(pcc_val)
        condition_pvals.append(p_val)
        
        
        print(f"{condition_names[i]:<6} | PCC: {pcc_val:.4f} | P-value: {p_val:.2e}")
    else:
        print(f"{condition_names[i]:<6} | Insufficient valid data points.")



LARGE_FONT = 18
MEDIUM_FONT = 15
SMALL_FONT = 12


residuals = np.array(observed) - pred_half_cv


fig, ax = plt.subplots(figsize=(22, 8)) 
gene_names = AllMice['gene'].values
condition_names = ['G12C', 'G12D', 'EGFR']


sns.heatmap(residuals, 
            ax=ax,
            xticklabels=gene_names, 
            yticklabels=condition_names, 
            cmap='RdBu_r', 
            center=0,
            cbar_kws={
                'label': 'Residual (Log Fitness Error)', 
                'pad': 0.02,     
                'shrink': 0.8    
            })


ax.figure.axes[-1].yaxis.label.set_size(MEDIUM_FONT)

# 4. Calculate and Annotate Stats

for i, name in enumerate(condition_names):
    y_true = np.ravel(observed[i, :])
    y_pred = np.ravel(pred_half_cv[i, :])
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    
    if np.any(mask):
        pcc, pval = pearsonr(y_true[mask], y_pred[mask])
        stats_text = f"PCC: {pcc:.3f}\nP: {pval:.2e}"
        
       
        ax.text(1.12, 1 - (i + 0.5)/len(condition_names), stats_text, 
                transform=ax.transAxes,
                va='center', ha='left', fontsize=MEDIUM_FONT, 
                fontweight='bold', color='black',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', lw=1))

ax.set_title("Leave-Half-Out CV Residual Heatmap", fontsize=LARGE_FONT, pad=30)
ax.set_xlabel("Mutants (Genes)", fontsize=MEDIUM_FONT, labelpad=15)
ax.set_ylabel("Conditions (Mice Models)", fontsize=MEDIUM_FONT, labelpad=15)


plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=SMALL_FONT)
plt.setp(ax.get_yticklabels(), rotation=0, fontsize=MEDIUM_FONT)

plt.tight_layout(rect=[0, 0, 0.85, 1]) 
plt.savefig("CorrectedStartCV2.pdf",dpi=300)
plt.show()



def run_loocv(prob_obj, solver, observed, norm, l2_lambda):
    n_conditions, n_mutants = observed.shape
    total_points = n_conditions * n_mutants
    loocv_predictions = np.zeros_like(observed)
    
    print(f"Starting LOOCV for {total_points} total points...")
    
    
    for c in range(n_conditions):
        for m in range(n_mutants):
           
            mask = np.ones((n_conditions, n_mutants))
            mask[c, m] = 0.0
            mask_jax = jnp.array(mask)
            
            
            train_data = jnp.where(mask_jax == 1.0, observed, jnp.nan)
            z_fold = jnp.nan_to_num(-jnp.nanmean(train_data, axis=1, keepdims=True), nan=0.0)
            p_fold = jnp.nan_to_num(-jnp.nanmean(train_data, axis=0, keepdims=True), nan=0.0)
            
            fold_init_pv = prob_obj.get_parameter_vector(
                jnp.tile(z_fold, (1, TARGET_D)), 
                jnp.tile(p_fold.T, (1, TARGET_D)), 
                1.0
            )
            
            
            res = solver.run(fold_init_pv, observed, norm, l2_lambda, mask_jax)
            Z_cv, P_cv, X_cv = prob_obj.reconstruct_ZP(res.params)
            

            pred_full = ls_obj.calculate_fitness(Z_cv, P_cv, X_cv)
            loocv_predictions[c, m] = pred_full[c, m]
            
        print(f"Condition {condition_names[c]} complete.")
        
    return loocv_predictions

# Execute the LOOCV
pred_loocv = run_loocv(prob_obj, solver, observed, norm, L2_LAMBDA)

# Calculate final metrics
r2_loocv, pcc_loocv = robust_metrics(observed, pred_loocv)


import matplotlib.gridspec as gridspec


loocv_residuals = np.array(observed) - pred_loocv


fig = plt.figure(figsize=(22, 8))
gs = gridspec.GridSpec(1, 3, width_ratios=[15, 0.5, 3], wspace=0.1)

ax_main = fig.add_subplot(gs[0])
ax_cbar = fig.add_subplot(gs[1])
ax_text = fig.add_subplot(gs[2])

sns.heatmap(loocv_residuals, 
            ax=ax_main,
            xticklabels=gene_names, 
            yticklabels=condition_names, 
            cmap='RdBu_r', 
            center=0,
            cbar_ax=ax_cbar, 
            cbar_kws={'label': 'Residual (Log Fitness Error)'})


ax_text.axis('off') 
for i, name in enumerate(condition_names):
    y_true = np.ravel(observed[i, :])
    y_pred = np.ravel(pred_loocv[i, :])
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    
    if np.any(mask):
        pcc, pval = pearsonr(y_true[mask], y_pred[mask])
        stats_text = f"PCC: {pcc:.3f}\nP: {pval:.2e}"
        
       
        y_pos = 1 - (i + 0.5) / len(condition_names)
        
        ax_text.text(0.1, y_pos, stats_text, 
                     transform=ax_text.transAxes,
                     va='center', ha='left', fontsize=14, 
                     fontweight='bold', 
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))


ax_main.set_title("LOOCV Residual Heatmap: Measured - Predicted", fontsize=18, pad=20)
ax_main.set_xlabel("Mutants (Genes)", fontsize=15, labelpad=10)
ax_main.set_ylabel("Conditions (Mice Models)", fontsize=15)

plt.setp(ax_main.get_xticklabels(), rotation=45, ha='right', fontsize=12)
plt.setp(ax_main.get_yticklabels(), rotation=0, fontsize=14)


ax_cbar.yaxis.label.set_size(13)
plt.savefig("LOOAverage.pdf", dpi=300)
plt.show()
