# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 12:39:47 2026

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
MAX_REGRESS_ITER = 500000
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

def run_leave_out_env_cv(regression_obj, fitness_data, norm_data, Z_init, P_init, X_init, 
                         env_idx=0, leave_out_frac=0.2, SplitSeed=42):
    """
    Leaves out a fraction of a single environment and calculates MAE and Pearson r.
    """
    num_envs, num_mutants = fitness_data.shape
    
    # 1. Select indices within the target environment
    np.random.seed(SplitSeed)
    indices_in_env = np.arange(num_mutants)
    np.random.shuffle(indices_in_env)
    
    num_to_mask = int(num_mutants * leave_out_frac)
    test_mutant_indices = indices_in_env[:num_to_mask]
    
    # 2. Convert to flattened index
    test_indices_flat = (env_idx * num_mutants) + test_mutant_indices
    
    # 3. Run the regression
    obs_flat = fitness_data.ravel()
    norm_flat = norm_data.ravel()
    
    final_params = regress_LBFGS(
        regression_obj, obs_flat, norm_flat, 
        Z_init, P_init, X_init, 
        masked_indices=test_indices_flat
    )
    
    # 4. Reconstruct and calculate stats
    Z_f, P_f, X_f = regression_obj.reconstruct_ZP(final_params, regression_obj.D)
    full_pred = regression_obj.landscape.calculate_fitness(Z_f, P_f, X_f)
    
    # Extract hidden (test) values
    actual_test = obs_flat[test_indices_flat]
    pred_test = full_pred.ravel()[test_indices_flat]
    sigma_test = norm_flat[test_indices_flat]
    
    # --- METRICS FOR MISSING DATA ---
    weighted_mae = jnp.mean(jnp.abs((actual_test - pred_test) / sigma_test))
    
    # Pearson Correlation Coefficient (r)
    # returns a 2x2 matrix; [0,1] is the correlation between the two arrays
    pearson_matrix = jnp.corrcoef(actual_test, pred_test)
    pearson_r = pearson_matrix[0, 1]

    print(f"--- Env {env_idx} (Hidden {leave_out_frac*100}%) ---")
    print(f"Weighted MAE: {weighted_mae:.4f}")
    print(f"Pearson r:    {pearson_r:.4f}")

    return {
        "parameters": {"Z": Z_f, "P": P_f, "X": X_f},
        "predictions": full_pred,
        "test_indices": test_indices_flat,
        "weighted_mae": weighted_mae,
        "pearson_r": pearson_r
    }
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

# Leave out 30% of the second environment (index 1)
results = run_leave_out_env_cv(
    regression_obj, 
    mice_fitness, 
    ALLMiceNormNP, 
    Z_start, P_start, X_start, 
    env_idx=1,           # Target environment
    leave_out_frac=0.3,  # 30% of that environment
    SplitSeed=42
)

# Plotting specific to the test set
plt.scatter(mice_fitness.ravel()[results['test_indices']], 
            results['predictions'].ravel()[results['test_indices']])
plt.title("Performance on Hidden Portion of Environment 1")
plt.show()

def plot_env_specific_results(results, fitness_data, env_idx):
    obs_flat = np.array(fitness_data.ravel())
    pred_flat = np.array(results['predictions'].ravel())
    test_idx = results['test_indices']
    
    # Split data for visualization
    train_mask = np.ones(obs_flat.size, dtype=bool)
    train_mask[test_idx] = False
    
    plt.figure(figsize=(8, 8))
    
    # Background: Training data
    plt.scatter(obs_flat[train_mask], pred_flat[train_mask], 
                alpha=0.2, color='gray', label='Training Data', s=20)
    
    # Foreground: Missing data
    plt.scatter(obs_flat[test_idx], pred_flat[test_idx], 
                alpha=0.9, color='#27ae60', edgecolors='k', 
                label=f'Missing Data (Env {env_idx})', s=70)
    
    # Identity line and scaling
    lo = float(np.min(obs_flat)) - 0.5
    hi = float(np.max(obs_flat)) + 0.5
    plt.plot([lo, hi], [lo, hi], 'r--', lw=2)
    plt.xlim(lo, hi)
    plt.ylim(lo, hi)
    
    # Annotate Pearson r and MAE for missing data
    stats_text = (f"Missing Data Stats:\n"
                  f"Pearson $r$: {results['pearson_r']:.3f}\n"
                  f"MAE: {results['weighted_mae']:.3f}")
    
    plt.annotate(stats_text, xy=(0.05, 0.85), xycoords='axes fraction', 
                 fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#27ae60", lw=2))
    
    plt.xlabel("Observed Log-Fitness")
    plt.ylabel("Predicted Log-Fitness")
    plt.title(f"Performance on Left-Out Data: Environment {env_idx}")
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.show()

# Define the environment labels based on your data merging
from scipy import stats

env_labels = ["G12C", "G12D", "EGFR"]
summary_results = []

print("--- Cross-Environment Validation with Significance (Leave-out 30%) ---")

for i, label in enumerate(env_labels):
    res = run_leave_out_env_cv(
        regression_obj, 
        mice_fitness, 
        ALLMiceNormNP, 
        Z_start, P_start, X_start, 
        env_idx=i, 
        leave_out_frac=0.10, 
        SplitSeed=42
    )
    
    # Extract the raw test values to compute p-value
    obs_flat = mice_fitness.ravel()
    pred_flat = res['predictions'].ravel()
    test_idx = res['test_indices']
    
    y_true = obs_flat[test_idx]
    y_pred = pred_flat[test_idx]
    
    # Calculate Pearson r and p-value
    # r_val is the correlation, p_val is the significance
    r_val, p_val = stats.pearsonr(y_true, y_pred)
    
    summary_results.append({
        "Environment": label,
        "Pearson_r": float(r_val),
        "p-value": float(p_val),
        "MAE": float(res['weighted_mae']),
        "Significant": "Yes" if p_val < 0.05 else "No"
    })

summary_df = pd.DataFrame(summary_results)
print("\nFinal Performance Summary:")
print(summary_df.to_string(index=False))
def plot_summary_comparison(df):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar plot for Pearson r
    color_r = '#27ae60'
    ax1.set_xlabel('Environment')
    ax1.set_ylabel('Pearson Correlation ($r$)', color=color_r, fontsize=12)
    bars = ax1.bar(df['Environment'], df['Pearson_r'], color=color_r, alpha=0.6, label='Pearson $r$')
    ax1.tick_params(axis='y', labelcolor=color_r)
    ax1.set_ylim(0, 1.1)

    # Line plot for MAE
    ax2 = ax1.twinx() 
    color_mae = '#e67e22'
    ax2.set_ylabel('Weighted MAE', color=color_mae, fontsize=12)
    ax2.plot(df['Environment'], df['MAE'], color=color_mae, marker='o', linewidth=3, markersize=10, label='MAE')
    ax2.tick_params(axis='y', labelcolor=color_mae)
    
    # Add a horizontal line for "Strong Correlation" threshold
    ax1.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5)
    ax1.text(-0.4, 0.72, 'Strong Correlation Threshold', color='gray', fontsize=10)

    plt.title('Predictive Performance Across Environments (Missing Data)', pad=20)
    fig.tight_layout()
    plt.show()

plot_summary_comparison(summary_df)
import seaborn as sns
def plot_significance_comparison(df):
    plt.figure(figsize=(10, 6))
    
    # Plot Pearson r
    sns.barplot(x='Environment', y='Pearson_r', data=df, palette='viridis', alpha=0.8)
    
    # Add significance stars
    for i, row in df.iterrows():
        stars = ""
        if row['p-value'] < 0.001: stars = "***"
        elif row['p-value'] < 0.01: stars = "**"
        elif row['p-value'] < 0.05: stars = "*"
        
        plt.text(i, row['Pearson_r'] + 0.02, stars, ha='center', fontsize=15, fontweight='bold')

    plt.ylim(0, 1.1)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.ylabel("Pearson Correlation ($r$)", fontsize=12)
    plt.title("Model Predictive Significance by Environment\n(*p<0.05, **p<0.01, ***p<0.001)", pad=20)
    plt.show()

plot_significance_comparison(summary_df)


env_labels = ["G12C", "G12D", "EGFR"]
summary_results = []
SIG_THRESHOLD = 0.01  # Only count mutations with an effect size > 0.1

print(f"--- Directional Accuracy (Sign Prediction) for Hidden Data ---")

for i, label in enumerate(env_labels):
    # 1. Run the specific leave-out
    res = run_leave_out_env_cv(
        regression_obj, mice_fitness, ALLMiceNormNP, 
        Z_start, P_start, X_start, 
        env_idx=i, leave_out_frac=0.4, SplitSeed=42
    )
    
    # 2. Extract observed and predicted values for the TEST set
    obs_flat = mice_fitness.ravel()
    pred_flat = res['predictions'].ravel()
    test_idx = res['test_indices']
    
    y_true_test = obs_flat[test_idx]
    y_pred_test = pred_flat[test_idx]
    
    # 3. Directional Logic (Sign check)
    # Filter for significant true effects
    significant_mask = jnp.abs(y_true_test) > SIG_THRESHOLD
    
    # Count matches in sign
    sign_match = (jnp.sign(y_true_test) == jnp.sign(y_pred_test))
    correct_signs = jnp.logical_and(sign_match, significant_mask)
    
    # Calculate Percentage
    total_sig = jnp.sum(significant_mask)
    acc = (jnp.sum(correct_signs) / total_sig * 100) if total_sig > 0 else 0.0
    
    summary_results.append({
        "Environment": label,
        "Pearson_r": float(res['pearson_r']),
        "Directional_Accuracy_%": float(acc),
        "Significant_Mutants_Tested": int(total_sig)
    })

# Display the sign-prediction results
sign_summary_df = pd.DataFrame(summary_results)
print("\nSign Prediction Performance:")
print(sign_summary_df.to_string(index=False))


def plot_directional_quadrants(y_true, y_pred, threshold=0.01):
    plt.figure(figsize=(8, 6))
    
    # Mask for significant points
    sig = np.abs(y_true) > threshold
    y_t = y_true[sig]
    y_p = y_pred[sig]
    
    # Colors: Green for correct sign, Red for flipped sign
    colors = ['#2ecc71' if np.sign(t) == np.sign(p) else '#e74c3c' for t, p in zip(y_t, y_p)]
    
    plt.scatter(y_t, y_p, c=colors, edgecolor='k', s=80, alpha=0.8)
    
    # Add Quadrant Lines
    plt.axhline(0, color='black', lw=1)
    plt.axvline(0, color='black', lw=1)
    
    plt.xlabel("Observed Effect (True Sign)")
    plt.ylabel("Predicted Effect (Predicted Sign)")
    plt.title("Directional Accuracy: Hits (Green) vs Misses (Red)")
    
    # Labels for clarity
    plt.text(plt.xlim()[1]*0.5, plt.ylim()[1]*0.8, "Predicted Beneficial\nTrue Beneficial", color='green')
    plt.text(plt.xlim()[0]*0.8, plt.ylim()[0]*0.8, "Predicted Deleterious\nTrue Deleterious", color='green')
    
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.show()

# Example for the last run environment
plot_directional_quadrants(y_true_test, y_pred_test, threshold=SIG_THRESHOLD)

def run_kfold_on_single_env(regression_obj, fitness_data, norm_data, Z_init, P_init, X_init, 
                            target_env_idx=0, k=5, SplitSeed=42):
    """
    Performs K-Fold CV on a single environment. 
    Every gene in that environment will be hidden exactly once.
    """
    num_envs, num_mutants = fitness_data.shape
    indices = np.arange(num_mutants)
    
    np.random.seed(SplitSeed)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    
    all_test_obs = []
    all_test_pred = []
    
    print(f"--- Running {k}-Fold CV on Environment {target_env_idx} ---")
    
    for i in range(k):
        test_mutant_indices = folds[i]
        
        # Convert local mutant indices to global flattened indices
        test_indices_flat = (target_env_idx * num_mutants) + test_mutant_indices
        
        # Run regression (masking the current fold)
        final_params = regress_LBFGS(
            regression_obj, fitness_data.ravel(), norm_data.ravel(), 
            Z_init, P_init, X_init, 
            masked_indices=test_indices_flat
        )
        
        # Predict
        Z_f, P_f, X_f = regression_obj.reconstruct_ZP(final_params, regression_obj.D)
        full_pred = regression_obj.landscape.calculate_fitness(Z_f, P_f, X_f)
        
        # Store only the "hidden" predictions for this fold
        all_test_obs.extend(fitness_data.ravel()[test_indices_flat])
        all_test_pred.extend(full_pred.ravel()[test_indices_flat])
        
        print(f"Fold {i+1}/{k} complete.")

    return jnp.array(all_test_obs), jnp.array(all_test_pred)

from scipy import stats

env_labels = ["G12C", "G12D", "EGFR"]
final_results = []
THRESHOLD = 0.1 # For sign accuracy

for env_idx, label in enumerate(env_labels):
    # Execute K-Fold for this environment
    y_true, y_pred = run_kfold_on_single_env(
        regression_obj, mice_fitness, ALLMiceNormNP, 
        Z_start, P_start, X_start, 
        target_env_idx=env_idx, k=7, # 28 mutants / 7 = 4 mutants per fold
        SplitSeed=42
    )
    
    # 1. Pearson Correlation
    r, p = stats.pearsonr(y_true, y_pred)
    
    # 2. MAE
    mae = jnp.mean(jnp.abs(y_true - y_pred))
    
    # 3. Directional Accuracy (Sign Check)
    sig_mask = jnp.abs(y_true) > THRESHOLD
    correct_signs = (jnp.sign(y_true) == jnp.sign(y_pred))
    acc = (jnp.sum(correct_signs[sig_mask]) / jnp.sum(sig_mask)) * 100
    
    final_results.append({
        "Env": label,
        "Pearson_r": float(r),
        "MAE": float(mae),
        "Sign_Acc_%": float(acc)
    })

# Display Final Table
performance_df = pd.DataFrame(final_results)
print("\nFinal Performance (Every Gene Tested once per Env):")
print(performance_df.to_string(index=False))

def plot_final_kfold_results(y_true, y_pred, label):
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.7, edgecolors='k', color='#34495e')
    
    # Identify Quadrants for sign accuracy
    plt.axhline(0, color='gray', lw=1, linestyle='--')
    plt.axvline(0, color='gray', lw=1, linestyle='--')
    
    # Identity line
    lims = [min(plt.xlim()[0], plt.ylim()[0]), max(plt.xlim()[1], plt.ylim()[1])]
    plt.plot(lims, lims, 'r-', alpha=0.75, zorder=0)
    
    plt.xlabel("Observed Log-Fitness")
    plt.ylabel("Predicted (when hidden)")
    plt.title(f"Full Gene-by-Gene CV: {label}")
    plt.grid(True, alpha=0.3)
    plt.show()

# Example: Plotting results for EGFR (env_idx 2)
# (Assuming you saved y_true/y_pred from the loop above)
plot_final_kfold_results(y_true, y_pred, "EGFR")


import seaborn as sns
from sklearn.metrics import confusion_matrix

env_labels = ["G12C", "G12D", "EGFR"]
accuracy_stats = []
all_y_true = []
all_y_pred = []
THRESHOLD = 0.0 # Only count effects larger than 0.05 in log-fitness

for env_idx, label in enumerate(env_labels):
    # Run K-Fold (using k=7 for 4 genes per fold)
    y_true, y_pred = run_kfold_on_single_env(
        regression_obj, mice_fitness, ALLMiceNormNP, 
        Z_start, P_start, X_start, 
        target_env_idx=env_idx, k=7, SplitSeed=42
    )
    
    # Filter for significant mutations
    mask = jnp.abs(y_true) > THRESHOLD
    y_true_sig = y_true[mask]
    y_pred_sig = y_pred[mask]
    
    # Calculate Correct Sign Percentage
    matches = (jnp.sign(y_true_sig) == jnp.sign(y_pred_sig))
    acc = (jnp.sum(matches) / len(y_true_sig)) * 100
    
    accuracy_stats.append({"Env": label, "Accuracy": float(acc)})
    
    # Store for heatmap (using 1 for positive, -1 for negative)
    all_y_true.extend(jnp.sign(y_true_sig))
    all_y_pred.extend(jnp.sign(y_pred_sig))

# Create DataFrames for plotting
acc_df = pd.DataFrame(accuracy_stats)

plt.rcParams.update({'font.size': 12})

# Set the default font family (e.g., to a common serif or sans-serif)
# 'serif' often looks good in papers. Use 'sans-serif' for a cleaner look.
plt.rcParams.update({'font.family': 'sans-serif'})

# Settings for axes labels
plt.rcParams.update({
    'axes.labelsize': 12,      # Font size of the x and y labels
    'axes.titlesize': 14,      # Font size of the plot title
    'axes.linewidth': 0.8,     # Thickness of the plot borders
    'axes.edgecolor': 'black'  # Color of the plot borders
})

# Settings for tick marks
plt.rcParams.update({
    'xtick.labelsize': 10,     # Font size of the x tick labels
    'ytick.labelsize': 10,     # Font size of the y tick labels
    'xtick.direction': 'in',   # Tick marks point inward
    'ytick.direction': 'in',   # Tick marks point inward
    'xtick.major.size': 4,     # Length of major x ticks
    'ytick.major.size': 4,     # Length of major y ticks
})

# Settings for the legend
plt.rcParams.update({
    'legend.fontsize': 10,     # Font size of the legend
    'legend.frameon': True,    # Draw a box around the legend
    'legend.edgecolor': 'black',
    'legend.fancybox': False   # Use a sharp-cornered box
})
plt.figure(figsize=(8, 6))
bars = plt.bar(acc_df['Env'], acc_df['Accuracy'], color=['#3498db', '#9b59b6', '#2ecc71'], edgecolor='black', alpha=0.8)
plt.axhline(50, color='red', linestyle='--', label='Random Chance (50%)')

plt.ylabel("Sign Prediction Accuracy (%)", fontsize=12)
plt.title("Model Ability to Predict Direction of Effect, k=7", pad=20)
plt.ylim(0, 105)

# Add labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.1f}%', ha='center', fontweight='bold')

plt.legend()
plt.savefig("PredictedSignByMM.pdf", dpi=300)

plt.show()



# Assuming 'AllMice' dataframe exists from your data loading step
gene_names = AllMice['gene'].values
env_labels = ["G12C", "G12D", "EGFR"]
THRESHOLD = 0.0

# Initialize an empty matrix for the heatmap (3 environments x 28 genes)
# 1 = Correct Sign, -1 = Incorrect Sign, 0 = Below Threshold
sign_matrix = np.zeros((len(env_labels), len(gene_names)))

for env_idx, label in enumerate(env_labels):
    # Run K-Fold to get predictions for EVERY gene in this environment
    y_true, y_pred = run_kfold_on_single_env(
        regression_obj, mice_fitness, ALLMiceNormNP, 
        Z_start, P_start, X_start, 
        target_env_idx=env_idx, k=7, SplitSeed=42
    )
    
    # Identify significant effects
    sig_mask = np.abs(y_true) > THRESHOLD
    
    # Compare signs
    correct_sign = (np.sign(y_true) == np.sign(y_pred))
    
    # Fill the matrix
    # We assign 1 for correct, -1 for incorrect, 0 for insignificant
    for gene_i in range(len(gene_names)):
        if not sig_mask[gene_i]:
            sign_matrix[env_idx, gene_i] = 0  # White/Neutral
        elif correct_sign[gene_i]:
            sign_matrix[env_idx, gene_i] = 1  # Green/Correct
        else:
            sign_matrix[env_idx, gene_i] = -1 # Red/Incorrect
            
import matplotlib.colors as mcolors

plt.figure(figsize=(16, 5))

# Define custom color map: Red for incorrect (-1), White for neutral (0), Green for correct (1)
cmap = mcolors.ListedColormap(['#e74c3c', '#fdfefe', '#27ae60'])
bounds = [-1.5, -0.5, 0.5, 1.5]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

sns.heatmap(sign_matrix, 
            annot=False, 
            cmap=cmap, 
            norm=norm,
            xticklabels=gene_names, 
            yticklabels=env_labels, 
            linewidths=0.5, 
            linecolor='gray',
            cbar_kws={"ticks":[-1, 0, 1], "label": "Prediction Status"})

# Adjust the colorbar labels
colorbar = plt.gca().collections[0].colorbar
colorbar.set_ticklabels(['Sign Flip (Error)', 'Neutral (<0.05)', 'Correct Sign'])

plt.title("Sign Prediction Accuracy per Gene and Environment\n(Tested via K-Fold Cross-Validation)", pad=20, fontsize=16)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

def print_gene_predictions(regression_obj, fitness_data, gene_names, env_labels):
    print(f"{'Environment':<12} | {'Gene':<15} | {'Observed (z)':>12} | {'Predicted (ẑ)':>12} | {'Error':>10} | {'Sign Correct?'}")
    print("-" * 90)

    for env_idx, label in enumerate(env_labels):
        # Run K-Fold so every gene gets a 'hidden' prediction
        y_true, y_pred = run_kfold_on_single_env(
            regression_obj, fitness_data, ALLMiceNormNP, 
            Z_start, P_start, X_start, 
            target_env_idx=env_idx, k=7, SplitSeed=42
        )
        
        for i, gene in enumerate(gene_names):
            obs = float(y_true[i])
            prd = float(y_pred[i])
            err = prd - obs
            
            # Check if sign matches (Beneficial vs Deleterious)
            # Only meaningful if the effect is large enough (e.g., > 0.05)
            sign_match = "YES" if np.sign(obs) == np.sign(prd) else "NO"
            if abs(obs) < 0.05: sign_match = "Neutral"

            print(f"{label:<12} | {gene:<15} | {obs:>12.4f} | {prd:>12.4f} | {err:>10.4f} | {sign_match}")
        print("-" * 90)

# Run the printer
print_gene_predictions(regression_obj, mice_fitness, gene_names, env_labels)
