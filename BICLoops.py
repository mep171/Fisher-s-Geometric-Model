# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 17:31:56 2025

@author: Meaghan Parks
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
import numpy as np
from jax import random
from jax.scipy.optimize import minimize
import math
import jax.numpy as jnp
import jaxopt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

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
# --- CONSTANTS ---
MAX_REGRESS_ITER = 500
C = 20 # Number of contexts
M = 20 # Number of mutations
n = C * M # Total number of data points
REAL_X = 25
GUESS_X = 1
NOISE_LEVEL = 0.005

# --- NEW: Dimension Setup ---
# The true dimension of the simulated data
D_true_values = [1, 2, 3, 4, 5]
# The guess dimension used for the regression (D_guess)
D_guess_values = [1, 2, 3, 4, 5]

# --- Multiple Seed Setup ---
# Key for generating the 'true' simulated data (fixed)
seed_sim = 90
key_sim = random.PRNGKey(seed_sim)

# Keys for the initial 'guess' parameters (variable)
seed_guess_list = [980, 100, 201, 333, 20]
# Note: Use one key per run to ensure independent initial guesses
key_guess_list = [random.PRNGKey(s) for s in seed_guess_list]


# --- LANDSCAPE CLASS (No Change) ---
class landscape:
    def __init__(self, C=20, D=2, M=20, scale=1, CONSTRAIN_ROTATION=True, **dimensions):
        self.C = C
        self.D = D
        self.M = M
        self.scale=scale
        self.CONSTRAIN_ROTATION = CONSTRAIN_ROTATION
        for dimension, value in dimensions.items():
            setattr(self, dimension, value)

    def simulate_dataset(self, key, noise=0):
        # NOTE: D is taken from self.D
        key, key_z = random.split(key)
        Z=random.normal(key_z, (self.C, self.D))
        key, key_p = random.split(key)
        P=random.normal(key_p, (self.M, self.D))
        Noise_for_Z = noise*random.normal(key_z, Z.shape)
        key, key_p = random.split(key)
        Noise_for_P = noise*random.normal(key_p, P.shape)
        if self.CONSTRAIN_ROTATION:
            P = jnp.tril(P, -1)
            P = P.at[jnp.triu_indices_from(P,-1)].set(jnp.abs(P[jnp.triu_indices_from(P,-1)]))

        return key, Z + Noise_for_Z, P + Noise_for_P

    def calculate_fitness(self, Z, P, X, noise=0):
        # NOTE: D is implicitly taken from Z/P shapes
        tiledZ = jnp.tile(Z, (M,1))
        repedP = jnp.repeat(P,C,axis=0)
        repMutant = tiledZ + repedP
        Fitness = X*((jnp.exp( -jnp.einsum('cd,cd->c', repMutant, repMutant)/2)))
        return jnp.log(Fitness)

# --- REGRESSION PROBLEM CLASS (Only Reconstruct ZP modified to use self.D) ---
class RegressionProblem:
    def __init__(self, landscape_obj, observed_fitnesses, norm, key, C=20, D=2, M=20, CONSTRAIN_ROTATION=True, LOG_FITNESS=True):
        self.landscape = landscape_obj
        self.LOG_FITNESS=LOG_FITNESS
        self.key=key
        if self.LOG_FITNESS:
            self.observed_fitnesses = observed_fitnesses
        else:
            self.observed_fitnesses = jnp.log(observed_fitnesses)
        self.C = C
        self.D = D # This is the GUESS D
        self.M = M
        self.CONSTRAIN_ROTATION=CONSTRAIN_ROTATION

    def check_determined(self,Z,P):
        observations=len(self.observed_fitnesses)
        free_parameters=len(self.get_parameter_vector(Z,P))
        print("Under-determined" if observations<free_parameters else "Over-Determined")

    # get_parameter_vector MUST be modified to compute P_flat size based on self.D (the GUESS D)
    def get_parameter_vector(self, Z, P, X):
        # The true P is not used here. The *guess* Z and P are used to define the shape.
        # This function must be modified to calculate the constrained indices correctly based on the D in Z/P
        # D is embedded in Z.shape[1] and P.shape[1]
        D_guess = Z.shape[1]
        
        # Calculate size of P_flat for a D_guess dimension with M rows
        # The number of unique, non-zero parameters in a constrained MxD matrix is M*D - (D*(D+1)/2)
        
        # Create a dummy P of the guess size to find the indices
        dummy_P = jnp.zeros((self.M, D_guess))
        
        # Get the indices for the lower triangle (excluding diagonal)
        triu_indices = jnp.triu_indices_from(dummy_P, 0)
        constrained_indices = (triu_indices[0][triu_indices[0] < triu_indices[1]], 
                               triu_indices[1][triu_indices[0] < triu_indices[1]])
                               
        P_flat = P.at[constrained_indices].set(0) # Zero out the unconstrained part
        P_flat = P[jnp.tril_indices_from(dummy_P, -1)] # Take only the lower triangle below diagonal
        
        Z_flat = jnp.ravel(Z)
        X=jnp.ravel(X)
        ZPflat=jnp.concatenate([Z_flat, P_flat])
        return jnp.concatenate([X, ZPflat])

    # reconstruct_ZP MUST use self.D (the GUESS D)
    def reconstruct_ZP(self, parameter_vector):
        D_guess = self.D
        
        # X is the first element
        X = parameter_vector[0]
        
        # Z is C * D_guess elements
        Z_start = 1
        Z_end = 1 + self.C * D_guess
        Z = parameter_vector[Z_start:Z_end].reshape((self.C, D_guess))

        # P starts after X and Z
        P_start = Z_end
        
        if self.CONSTRAIN_ROTATION:
            # Number of constrained parameters in P: M*D - (D*(D+1)/2)
            # which is the length of P_flat
            
            # Recreate the M x D_guess P matrix
            P = jnp.zeros((self.M, D_guess))
            
            # The rest of the parameter vector is P_flat
            P_flat_size = self.M * D_guess - (D_guess * (D_guess + 1) // 2)
            P_flat_vec = parameter_vector[P_start:]
            
            # The indices for the lower triangle (excluding diagonal)
            P_indices = jnp.tril_indices_from(P, -1)
            
            # Assign the flat vector to the lower triangle
            P = P.at[P_indices].set(P_flat_vec)
            
            # The unconstrained upper triangle is set to the absolute value
            P = P.at[jnp.triu_indices_from(P,-1)].set(jnp.abs(P[jnp.triu_indices_from(P,-1)]))
        else:
             # P is the rest: M * D_guess elements
            P = parameter_vector[P_start:].reshape((self.M, D_guess))
            
        return Z, P, X

    def loss_function(self,parameter_vector,observed_fitness,norm,scalar_residual=True):
        Z, P, X = self.reconstruct_ZP(parameter_vector)
        # Note: landscape_obj is based on the TRUE D, but calculate_fitness uses the shape of Z/P (which are D_guess)
        predicted_fitness = self.landscape.calculate_fitness(Z, P, X)/(jnp.ravel(norm))
        observed_fitness=observed_fitness/(jnp.ravel(abs(norm)))
        loss=jaxopt.loss.huber_loss(observed_fitness, predicted_fitness)
        return loss.sum() if scalar_residual else loss.ravel()

# --- HELPER FUNCTIONS (No Change) ---
def regress_LBFGS(regression_obj, simulated_fitness, norm, Z, P, X):
    parameter_vector = regression_obj.get_parameter_vector(Z, P, X)
    solver = jaxopt.LBFGS(fun=regression_obj.loss_function, maxiter=MAX_REGRESS_ITER)
    res = solver.run(parameter_vector, observed_fitness=simulated_fitness, norm=norm)
    return res.params, res.state.value

def calculate_bic(min_loss, n_data_points, n_parameters):
    return n_parameters * jnp.log(n_data_points) + 2 * min_loss

# --- Multi-Seed/Multi-D Regression Loop ---
results_list = []

print("Starting Multi-Seed/Multi-D Regression Loop...\n")

# --- OUTER LOOP: Iterate over TRUE latent dimensions D_true ---
for D_true in D_true_values:
    print(f"==========================================")
    print(f"🧬 SIMULATING Data with TRUE Dimension: {D_true}")
    print(f"==========================================")
    
    # 1. Setup landscape and simulate 'real' data using D_true
    landscape_obj_true = landscape(C=C, D=D_true, M=M, CONSTRAIN_ROTATION=True)

    # Use a fixed key split for the simulated data for consistency across D_guess runs
    key_sim_split, key_sim_data = random.split(key_sim)
    key_sim = key_sim_split
    
    # Simulate 'real' Z and P. This is the **True** dataset.
    key_sim_data, real_Z, real_P = landscape_obj_true.simulate_dataset(key_sim_data, noise=NOISE_LEVEL)
    Simulated_fitness = landscape_obj_true.calculate_fitness(real_Z, real_P, REAL_X)

    # Generate a new Norm for the current shape
    key_norm, key_norm_split = random.split(key_sim)
    key_sim = key_norm
    Norm = abs(random.normal(key_norm_split, Simulated_fitness.shape))
    
    # --- MIDDLE LOOP: Iterate over different initial guess seeds (for robustness) ---
    for i, key_guess_init in enumerate(key_guess_list):
        seed_guess_init = seed_guess_list[i]
        key_guess_runner = key_guess_init
        
        # --- INNER LOOP: Iterate over different GUESS latent dimensions D_guess ---
        for D_guess in D_guess_values:
            
            # Split key for initial guess simulation
            key_guess_runner, key_guess_split = random.split(key_guess_runner)
            
            # 2. Setup landscape for guess (D_guess)
            # The landscape object for regression must use D_guess for the shapes of Z/P
            landscape_obj_guess = landscape(C=C, D=D_guess, M=M, CONSTRAIN_ROTATION=True)

            # 3. Generate initial guess for Z and P using D_guess
            key_guess_split, guess_Z, guess_P = landscape_obj_guess.simulate_dataset(key_guess_split)

            # 4. Setup and Run Regression
            regression_obj = RegressionProblem(
                landscape_obj_guess, 
                Simulated_fitness, 
                Norm, 
                key_sim, 
                C=C, 
                D=D_guess, # D in RegressionProblem is D_guess
                M=M, 
                CONSTRAIN_ROTATION=True
            )
            
            # Calculate number of parameters (k) for BIC
            initial_param_vector = regression_obj.get_parameter_vector(guess_Z, guess_P, jnp.array([GUESS_X]))
            n_parameters = len(initial_param_vector)

            print(f"--- D_true={D_true}, D_guess={D_guess} (k={n_parameters}), Seed={seed_guess_init} ---")
            regZP, minimized_loss = regress_LBFGS(regression_obj, Simulated_fitness, Norm, guess_Z, guess_P, jnp.array([GUESS_X]))

            # 5. Reconstruct parameters and predict fitness
            # Reconstruct Z, P, X based on the D_guess size
            rereconZ, rereconP, reX = regression_obj.reconstruct_ZP(regZP)
            
            # Calculate predicted fitness using the reconstructed Z, P, X
            predFit = landscape_obj_guess.calculate_fitness(rereconZ, rereconP, reX)

            # 6. Calculate R2 and BIC
            r2 = r2_score(Simulated_fitness.flatten(), predFit.flatten())
            bic_score = calculate_bic(minimized_loss, n, n_parameters)
            
            # 7. Store results
            results_list.append({
                'Initial_Guess_Seed': seed_guess_init,
                'True_Dimension': D_true,
                'Guess_Dimension': D_guess,
                'Predicted_R2': r2,
                'BIC_Score': bic_score,
                'Num_Parameters': n_parameters,
                'Min_Loss': minimized_loss
            })
            
            print(f"    R-squared: {r2:.4f} | BIC: {bic_score:.2f}")

# --- Final Data Structure and Plotting ---

# Convert the list of results into a pandas DataFrame
df_results = pd.DataFrame(results_list)

print("\n\n--- Summary of All Runs ---")
# Only display a snapshot, the full results list is very long
print(df_results.head(20).to_string())

## 📊 Plotting Multi-Seed Results
# The plotting logic is modified to plot R2 and BIC vs. D_guess, separated by D_true.

D_true_for_plot = D_true_values # Use the list of true dimensions for iterating plots

import pandas as pd
import jax.numpy as jnp
# Assuming df_results is already generated from your multi-loop simulation

# --- Preparation for Combined Plotting ---

# Group the results by True_Dimension and Guess_Dimension to get the mean/std
df_summary_combined = df_results.groupby(['True_Dimension', 'Guess_Dimension']).agg(
    mean_R2=('Predicted_R2', 'mean'),
    std_R2=('Predicted_R2', 'std'),
    mean_BIC=('BIC_Score', 'mean'),
    std_BIC=('BIC_Score', 'std')
).reset_index()

# Handle cases where std is NaN (i.e., only one seed was run)
df_summary_combined = df_summary_combined.fillna(0)

D_true_for_plot = sorted(df_summary_combined['True_Dimension'].unique().tolist())
D_guess_values_plot = sorted(df_summary_combined['Guess_Dimension'].unique().tolist())

plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
colors = plt.cm.plasma(jnp.linspace(0, 0.9, len(D_true_for_plot))) # Use a color map for distinct lines

# --- Plot R-squared vs. Guess Dimension ---

for idx, D_true in enumerate(D_true_for_plot):
    # Filter data for the current true dimension
    df_filtered = df_summary_combined[df_summary_combined['True_Dimension'] == D_true]
    
    # Plot R-squared with error bars
    axes[0].errorbar(
        df_filtered['Guess_Dimension'], 
        df_filtered['mean_R2'], 
        yerr=df_filtered['std_R2'], 
        marker='o', 
        linewidth=2, 
        linestyle='-', 
        color=colors[idx],
        capsize=5,
        label=r'$D_{true} = ' + str(D_true) + r'$'
    )

axes[0].set_title(r'Avg. $\mathbf{R}^2$ Score vs. Guess Dimension $\mathbf{D}_{guess}$', fontsize=16)
axes[0].set_xlabel("Guess Dimension $\mathbf{D}_{guess}$", fontsize=14)
axes[0].set_ylabel(r'Avg. $\mathbf{R}^2$ Score', fontsize=14)
axes[0].set_xticks(D_guess_values_plot)
axes[0].legend(title="True Dimension", loc='upper left')
axes[0].grid(True, linestyle='--', alpha=0.7)
axes[0].set_ylim(0, 1.05) # Better visualization for R2

# --- Plot BIC vs. Guess Dimension ---

for idx, D_true in enumerate(D_true_for_plot):
    # Filter data for the current true dimension
    df_filtered = df_summary_combined[df_summary_combined['True_Dimension'] == D_true]
    
    # Plot BIC with error bars
    axes[1].errorbar(
        df_filtered['Guess_Dimension'], 
        df_filtered['mean_BIC'], 
        yerr=df_filtered['std_BIC'], 
        marker='s', 
        linewidth=2, 
        linestyle='--', 
        color=colors[idx], # Use the same color for the same D_true
        capsize=5,
        label=r'$D_{true} = ' + str(D_true) + r'$'
    )

axes[1].set_title(r'Avg. BIC Score vs. Guess Dimension $\mathbf{D}_{guess}$', fontsize=16)
axes[1].set_xlabel("Guess Dimension $\mathbf{D}_{guess}$", fontsize=14)
axes[1].set_ylabel("Avg. BIC Score (Lower is Better)", fontsize=14)
axes[1].set_xticks(D_guess_values_plot)
axes[1].legend(title="True Dimension", loc='upper right')
axes[1].grid(True, linestyle='--', alpha=0.7)

#plt.suptitle(r'Model Selection Metrics by True Latent Dimension ($\mathbf{D}_{true}$)', fontsize=18, y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig("MtoMPlots\Synthetic_Ds.pdf", dpi=300)

plt.show()

print("\n--- Combined Plotting Complete ---")
print("You now have two plots:")
print("1. **R-squared Plot:** Shows how well the model fits (higher is better) for different D_guess, with each line representing a true dataset D_true.")
print("2. **BIC Plot:** Shows the model selection score (lower is better) for different D_guess. The minimum of each line should ideally occur at D_guess = D_true, indicating correct dimension selection.")