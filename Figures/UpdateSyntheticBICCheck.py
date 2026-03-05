# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:53:09 2026

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

plt.rcParams.update({'font.size': 12})

# Set the default font family (e.g., to a common serif or sans-serif)
# 'serif' often looks good in papers. Use 'sans-serif' for a cleaner look.
plt.rcParams.update({'font.family': 'sans-serif'})

# Settings for axes labels
plt.rcParams.update({
    'axes.labelsize': 14,      # Font size of the x and y labels
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
MAX_REGRESS_ITER = 50000
#CONSTRAIN_ROTATION=True
#D=2
#C=13
#M=18
seed = 980
seed2 = 90
key = random.PRNGKey(seed)
key2 = random.PRNGKey(seed2)
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
        tiledZ = jnp.tile(Z, (landscape_obj.M,1))
        repedP = jnp.repeat(P,landscape_obj.C,axis=0)
        repMutant = tiledZ+ repedP
        Fitness = X*((jnp.exp( -jnp.einsum('cd,cd->c', repMutant, repMutant)/2)))
        #(jnp.exp( -jnp.einsum('cmd,cmd->mc', Mutants_cdm, Mutants_cdm)/2)))
        #assert (Fitness <=0).all().all()
        return jnp.log(Fitness)

class RegressionProblem:
    def __init__(self, landscape_obj, observed_fitnesses,norm,C=20, D=2, M=20, CONSTRAIN_ROTATION=True,LOG_FITNESS=True):
        self.landscape = landscape_obj
        self.LOG_FITNESS=LOG_FITNESS
        if self.LOG_FITNESS:
            self.observed_fitnesses = observed_fitnesses
        else:
            self.observed_fitnesses = jnp.log(observed_fitnesses)
        self.C = C
        self.D = D
        self.M = M
        self.CONSTRAIN_ROTATION=CONSTRAIN_ROTATION
    
    def check_determined(self,Z,P):
        # https://en.wikipedia.org/wiki/Underdetermined_system
        observations=len(self.observed_fitnesses)
        free_parameters=len(self.get_parameter_vector(Z,P))
        print("Under-determined" if observations<free_parameters else "Over-Determined")
    
    def get_NA_location(self):
        return jnp.argwhere(jnp.isnan(self.observed_fitnesses))
    
    def replace_NA(self):
        observed_fitnesses_no_NA=self.observed_fitnesses.at[self.get_NA_location()].set(0)
        return observed_fitnesses_no_NA
    
    def get_parameter_vector(self, Z, P, X):
        P_flat=P[jnp.tril_indices_from(P,-1)]
        Z_flat = jnp.ravel(Z)
        X=jnp.ravel(X)
        ZPflat=jnp.concatenate([Z_flat, P_flat]) 
        return jnp.concatenate([X,ZPflat])     

    def reconstruct_ZP(self, parameter_vector,D):
       P=parameter_vector[-(self.M * self.D):].reshape((self.M, self.D))
       Z = parameter_vector[1:self.C*self.D+1].reshape((self.C, self.D))
       X = parameter_vector[0]
       if self.CONSTRAIN_ROTATION:
           P=jnp.zeros((self.M,self.D))
           P=P.at[jnp.tril_indices_from(P,-1)].set(parameter_vector[self.C*self.D+1:])
           P = jnp.tril(P, -1)
           P = P.at[jnp.triu_indices_from(P,-1)].set(jnp.abs(P[jnp.triu_indices_from(P,-1)]))
       return Z, P, X

    def loss_function(self, parameter_vector, observed_fitness, norm, scalar_residual=True):
        Z, P, X = self.reconstruct_ZP(parameter_vector, self.D)
        predicted_fitness = self.landscape.calculate_fitness(Z, P, X)
    
    # Flatten both for element-wise weighting
        pred_flat = jnp.ravel(predicted_fitness)
        obs_flat = jnp.ravel(observed_fitness)
        weight_flat = jnp.ravel(norm)
    
    # Residuals weighted by the inverse of the log-error (delta z)
    # This ensures that points with large relative errors [cite: 143] 
    # contribute less to the total loss.
        weighted_residuals = (obs_flat - pred_flat) / weight_flat
    
    # Using Huber loss on the weighted residuals
        loss = jaxopt.loss.huber_loss(weighted_residuals, jnp.zeros_like(weighted_residuals))
    
        return loss.sum() if scalar_residual else loss

def regress_LBFGS(regression_obj, landscape_obj,simulated_fitness,norm,Z,P,X):
    parameter_vector = regression_obj.get_parameter_vector(Z,P,X)
    solver = jaxopt.LBFGS(fun=regression_obj.loss_function, maxiter=MAX_REGRESS_ITER)
    res = solver.run(parameter_vector, observed_fitness=simulated_fitness,norm=norm)
    return res.params, res.state.value

landscape_obj=landscape()

key2, real_Z, real_P=landscape_obj.simulate_dataset(key2)
real_X=25
Simulated_fitness=landscape_obj.calculate_fitness(real_Z, real_P, real_X, noise=0.005)

key, guess_Z, guess_P=landscape_obj.simulate_dataset(key)
guess_X=1

key, key_fitness = random.split(key)
Norm = abs(random.normal(key_fitness, Simulated_fitness.shape))

regression_obj=RegressionProblem(landscape_obj, Simulated_fitness,Norm)
C=20
M=20
xcount=1
def calculate_bic(min_loss, n_data_points, n_parameters):
    return n_parameters * jnp.log(n_data_points) + 2 * min_loss

results_list = []
n_samples = C * M # Total observations
# The true dimension of the simulated data
D_true_values = [1, 2, 3, 4, 5]
# The guess dimension used for the regression (D_guess)
D_guess_values = [1,2,3,4,5]

# --- Multiple Seed Setup ---
# Key for generating the 'true' simulated data (fixed)
seed_sim = 90
key_sim = random.PRNGKey(seed_sim)

# Keys for the initial 'guess' parameters (variable)
seed_guess_list = [980]
# Note: Use one key per run to ensure independent initial guesses
key_guess_list = [random.PRNGKey(s) for s in seed_guess_list]
C = 20 # Number of contexts
M = 20 # Number of mutations
n = C * M # Total number of data points
REAL_X = 25
GUESS_X = 1
NOISE_LEVEL = 0.05


print("Starting Multi-Seed/Multi-D Regression Loop...\n")

for D_true in D_true_values:
    # 1. Setup True Landscape
    landscape_obj_true = landscape(C=C, D=D_true, M=M, CONSTRAIN_ROTATION=True)
    
    key_sim, key_sim_data = random.split(key_sim)
    _, real_Z, real_P = landscape_obj_true.simulate_dataset(key_sim_data, noise=NOISE_LEVEL)
    
    # Calculate True Fitness
    Simulated_fitness = landscape_obj_true.calculate_fitness(real_Z, real_P, REAL_X)
    
    # Generate Noise Norm
    key_sim, key_norm = random.split(key_sim)
    Norm = abs(random.normal(key_norm, Simulated_fitness.shape)) + 1e-6

    for i, key_guess_init in enumerate(key_guess_list):
        seed_guess_init = seed_guess_list[i]
        
        for D_guess in D_guess_values:
            # 2. Setup Guess Landscape
            landscape_obj_guess = landscape(C=C, D=D_guess, M=M, CONSTRAIN_ROTATION=True)
            
            # 3. Generate initial guess
            key_guess_init, key_sub = random.split(key_guess_init)
            _, guess_Z, guess_P = landscape_obj_guess.simulate_dataset(key_sub)
            
            # 4. Run Regression
            regression_obj = RegressionProblem(
                landscape_obj_guess, Simulated_fitness, Norm, 
                C=C, D=D_guess, M=M, CONSTRAIN_ROTATION=True
            )
            
            # Get parameter count for BIC
            initial_params = regression_obj.get_parameter_vector(guess_Z, guess_P, GUESS_X)
            n_parameters = len(initial_params)

            reg_params, minimized_loss = regress_LBFGS(
                regression_obj,landscape_obj, Simulated_fitness, Norm, guess_Z, guess_P, GUESS_X
            )

            # 5. Reconstruction & Metrics
            rereconZ, rereconP, reX = regression_obj.reconstruct_ZP(reg_params,D_guess)
            predFit = landscape_obj_guess.calculate_fitness(rereconZ, rereconP, reX)

            # Simple R2 calculation
            ss_res = jnp.sum((Simulated_fitness - predFit) ** 2)
            ss_tot = jnp.sum((Simulated_fitness - jnp.mean(Simulated_fitness)) ** 2)
            r2 = 1 - (ss_res / ss_tot)
            
            bic_score = calculate_bic(minimized_loss, n_samples, (n_parameters-((D_guess*(D_guess+1))/2)))

            results_list.append({
                'Initial_Guess_Seed': seed_guess_init,
                'True_Dimension': D_true,
                'Guess_Dimension': D_guess,
                'Predicted_R2': float(r2),
                'BIC_Score': float(bic_score),
                'Num_Parameters': n_parameters,
                'Min_Loss': float(minimized_loss)
            })
            
            print(f"D_true:{D_true} | D_guess:{D_guess} | R2:{r2:.3f} | BIC:{bic_score:.1f}")

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

axes[0].set_title(r' $\mathbf{R}^2$ Score vs. Model Dimension $\mathbf{D}_{model}$', fontsize=16)
axes[0].set_xlabel("Model Dimension $\mathbf{D}_{model}$", fontsize=14)
axes[0].set_ylabel(r'Avg. $\mathbf{R}^2$ Score', fontsize=14)
axes[0].set_xticks(D_guess_values_plot)
axes[0].legend(title="True Dimension", loc='upper left')
axes[0].grid(True, linestyle='--', alpha=0.7)
#axes[0].set_ylim(0, 1.05) # Better visualization for R2

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

axes[1].set_title(r' BIC Score vs. Model Dimension $\mathbf{D}_{model}$', fontsize=16)
axes[1].set_xlabel("Model Dimension $\mathbf{D}_{model}$", fontsize=14)
axes[1].set_ylabel("BIC Score (Lower is Better)", fontsize=14)
axes[1].set_xticks(D_guess_values_plot)
axes[1].legend(title="True Dimension", loc='upper right')
axes[1].grid(True, linestyle='--', alpha=0.7)

#plt.suptitle(r'Model Selection Metrics by True Latent Dimension ($\mathbf{D}_{true}$)', fontsize=18, y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig("Synthetic_DCI.pdf", dpi=300)

plt.show()
