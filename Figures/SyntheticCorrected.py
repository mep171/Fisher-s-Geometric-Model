# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 12:06:20 2026

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
    return res.params

landscape_obj=landscape()

key2, real_Z, real_P=landscape_obj.simulate_dataset(key2)
real_X=25
Simulated_fitness=landscape_obj.calculate_fitness(real_Z, real_P, real_X, noise=0.005)

key, guess_Z, guess_P=landscape_obj.simulate_dataset(key)
guess_X=1

key, key_fitness = random.split(key)
Norm = abs(random.normal(key_fitness, Simulated_fitness.shape))

regression_obj=RegressionProblem(landscape_obj, Simulated_fitness,Norm)

# 5. Initialize Parameters

regZP=regress_LBFGS(regression_obj, landscape_obj, Simulated_fitness, Norm, guess_Z, guess_P, guess_X)

rereconZ,rereconP,reX=regression_obj.reconstruct_ZP(regZP,2)
predFit=landscape_obj.calculate_fitness(rereconZ, rereconP,reX,key2)

from sklearn.metrics import r2_score

r2 = r2_score(Simulated_fitness, predFit)
from sklearn.metrics import mean_absolute_error
MAE = mean_absolute_error(Simulated_fitness, predFit)
import scipy.stats as stats

correlation_coefficient, p_value = stats.pearsonr(Simulated_fitness, predFit)

r2_score(Simulated_fitness.flatten(),predFit)
plt.scatter(jnp.ravel(Simulated_fitness),predFit,c=Norm,cmap="summer",marker="o",alpha=.5)
plt.title("Regressed vs. Measured Synthetic Fitness")
plt.ylabel("Log Regressed Fitness")
plt.xlabel("Log Observed Synthetic Fitness")
plt.text(-4, -6, "MAE= 0.09387", fontsize=12)
plt.text(-4, -6.8, "PCC = 0.9716, p < .00001  ", fontsize=12)
plt.colorbar(label='Synthetically Generated Uncertainty')
plt.plot((-8,3),(-8,3),color="black")
plt.savefig(r"SyntheticFitness.pdf", dpi=300)
plt.show()


# --- Setup for the loop ---
D_values = [1, 2, 3, 4] # Dimensions to test
results = {} # Dictionary to store R2 score for each D

# Constants C and M from your original setup
C = 20
M = 20
key_sim = random.PRNGKey(seed2) # Key for real data simulation
key_guess = random.PRNGKey(seed) # Key for initial guess

print("Starting LBFGS Regression Loop...\n")

# --- Loop over different dimensions D ---
for D_current in D_values:
    # 1. Update keys for determinism and split
    key_sim, key_sim_split = random.split(key_sim)
    key_guess, key_guess_split = random.split(key_guess)
    
    # 2. Setup landscape and simulate 'real' data
    # Create a new landscape object with the current D
    landscape_obj = landscape(C=C, D=D_current, M=M, CONSTRAIN_ROTATION=True)

    # Simulate 'real' Z and P
    key_sim_split, real_Z, real_P = landscape_obj.simulate_dataset(key_sim_split)
    real_X = 25
    Simulated_fitness = landscape_obj.calculate_fitness(real_Z, real_P, real_X, noise=0.005)

    # Generate a new Norm for the current shape
    key_norm, key_norm_split = random.split(key_sim_split)
    Norm = abs(random.normal(key_norm_split, Simulated_fitness.shape))

    # 3. Generate initial guess for Z and P
    # Create new guess Z and P matrices with the current D
    key_guess_split, guess_Z, guess_P = landscape_obj.simulate_dataset(key_guess_split)
    guess_X = 1

    # 4. Setup and Run Regression
    regression_obj = RegressionProblem(landscape_obj, Simulated_fitness, Norm, key_sim, C=C, D=D_current, M=M, CONSTRAIN_ROTATION=True)
    
    print(f"--- Running Regression for D = {D_current} ---")
    regZP = regress_LBFGS(regression_obj, landscape_obj, Simulated_fitness, Norm, guess_Z, guess_P, guess_X)

    # 5. Reconstruct parameters and predict fitness
    # Note: D_current is passed to reconstruct_ZP
    rereconZ, rereconP, reX = regression_obj.reconstruct_ZP(regZP, D_current)
    predFit = landscape_obj.calculate_fitness(rereconZ, rereconP, reX)

    # 6. Calculate R2 score and store the result
    r2 = r2_score(Simulated_fitness.flatten(), predFit.flatten())
    results[D_current] = r2
    
    print(f"R-squared for D = {D_current}: {r2:.4f}\n")

# --- Plot and Final Output ---

print("\n--- Summary of Results ---")
for D, r2 in results.items():
    print(f"Dimension D={D}: R2 Score = {r2:.4f}")

# Optional: Visualize results (e.g., plot D vs R2)
plt.figure(figsize=(8, 5))
plt.plot(list(results.keys()), list(results.values()), marker='o',linestyle='-', color='b')
plt.title(r'$\mathbf{R}^2$ Score vs. Latent Dimension $\mathbf{D}$')
plt.xlabel("Latent Dimension D")
plt.ylabel(r'$\mathbf{R}^2$ Score')
plt.grid(True)
plt.xticks(D_values)
plt.show()


# --- BIC Calculation Function ---
def calculate_bic(min_loss, n_data_points, n_parameters):
    # n_data_points (n): C * M = 400
    # n_parameters (k): 1 + C*D + len(P_flat)
    
    # We use the approximation BIC = k * ln(n) + 2 * L_min
    # where L_min is the minimized sum of Huber losses.
    return n_parameters * jnp.log(n_data_points) + 2 * min_loss


# --- Setup and Run Loop (Re-running the loop to capture L_min) ---
D_values = [1, 2, 3, 4]
results = {} # To store (R2, BIC) for each D
C = 20
M = 20
n = C * M # Total number of data points
key_sim = random.PRNGKey(seed2)
key_guess = random.PRNGKey(seed)

print("Starting LBFGS Regression Loop for BIC calculation...\n")

for D_current in D_values:
    # Key management and landscape setup (as before)
    key_sim, key_sim_split = random.split(key_sim)
    key_guess, key_guess_split = random.split(key_guess)
    landscape_obj = landscape(C=C, D=D_current, M=M, CONSTRAIN_ROTATION=True)

    # Simulate 'real' data
    key_sim_split, real_Z, real_P = landscape_obj.simulate_dataset(key_sim_split)
    real_X = 25
    Simulated_fitness = landscape_obj.calculate_fitness(real_Z, real_P, real_X, noise=0.005)
    key_norm, key_norm_split = random.split(key_sim_split)
    Norm = abs(random.normal(key_norm_split, Simulated_fitness.shape))

    # Generate initial guess and setup regression
    key_guess_split, guess_Z, guess_P = landscape_obj.simulate_dataset(key_guess_split)
    guess_X = 1
    regression_obj = RegressionProblem(landscape_obj, Simulated_fitness, Norm, key_sim, C=C, D=D_current, M=M, CONSTRAIN_ROTATION=True)

    # --- Run LBFGS and capture results object ---
    parameter_vector_guess = regression_obj.get_parameter_vector(guess_Z, guess_P, guess_X)
    solver = jaxopt.LBFGS(fun=regression_obj.loss_function, maxiter=MAX_REGRESS_ITER)
    # The solver run is needed to get res.fun_val
    res = solver.run(parameter_vector_guess, observed_fitness=Simulated_fitness, norm=Norm)
    
    # --- Capture the minimized loss and parameter count ---
    minimized_loss = res.state.value
    n_parameters = len(res.params) # k is the length of the final parameter vector
    
    # --- Calculate BIC ---
    bic_score = calculate_bic(minimized_loss, n, n_parameters)

    # --- R2 calculation (as before) ---
    rereconZ, rereconP, reX = regression_obj.reconstruct_ZP(res.params, D_current)
    predFit = landscape_obj.calculate_fitness(rereconZ, rereconP, reX)
    r2 = r2_score(Simulated_fitness.flatten(), predFit.flatten())
    
    # Store results
    results[D_current] = {'R2': r2, 'BIC': bic_score}
    
    print(f"--- Results for D = {D_current} ---")
    print(f"R-squared: {r2:.4f}")
    print(f"Parameter Count (k): {n_parameters}")
    print(f"Minimized Loss (L_min): {minimized_loss:.2f}")
    print(f"BIC Score: {bic_score:.2f}\n")


# --- Final Output ---
print("\n--- Summary of Model Selection Scores ---")
for D, scores in results.items():
    print(f"Dimension D={D}: R2 = {scores['R2']:.4f}, BIC = {scores['BIC']:.2f}")
    
# The model with the **lowest BIC** is generally preferred.
# The R2 score should be high, but BIC helps select the simplest model.

# Assuming the previous loop has been run and the 'results' dictionary is populated
# with {'R2': ..., 'BIC': ...} for each D.

# Extract D values, R2 scores, and BIC scores for plotting
D_values_plot = list(results.keys())
r2_scores_plot = [scores['R2'] for scores in results.values()]
bic_scores_plot = [scores['BIC'] for scores in results.values()]

# Create a figure with two subplots
plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(1, 2, figsize=(14, 6)) # 1 row, 2 columns

# Plot R-squared
axes[0].plot(D_values_plot, r2_scores_plot, marker='o', linewidth=2, linestyle='-', color='darkcyan')
axes[0].set_title(r'$\mathbf{R}^2$ Score vs. Latent Dimension $\mathbf{D}$',fontsize=16)
axes[0].set_xlabel("Latent Dimension D",fontsize=14)
axes[0].set_ylabel(r'$\mathbf{R}^2$ Score',fontsize=14)
axes[0].set_xticks(D_values_plot) # Ensure all D values are shown as ticks
axes[0].grid(True, linestyle='--', alpha=0.7)

# Plot BIC
axes[1].plot(D_values_plot, bic_scores_plot, marker='s', linewidth=2, linestyle='--', color='seagreen')
axes[1].set_title(r'BIC Score vs. Latent Dimension $\mathbf{D}$',fontsize=16)
axes[1].set_xlabel("Latent Dimension D",fontsize=14)
axes[1].set_ylabel("BIC Score (Lower is Better)",fontsize=14)
axes[1].set_xticks(D_values_plot) # Ensure all D values are shown as ticks
axes[1].grid(True, linestyle='--', alpha=0.7)

# Adjust layout to prevent overlapping titles/labels
plt.tight_layout()

# Display the plots
plt.show()

print("\n--- Plotting Complete ---")
print("R-squared generally increases with model complexity (D).")
print("BIC penalizes complexity; the optimal D is often where BIC is minimized.")




