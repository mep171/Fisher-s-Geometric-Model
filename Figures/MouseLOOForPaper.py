# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 15:48:01 2025

@author: Meaghan Parks
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 9 15:50:52 2025

@author: Meaghan Parks
"""

# --- Imports ---
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
import numpy as np
from jax import random
import jaxopt
import pandas as pd
import seaborn as sns # Added for heatmap plotting

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
# --- Constants ---
MAX_REGRESS_ITER = 1000
seed = 111
C = 3     # smaller for demo; increase to 24 later
M = 28    # smaller for demo; increase to 1930 later
D = 2     # latent dimension


# --- Landscape Class ---
class landscape:
    def __init__(self, C=24, D=2, M=1930, scale=1, CONSTRAIN_ROTATION=True, **dimensions):
        self.C = C
        self.D = D
        self.M = M
        self.scale = scale
        self.CONSTRAIN_ROTATION = CONSTRAIN_ROTATION
        for dimension, value in dimensions.items():
            setattr(self, dimension, value)

    def simulate_dataset(self, key, noise=0):
        key, key_z = random.split(key)
        Z = random.normal(key_z, (self.C, self.D))
        key, key_p = random.split(key)
        P = random.normal(key_p, (self.M, self.D))

        if self.CONSTRAIN_ROTATION:
            P = jnp.tril(P, -1)
            P = P.at[jnp.triu_indices_from(P, -1)].set(
                jnp.abs(P[jnp.triu_indices_from(P, -1)])
            )

        return key, Z, P

    def calculate_fitness(self, Z, P, X):
        # replicate environments and mutants
        tiledZ = jnp.tile(Z, (self.M, 1))
        repedP = jnp.repeat(P, self.C, axis=0)
        repMutant = tiledZ + repedP
        Fitness = X * (jnp.exp(-jnp.einsum("cd,cd->c", repMutant, repMutant) / 2))
        return jnp.log(Fitness)


# --- RegressionProblem Class ---
class RegressionProblem:
    def __init__(
        self,
        landscape_obj,
        observed_fitnesses,
        norm,
        C,
        D,
        M,
        CONSTRAIN_ROTATION=True,
        LOG_FITNESS=True,
    ):
        self.landscape = landscape_obj
        self.LOG_FITNESS = LOG_FITNESS
        if self.LOG_FITNESS:
            self.observed_fitnesses = observed_fitnesses
        else:
            self.observed_fitnesses = jnp.log(observed_fitnesses)
        self.C = C
        self.D = D
        self.M = M
        self.CONSTRAIN_ROTATION = CONSTRAIN_ROTATION
        self.norm = norm

    def get_parameter_vector(self, Z, P, X):
        Z_flat = jnp.ravel(Z)
        if self.CONSTRAIN_ROTATION:
            indices = [(i, j) for i in range(self.M) for j in range(self.D) if i > j]
            P_flat = jnp.array([P[i, j] for i, j in indices])
        else:
            P_flat = jnp.ravel(P)
        ZPflat = jnp.concatenate([Z_flat, P_flat])
        X = jnp.ravel(X)
        return jnp.concatenate([X, ZPflat])

    def reconstruct_ZP(self, parameter_vector):
        k_Z = self.C * self.D
        X = parameter_vector[0]
        Z = parameter_vector[1 : 1 + k_Z].reshape((self.C, self.D))
        P_flat = parameter_vector[1 + k_Z :]
        if self.CONSTRAIN_ROTATION:
            P = jnp.zeros((self.M, self.D))
            indices = [(i, j) for i in range(self.M) for j in range(self.D) if i > j]
            for idx, (i, j) in enumerate(indices):
                P = P.at[i, j].set(P_flat[idx])
        else:
            P = P_flat.reshape((self.M, self.D))
        return Z, P, X

    def loss_function(self, parameter_vector, observed_fitness):
        Z, P, X = self.reconstruct_ZP(parameter_vector)
        predicted_fitness = self.landscape.calculate_fitness(Z, P, X)

        # Flatten arrays
        observed_fitness = jnp.ravel(observed_fitness)
        predicted_fitness = jnp.ravel(predicted_fitness)
        norm = jnp.ravel(self.norm)

        # Handle mismatched shapes (for leave-one-out subsets)
        n_obs = observed_fitness.shape[0]
        if norm.shape[0] != n_obs:
            norm = norm[:n_obs]
        if predicted_fitness.shape[0] != n_obs:
            predicted_fitness = predicted_fitness[:n_obs]

        loss = jaxopt.loss.huber_loss(
            observed_fitness / norm, predicted_fitness / norm
        )
        return loss.sum()


# --- Regression with L-BFGS ---
def regress_LBFGS(regression_obj, simulated_fitness, Z, P, X):
    parameter_vector = regression_obj.get_parameter_vector(Z, P, X)
    solver = jaxopt.LBFGS(fun=regression_obj.loss_function, maxiter=MAX_REGRESS_ITER)
    res = solver.run(parameter_vector, observed_fitness=simulated_fitness)
    return res.params


# --- Leave-One-Out Function ---
def leave_one_out(regression_obj, observed_fitness, Z_init, P_init, X_init):
    y = jnp.ravel(observed_fitness)
    n = y.shape[0]
    predictions = []
    errors = []

    for i in range(n):
        mask = jnp.ones(n, dtype=bool).at[i].set(False)
        y_train = y[mask]
        norm_train = regression_obj.norm.ravel()[mask]

        # create temporary regression object with subset norm
        temp_reg_obj = RegressionProblem(
            regression_obj.landscape,
            y_train,
            norm_train,
            regression_obj.C,
            regression_obj.D,
            regression_obj.M,
            regression_obj.CONSTRAIN_ROTATION,
        )

        # Fit model
        res_params = regress_LBFGS(temp_reg_obj, y_train, Z_init, P_init, X_init)

        # Predict left-out value
        Z_hat, P_hat, X_hat = regression_obj.reconstruct_ZP(res_params)
        y_pred_all = regression_obj.landscape.calculate_fitness(Z_hat, P_hat, X_hat)
        y_pred = jnp.ravel(y_pred_all)[i]

        predictions.append(y_pred)
        errors.append(y_pred - y[i])

        print(f"LOO {i+1}/{n} complete. Observed={y[i]:.4f}, Predicted={y_pred:.4f}")

    predictions = jnp.array(predictions)
    errors = jnp.array(errors)
    return predictions, errors


# --- Data Loading (NOTE: Files must exist in the specified path for this to run) ---
# Assuming these files are in your 'Documents\McFarlandLabProjects' folder
try:
    MiceG12C=pd.read_csv(r"Figure5A.csv")
    MiceG12D=pd.read_csv(r"Figure5B.csv")
    MiceEGFR=pd.read_csv(r"Figure5D.csv")
except FileNotFoundError as e:
    print(f"Error: Required data file not found. Please ensure the path is correct: {e}")
    # Exit or use dummy data if necessary
    exit()

mergeMiceG12=pd.merge(MiceG12C,MiceG12D,on='gene',how='inner')
AllMice=pd.merge(mergeMiceG12,MiceEGFR,on="gene",how='inner')

# Extract fitness values and normalize them (using CI range)
AllMiceVals=AllMice.loc[:, ['tumor_enrichment_x', 'tumor_enrichment_y','tumor_enrichment']]
AllMiceCI=np.log(AllMice.loc[:,['CI_lower_x','CI_upper_x','CI_lower_y','CI_upper_y','CI_lower','CI_upper' ]])
AllMiceCI['sumCI_x']=(AllMiceCI['CI_upper_x']-AllMiceCI['CI_lower_x'])
AllMiceCI['sumCI_y']=(AllMiceCI['CI_upper_y']-AllMiceCI['CI_lower_y'])
AllMiceCI['sumCI']=(AllMiceCI['CI_upper']-AllMiceCI['CI_lower'])
ALLMiceNorm=AllMiceCI.loc[:,['sumCI_x','sumCI_y','sumCI']]
ALLMiceNormNP=ALLMiceNorm.to_numpy()

# Prepare fitness data for JAX
mice_fitness=jnp.transpose(AllMiceVals.to_numpy())
mice_fitness=jnp.log(mice_fitness)

# Get labels for plotting
MutationLabels=MiceG12C.loc[:,"gene"].to_list()
ConditionLabels = ["G12C", "G12D", "EGFR;p53"]
M = len(MutationLabels) # Recalculate M based on loaded data
C = len(ConditionLabels) # Recalculate C based on defined labels
# --- End of Data Loading ---


# --- Residual Heatmap Plotting Function ---

def plot_LOO_residual_heatmap(preds, observed_fitness, C, M,
                             gene_labels=None, condition_labels=None,
                             title="LOO Residual Heatmap (Predicted - Observed)"):
    """
    Plot the residual heatmap (Predicted - Observed) from LOO cross-validation.
    """
    # --- reshape into C x M grids and calculate residuals ---
    obs_mat = np.array(observed_fitness).reshape(C, M)
    pred_mat = np.array(preds).reshape(C, M)
    resid_mat = pred_mat - obs_mat

    # --- figure setup ---
    plt.figure(figsize=(10, 6)) # Adjust figure size as needed

    # Default axis labels if not provided
    x_labels = gene_labels if gene_labels is not None else np.arange(M)
    y_labels = condition_labels if condition_labels is not None else np.arange(1, C + 1)

    # --- Residuals Heatmap ---
    sns.heatmap(
        resid_mat,
        cmap="coolwarm", # Use a divergent color map centered at 0
        center=0,
        cbar=True,
        xticklabels=x_labels,
        yticklabels=y_labels
    )
    
    plt.title("Leave-One-Out Residual Heatmap (Predicted - Observed)")
    plt.xlabel("Gene (Mutant)")
    plt.ylabel("Mouse Model")

    # Improve readability for many genes
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig("MtoMPlots\ResLOOMouse.pdf", dpi=300)
    plt.show()

# --- Main Execution ---
if __name__ == "__main__":
    key = random.PRNGKey(seed)
    # Ensure C and M are set to the actual data dimensions
    land = landscape(C=C, D=D, M=M) 
    key, Z_init, P_init = land.simulate_dataset(key)
    X_init = jnp.array(1.0)

    observed_fitness = mice_fitness
    
    reg_obj = RegressionProblem(
        land,
        observed_fitness,
        norm=ALLMiceNormNP,
        C=C,
        D=D,
        M=M,
    )

    print("\n--- Running Leave-One-Out Cross Validation ---")
    # This step performs the computationally intensive LOO calculation
    preds, errs = leave_one_out(reg_obj, observed_fitness, Z_init, P_init, X_init)

    print("\nFinished Leave-One-Out!\n")
    print("Mean Absolute Error:", jnp.mean(jnp.abs(errs)))

    # --- Plotting ONLY the Residual Heatmap ---
    print("\n--- Plotting Residual Heatmap ---")
    plot_LOO_residual_heatmap(
        preds,
        observed_fitness,
        C=C,
        M=M,
        gene_labels=MutationLabels,
        condition_labels=ConditionLabels
)
plt.savefig(r"MtoMPlots.pdf", dpi=300)
plt.show

