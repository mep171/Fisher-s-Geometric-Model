# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 12:47:07 2026

@author: Meaghan Parks
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
import matplotlib.colors as mcolors
import numpy as np
from jax import random
from jax.scipy.optimize import minimize
import math
import jax.numpy as jnp
import jaxopt 
import pandas as pd
import matplotlib.pyplot as plt
MAX_REGRESS_ITER = 50000
#CONSTRAIN_ROTATION=True
#D=2
#C=13
#M=18
seed = 302

key = random.PRNGKey(seed)
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
class landscape:
    def __init__(self, C=3, D=2, M=28, scale=1, CONSTRAIN_ROTATION=True, **dimensions):
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
    
    def calculate_fitness(self, Z, P, X):
        tiledZ = jnp.tile(Z, (landscape_obj.M,1))
        repedP = jnp.repeat(P,landscape_obj.C,axis=0)
        repMutant = tiledZ+ repedP
        Fitness = X*((jnp.exp( -jnp.einsum('cd,cd->c', repMutant, repMutant)/2)))
        #(jnp.exp( -jnp.einsum('cmd,cmd->mc', Mutants_cdm, Mutants_cdm)/2)))
        #assert (Fitness <=0).all().all()
        return jnp.log(Fitness)

class RegressionProblem:
    def __init__(self, landscape_obj, observed_fitnesses,norm,C=3, D=2, M=28, CONSTRAIN_ROTATION=True,LOG_FITNESS=True):
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


MiceG12C=pd.read_csv(r"C:\Users\Meaghan Parks\Documents\McFarlandLabProjects\Figure5A.csv")
MiceG12D=pd.read_csv(r"C:\Users\Meaghan Parks\Documents\McFarlandLabProjects\Figure5B.csv")
MiceEGFR=pd.read_csv(r"C:\Users\Meaghan Parks\Documents\McFarlandLabProjects\Figure5D.csv")


mergeMiceG12=pd.merge(MiceG12C,MiceG12D,on='gene',how='inner')
AllMice=pd.merge(mergeMiceG12,MiceEGFR,on="gene",how='inner')

AllMiceVals=AllMice.loc[:, ['tumor_enrichment_x', 'tumor_enrichment_y','tumor_enrichment']]
mice_fitness = jnp.log(jnp.transpose(AllMiceVals.to_numpy()))

# 2. Calculate the log-error (delta z) 
# Based on Source, delta_z = 0.434 * (delta_y / y)
# If your CI columns are already absolute errors in y-space:
dy_x = (AllMice['CI_upper_x'] - AllMice['CI_lower_x']) / 2
dy_y = (AllMice['CI_upper_y'] - AllMice['CI_lower_y']) / 2
dy_egfr = (AllMice['CI_upper'] - AllMice['CI_lower']) / 2

# 2. Calculate Log Error (delta z) using the relative error formula from the PDF
# Formula: delta_z ≈ 0.434 * (delta_y / y) 
rel_error_x = (dy_x / AllMice['tumor_enrichment_x'])
rel_error_y =  (dy_y / AllMice['tumor_enrichment_y'])
rel_error_egfr =  (dy_egfr / AllMice['tumor_enrichment'])

# 3. Convert to arrays and stack to avoid the TypeError
# .values converts the pandas Series into a format JAX can use
ALLMiceNormNP = jnp.stack([
    rel_error_x.values, 
    rel_error_y.values, 
    rel_error_egfr.values
])
key, Z, P = landscape_obj.simulate_dataset(key)
X=jnp.array(1)

mice_fitness=jnp.transpose(AllMiceVals.to_numpy())
mice_fitness=jnp.log(mice_fitness)
regression_obj=RegressionProblem(landscape_obj, mice_fitness,ALLMiceNormNP)


MiceZPSeed302=pd.read_csv(r"C:\Users\Meaghan Parks\Documents\McFarlandLabProjects\regressedZPBlairMouse302WithCorrectCI.csv").to_numpy()
rereconZ,rereconP,reX=regression_obj.reconstruct_ZP(jnp.ravel(MiceZPSeed302),2)


### --- PLOT RECONSTRUCTED LANDSCAPE (correct fitness) -----------------------

import numpy as np

# Convert JAX arrays to numpy
Z_np = np.array(rereconZ)
P_np = np.array(rereconP)

# Compute mutant positions (Z_i + P_j)
mutants = []
for i in range(Z_np.shape[0]):      # C clones
    for j in range(P_np.shape[0]):  # M mutations
        mutants.append(Z_np[i] + P_np[j])
mutants = np.array(mutants)

# Define plotting region
xmin = min(mutants[:,0].min(), -4)
xmax = max(mutants[:,0].max(),  4)
ymin = min(mutants[:,1].min(), -4)
ymax = max(mutants[:,1].max(),  4)

#xx, yy = np.meshgrid(
#    np.linspace(xmin, xmax, 200),
#    np.linspace(ymin, ymax, 200)
#)

# Compute fitness on grid: F = X * exp( - ((x+y)^2) / 2 )
#xy_sum = xx + yy
#grid_fit = np.exp(-(xy_sum**2)/2) * float(reX)
def get_log_fitness(coords, X_scale=reX):
    dot_product = np.sum(coords**2, axis=-1)
    return np.log(X_scale) - (dot_product / 2.0)

# 2. Setup the grid
x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)
Z = get_log_fitness(np.stack([X, Y], axis=-1))

# 3. Create the plot
plt.figure(figsize=(8, 6))

# Filled contours represent the "elevation"
#cp = plt.contourf(X, Y, Z, levels=20, cmap='viridis')
#plt.colorbar(cp, label='Log-Fitness')

### --- Plotting -------------------------------------------------------------
plt.rcParams.update({'font.size': 12}) # Default font size increased from ~10 to 12
plt.figure(figsize=(14, 9)) 


# --- Gene Names and Colors ---
# Extract unique gene names
#gene_names = AllMice['gene'].unique() 
# Define the color palette used for the M=28 genes
#gene_colors = [
 #   "#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
 #   "#882255", "#44AA99", "#117733", "#332288", "#DDCC77", "#999933", "#AA4499",
 #   "#661100", "#6699CC", "#88CCEE", "#AA4466", "#44AA66", "#1177AA", "#882288",
 #   "#88AADD", "#DD7788", "#77AADD", "#44BB99", "#BB5566", "#008080"
#]
#assert len(gene_names) == len(gene_colors), "Number of genes and colors do not match"


# Contour plot
#cont = plt.contourf(xx, yy, jnp.log(grid_fit), levels=40, cmap="viridis")
#plt.colorbar(cont, label="Fitness (Log Scale)")
vmin, vmax, vcenter = -8, 5.5, 0
norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

cont = plt.contourf(X, Y, Z, levels=20, cmap='PuOr',norm=norm)
plt.colorbar(cont, label='Log-Fitness')

# Plot Z+P mutants
g12c_scatter = plt.scatter(mutants[0:28,0], mutants[0:28,1], color="royalblue",alpha=.5, marker="8", s=70)
g12d_scatter = plt.scatter(mutants[28:56,0], mutants[28:56,1], color="brown",alpha=.5, marker="^", s=70)
egfr_scatter = plt.scatter(mutants[56:,0], mutants[56:,1], color="darkcyan", alpha=.5, marker="s", s=70)

# Plot clone centers Z
#z_g12c = plt.scatter(Z_np[0,0], Z_np[0,1], color=("slateblue"), edgecolor="black",
 #            s=120, marker="8")
#z_g12d = plt.scatter(Z_np[1,0], Z_np[1,1], color=("slateblue"), edgecolor="black",
         #    s=120, marker="^")
#z_egfr = plt.scatter(Z_np[2,0], Z_np[2,1], color=("slateblue"), edgecolor="black",
 #            s=120, marker="s")


# --- Custom Legend Creation ---

# 1. Create Handles and Labels for Clone/Mutant Groups (Shape Legend)
group_handles = [
    plt.Line2D([0], [0], marker='8', color='w', markerfacecolor='royalblue', markersize=10),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='brown', markersize=10),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='darkcyan', markersize=10)
]
group_labels = ["G12C Mutants", "G12D Mutants", "EGFR Mutants"]

# 2. Create Handles and Labels for Clone Centers (Center Legend)
# Use the correct marker strings ('8', '^', 's')
#center_handles = [
#    plt.Line2D([0], [0], marker='8', color='w', markerfacecolor='slateblue', markeredgecolor='black', markersize=10),
#    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='slateblue', markeredgecolor='black', markersize=10),
#    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='slateblue', markeredgecolor='black', markersize=10)
#]
#center_labels = ["Clone Center Z (G12C)", "Clone Center Z (G12D)", "Clone Center Z (EGFR)"]

# 3. Create Handles and Labels for Genes (Color Legend) - Split into two columns
# (This section is fine and unchanged)
#half = len(gene_names) // 2
#gene_handles_col1 = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8) 
 #                    for color in gene_colors[:half]]
#gene_labels_col1 = list(gene_names[:half])

#gene_handles_col2 = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8) 
 #                    for color in gene_colors[half:]]
#gene_labels_col2 = list(gene_names[half:])

# Combine the three legend components
#all_handles = gene_handles_col1 + gene_handles_col2
#all_labels = gene_labels_col1 + gene_labels_col2
#group_handles  +
#group_labels +
# Plotting the combined legend
# Use 'bbox_to_anchor' to place the legend outside the plot area on the right.
# Use ncol=2 to split the gene legend into two columns, drastically reducing the height.


plt.legend(
    group_handles, 
    group_labels, 
    loc='upper left',
    bbox_to_anchor=(1.2, 0.6), # Shifted down so it doesn't overlap leg1
    title="Mouse Models",
    frameon=False,
    title_fontsize=16,
    fontsize="large"
)
#plt.gca().add_artist(leg1)
#plt.legend(
 #   all_handles, 
  #  all_labels, 
  #  loc='upper left',        # Changed to upper left for easier alignment
  #  bbox_to_anchor=(1.15, 1), # 1.05 is just outside the plot border
  #  title="Mutations",
  #  ncol=2,  
  #  columnspacing=1.0,       # Adjust this to make columns tighter or wider
  #  frameon=False,
  #  title_fontsize=14,
  #  fontsize="large"        # 'medium' might save more horizontal space than 'large'
#)
# Position the second legend below the first


plt.xlabel("Dimension 1", fontsize=14)
plt.ylabel("Dimension 2", fontsize=14)
plt.xlabel("Dimension 1",fontsize=14)
plt.tick_params(axis='both', labelsize=14)
plt.ylabel("Dimension 2",fontsize=14)
plt.title("Mouse Fitness Landscape",fontsize=16)
# Adjust layout to make room for the wider, two-column legend on the right
#plt.tight_layout(rect=[0, 0, 0.75, 1])#rect=[0, 0, 0.82, 1]) 

plt.savefig(r"Mouse_fitness_landscape_Diverge.pdf", dpi=300,bbox_inches='tight')
plt.show()



g12c_muts = (mutants[0:28,0], mutants[0:28,1])
g12d_muts = (mutants[28:56,0], mutants[28:56,1])
egfr_muts = (mutants[56:,0], mutants[56:,1])

# Plot clone centers Z
z_g12c_base = (Z_np[0,0], Z_np[0,1])
z_g12d_base = (Z_np[1,0], Z_np[1,1])
z_egfr_base = (Z_np[2,0], Z_np[2,1])


xmin = min(mutants[:,0].min(), -4)
xmax = max(mutants[:,0].max(),  4)
ymin = min(mutants[:,1].min(), -4)
ymax = max(mutants[:,1].max(),  4)

#xx, yy = np.meshgrid(
#    np.linspace(xmin, xmax, 200),
#    np.linspace(ymin, ymax, 200)
#)

# Compute fitness on grid: F = X * exp( - ((x+y)^2) / 2 )
#xy_sum = xx + yy
#grid_fit = np.exp(-(xy_sum**2)/2) * float(reX)
def get_log_fitness(coords, X_scale=reX):
    dot_product = np.sum(coords**2, axis=-1)
    return np.log(X_scale) - (dot_product / 2.0)

# 2. Setup the grid
x = np.linspace(-4, 1.5, 100)
y = np.linspace(0, 4, 100)
X, Y = np.meshgrid(x, y)
Z = get_log_fitness(np.stack([X, Y], axis=-1))

# 3. Create the plot
plt.figure(figsize=(8, 6))

# Filled contours represent the "elevation"
#cp = plt.contourf(X, Y, Z, levels=20, cmap='viridis')
#plt.colorbar(cp, label='Log-Fitness')

### --- Plotting -------------------------------------------------------------
plt.rcParams.update({'font.size': 12}) # Default font size increased from ~10 to 12
plt.figure(figsize=(14, 9)) 


# --- Gene Names and Colors ---
# Extract unique gene names
gene_names = AllMice['gene'].unique() 
# Define the color palette used for the M=28 genes
gene_colors = [
    "#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
    "#882255", "#44AA99", "#117733", "#332288", "#DDCC77", "#999933", "#AA4499",
    "#661100", "#6699CC", "#88CCEE", "#AA4466", "#44AA66", "#1177AA", "#882288",
    "#88AADD", "#DD7788", "#77AADD", "#44BB99", "#BB5566", "#008080"
]
assert len(gene_names) == len(gene_colors), "Number of genes and colors do not match"


# Contour plot
#cont = plt.contourf(xx, yy, jnp.log(grid_fit), levels=40, cmap="viridis")
#plt.colorbar(cont, label="Fitness (Log Scale)")
vmin, vmax, vcenter = -8, 5.5, 0
norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

cont = plt.contourf(X, Y, Z, levels=20, cmap='PuOr',norm=norm)
plt.colorbar(cont, label='Log-Fitness')

# Plot Z+P mutants
egfr_scatter = plt.scatter(mutants[56:,0], mutants[56:,1], color='darkcyan', marker="s", s=70)

# Plot clone centers Z

#z_egfr = plt.scatter(Z_np[2,0], Z_np[2,1], color=("slateblue"), edgecolor="black",
 #            s=120, marker="s")


# --- Custom Legend Creation ---

# 1. Create Handles and Labels for Clone/Mutant Groups (Shape Legend)
group_handles = [
    plt.Line2D([0], [0], marker='8', color='w', markerfacecolor='gray', markersize=8),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=8),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=8)
]
group_labels = ["G12C Mutants", "G12D Mutants", "EGFR Mutants"]

# 2. Create Handles and Labels for Clone Centers (Center Legend)
# Use the correct marker strings ('8', '^', 's')
#center_handles = [
#    plt.Line2D([0], [0], marker='8', color='w', markerfacecolor='slateblue', markeredgecolor='black', markersize=10),
#    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='slateblue', markeredgecolor='black', markersize=10),
#    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='slateblue', markeredgecolor='black', markersize=10)
#]
#center_labels = ["Clone Center Z (G12C)", "Clone Center Z (G12D)", "Clone Center Z (EGFR)"]

# 3. Create Handles and Labels for Genes (Color Legend) - Split into two columns
# (This section is fine and unchanged)
half = len(gene_names) // 2
gene_handles_col1 = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8) 
                     for color in gene_colors[:half]]
gene_labels_col1 = list(gene_names[:half])

gene_handles_col2 = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8) 
                     for color in gene_colors[half:]]
gene_labels_col2 = list(gene_names[half:])

# Combine the three legend components
all_handles = gene_handles_col1 + gene_handles_col2
all_labels = gene_labels_col1 + gene_labels_col2
#group_handles  +
#group_labels +
# Plotting the combined legend
# Use 'bbox_to_anchor' to place the legend outside the plot area on the right.
# Use ncol=2 to split the gene legend into two columns, drastically reducing the height.


#leg1=plt.legend(
 #   group_handles, 
  #  group_labels, 
  #  loc='upper left',
   # bbox_to_anchor=(1.2, 0.3), # Shifted down so it doesn't overlap leg1
   # title="Mouse Models",
   # frameon=False,
   # title_fontsize=14,
   # fontsize="large"
#)
#plt.gca().add_artist(leg1)
#plt.legend(
 #   all_handles, 
  #  all_labels, 
   # loc='upper left',        # Changed to upper left for easier alignment
    #bbox_to_anchor=(1.15, 1), # 1.05 is just outside the plot border
    #title="Mutations",
    #ncol=2,  
    #columnspacing=1.0,       # Adjust this to make columns tighter or wider
    #frameon=False,
    #title_fontsize=14,
    #fontsize="large"        # 'medium' might save more horizontal space than 'large'
#)
# Position the second legend below the first


plt.xlabel("Dimension 1", fontsize=14)
plt.ylabel("Dimension 2", fontsize=14)
plt.xlabel("Dimension 1",fontsize=14)
plt.tick_params(axis='both', labelsize=14)
plt.ylabel("Dimension 2",fontsize=14)
plt.title("EGFR Mice Fitness Landscape",fontsize=16)
# Adjust layout to make room for the wider, two-column legend on the right
#plt.tight_layout(rect=[0, 0, 0.75, 1])#rect=[0, 0, 0.82, 1]) 

plt.savefig(r"EGFR_Mouse_fitness_landscape.pdf", dpi=300,bbox_inches='tight')
plt.show()


def get_log_fitness(coords, X_scale=reX):
    dot_product = np.sum(coords**2, axis=-1)
    return np.log(X_scale) - (dot_product / 2.0)

# 2. Setup the grid
x = np.linspace(-2, 4, 100)
y = np.linspace(-4, .1, 100)
X, Y = np.meshgrid(x, y)
Z = get_log_fitness(np.stack([X, Y], axis=-1))

# 3. Create the plot
plt.figure(figsize=(8, 6))

# Filled contours represent the "elevation"
#cp = plt.contourf(X, Y, Z, levels=20, cmap='viridis')
#plt.colorbar(cp, label='Log-Fitness')

### --- Plotting -------------------------------------------------------------
plt.rcParams.update({'font.size': 12}) # Default font size increased from ~10 to 12
plt.figure(figsize=(14, 9)) 


# --- Gene Names and Colors ---
# Extract unique gene names
gene_names = AllMice['gene'].unique() 
# Define the color palette used for the M=28 genes
gene_colors = [
    "#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
    "#882255", "#44AA99", "#117733", "#332288", "#DDCC77", "#999933", "#AA4499",
    "#661100", "#6699CC", "#88CCEE", "#AA4466", "#44AA66", "#1177AA", "#882288",
    "#88AADD", "#DD7788", "#77AADD", "#44BB99", "#BB5566", "#008080"
]
assert len(gene_names) == len(gene_colors), "Number of genes and colors do not match"


# Contour plot
#cont = plt.contourf(xx, yy, jnp.log(grid_fit), levels=40, cmap="viridis")
#plt.colorbar(cont, label="Fitness (Log Scale)")

vmin, vmax, vcenter = -8, 5.5, 0
norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

cont = plt.contourf(X, Y, Z, levels=20, cmap='PuOr',norm=norm)
plt.colorbar(cont, label='Log-Fitness')

# Plot Z+P mutants
g12c_scatter = plt.scatter(mutants[0:28,0], mutants[0:28,1], color="royalblue", marker="8", s=70)
#g12d_scatter = plt.scatter(mutants[28:56,0], mutants[28:56,1], color=gene_colors, alpha=.5, marker="^", s=70)
#egfr_scatter = plt.scatter(mutants[56:,0], mutants[56:,1], color=gene_colors, alpha=.5, marker="s", s=70)

# Plot clone centers Z
#z_g12c = plt.scatter(Z_np[0,0], Z_np[0,1], color=("slateblue"), edgecolor="black",
 #            s=120, marker="8")


# --- Custom Legend Creation ---

# 1. Create Handles and Labels for Clone/Mutant Groups (Shape Legend)
group_handles = [
    plt.Line2D([0], [0], marker='8', color='w', markerfacecolor='gray', markersize=8),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=8),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=8)
]
group_labels = ["G12C Mutants", "G12D Mutants", "EGFR Mutants"]

# 2. Create Handles and Labels for Clone Centers (Center Legend)
# Use the correct marker strings ('8', '^', 's')
#center_handles = [
#    plt.Line2D([0], [0], marker='8', color='w', markerfacecolor='slateblue', markeredgecolor='black', markersize=10),
#    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='slateblue', markeredgecolor='black', markersize=10),
#    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='slateblue', markeredgecolor='black', markersize=10)
#]
#center_labels = ["Clone Center Z (G12C)", "Clone Center Z (G12D)", "Clone Center Z (EGFR)"]

# 3. Create Handles and Labels for Genes (Color Legend) - Split into two columns
# (This section is fine and unchanged)
half = len(gene_names) // 2
gene_handles_col1 = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8) 
                     for color in gene_colors[:half]]
gene_labels_col1 = list(gene_names[:half])

gene_handles_col2 = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8) 
                     for color in gene_colors[half:]]
gene_labels_col2 = list(gene_names[half:])

# Combine the three legend components
all_handles = gene_handles_col1 + gene_handles_col2
all_labels = gene_labels_col1 + gene_labels_col2
#group_handles  +
#group_labels +
# Plotting the combined legend
# Use 'bbox_to_anchor' to place the legend outside the plot area on the right.
# Use ncol=2 to split the gene legend into two columns, drastically reducing the height.


#leg1=plt.legend(
 #   group_handles, 
  #  group_labels, 
  #  loc='upper left',
   # bbox_to_anchor=(1.2, 0.3), # Shifted down so it doesn't overlap leg1
   # title="Mouse Models",
   # frameon=False,
   # title_fontsize=14,
   # fontsize="large"
#)
#plt.gca().add_artist(leg1)
#plt.legend(
#    all_handles, 
#    all_labels, 
#   loc='upper left',        # Changed to upper left for easier alignment
#   bbox_to_anchor=(1.15, 1), # 1.05 is just outside the plot border
#    title="Mutations",
#    ncol=2,  
#    columnspacing=1.0,       # Adjust this to make columns tighter or wider
#    frameon=False,
#    title_fontsize=14,
#    fontsize="large"        # 'medium' might save more horizontal space than 'large'
#)
# Position the second legend below the first


plt.xlabel("Dimension 1", fontsize=14)
plt.ylabel("Dimension 2", fontsize=14)
plt.xlabel("Dimension 1",fontsize=14)
plt.tick_params(axis='both', labelsize=14)
plt.ylabel("Dimension 2",fontsize=14)
plt.title("G12C Mice Fitness Landscape",fontsize=16)
# Adjust layout to make room for the wider, two-column legend on the right
#plt.tight_layout(rect=[0, 0, 0.75, 1])#rect=[0, 0, 0.82, 1]) 

plt.savefig(r"G12C_Mouse_fitness_landscape.pdf", dpi=300,bbox_inches='tight')
plt.show()


x = np.linspace(-4, 1,8, 100)
y = np.linspace(-.5, 4, 100)
X, Y = np.meshgrid(x, y)
Z = get_log_fitness(np.stack([X, Y], axis=-1))

# 3. Create the plot
plt.figure(figsize=(8, 6))

# Filled contours represent the "elevation"
#cp = plt.contourf(X, Y, Z, levels=20, cmap='viridis')
#plt.colorbar(cp, label='Log-Fitness')

### --- Plotting -------------------------------------------------------------
plt.rcParams.update({'font.size': 12}) # Default font size increased from ~10 to 12
plt.figure(figsize=(14, 9)) 


# --- Gene Names and Colors ---
# Extract unique gene names
gene_names = AllMice['gene'].unique() 
# Define the color palette used for the M=28 genes
gene_colors = [
    "#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
    "#882255", "#44AA99", "#117733", "#332288", "#DDCC77", "#999933", "#AA4499",
    "#661100", "#6699CC", "#88CCEE", "#AA4466", "#44AA66", "#1177AA", "#882288",
    "#88AADD", "#DD7788", "#77AADD", "#44BB99", "#BB5566", "#008080"
]
assert len(gene_names) == len(gene_colors), "Number of genes and colors do not match"


# Contour plot
#cont = plt.contourf(xx, yy, jnp.log(grid_fit), levels=40, cmap="viridis")
#plt.colorbar(cont, label="Fitness (Log Scale)")
#ont = plt.contourf(X, Y, Z, levels=20, cmap='spring')
#plt.colorbar(cont, label='Log-Fitness')
vmin, vmax, vcenter = -8, 5.5, 0
norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

cont = plt.contourf(X, Y, Z, levels=20, cmap='PuOr',norm=norm)
plt.colorbar(cont, label='Log-Fitness')

# Plot Z+P mutants
#g12c_scatter = plt.scatter(mutants[0:28,0], mutants[0:28,1], color=gene_colors, marker="8", s=70)
g12d_scatter = plt.scatter(mutants[28:56,0], mutants[28:56,1], color="brown", marker="^", s=70)
#egfr_scatter = plt.scatter(mutants[56:,0], mutants[56:,1], color=gene_colors, alpha=.5, marker="s", s=70)

# Plot clone centers Z
#z_g12d = plt.scatter(Z_np[1,0], Z_np[1,1], color=("slateblue"), edgecolor="black",
 #            s=120, marker="^")


# --- Custom Legend Creation ---

# 1. Create Handles and Labels for Clone/Mutant Groups (Shape Legend)
group_handles = [
    plt.Line2D([0], [0], marker='8', color='w', markerfacecolor='gray', markersize=8),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=8),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=8)
]
group_labels = ["G12C Mutants", "G12D Mutants", "EGFR Mutants"]

# 2. Create Handles and Labels for Clone Centers (Center Legend)
# Use the correct marker strings ('8', '^', 's')
#center_handles = [
#    plt.Line2D([0], [0], marker='8', color='w', markerfacecolor='slateblue', markeredgecolor='black', markersize=10),
#    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='slateblue', markeredgecolor='black', markersize=10),
#    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='slateblue', markeredgecolor='black', markersize=10)
#]
#center_labels = ["Clone Center Z (G12C)", "Clone Center Z (G12D)", "Clone Center Z (EGFR)"]

# 3. Create Handles and Labels for Genes (Color Legend) - Split into two columns
# (This section is fine and unchanged)
half = len(gene_names) // 2
gene_handles_col1 = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8) 
                     for color in gene_colors[:half]]
gene_labels_col1 = list(gene_names[:half])

gene_handles_col2 = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8) 
                     for color in gene_colors[half:]]
gene_labels_col2 = list(gene_names[half:])

# Combine the three legend components
all_handles = gene_handles_col1 + gene_handles_col2
all_labels = gene_labels_col1 + gene_labels_col2
#group_handles  +
#group_labels +
# Plotting the combined legend
# Use 'bbox_to_anchor' to place the legend outside the plot area on the right.
# Use ncol=2 to split the gene legend into two columns, drastically reducing the height.


#leg1=plt.legend(
 #   group_handles, 
  #  group_labels, 
  #  loc='upper left',
   # bbox_to_anchor=(1.2, 0.3), # Shifted down so it doesn't overlap leg1
   # title="Mouse Models",
   # frameon=False,
   # title_fontsize=14,
   # fontsize="large"
#)
#plt.gca().add_artist(leg1)
#plt.legend(
#    all_handles, 
#    all_labels, 
#    loc='upper left',        # Changed to upper left for easier alignment
#    bbox_to_anchor=(1.15, 1), # 1.05 is just outside the plot border
#    title="Mutations",
#    ncol=2,  
#    columnspacing=1.0,       # Adjust this to make columns tighter or wider
#    frameon=False,
#    title_fontsize=14,
#    fontsize="large"        # 'medium' might save more horizontal space than 'large'
#)
# Position the second legend below the first


plt.xlabel("Dimension 1", fontsize=14)
plt.ylabel("Dimension 2", fontsize=14)
plt.xlabel("Dimension 1",fontsize=14)
plt.tick_params(axis='both', labelsize=14)
plt.ylabel("Dimension 2",fontsize=14)
plt.title("G12D Mice Fitness Landscape",fontsize=16)
# Adjust layout to make room for the wider, two-column legend on the right
#plt.tight_layout(rect=[0, 0, 0.75, 1])#rect=[0, 0, 0.82, 1]) 

plt.savefig(r"G12D_Mouse_fitness_landscape.pdf", dpi=300,bbox_inches='tight')
plt.show()
