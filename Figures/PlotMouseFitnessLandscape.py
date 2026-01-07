# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 20:00:14 2025

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
MAX_REGRESS_ITER = 5000
#CONSTRAIN_ROTATION=True
#D=2
#C=13
#M=18
seed = 15
seed2 = 90
key = random.PRNGKey(seed)
key2 = random.PRNGKey(seed2)
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
    
    def calculate_fitness(self, Z, P, X, noise=0):
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
        self.key=key
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

    def loss_function(self,parameter_vector,observed_fitness,norm,scalar_residual=True):
        Z, P, X = self.reconstruct_ZP(parameter_vector,self.D)
        predicted_fitness = self.landscape.calculate_fitness(Z, P, X)/(jnp.ravel(norm))
        observed_fitness=observed_fitness/(jnp.ravel(abs(norm)))
        #predicted_fitness = predicted_fitness.at[21].set(0)
        loss=jaxopt.loss.huber_loss(observed_fitness, predicted_fitness)
        #print(loss.shape)
        #loss=loss.at[21].set(0)
        #loss=loss.at[1].set(0)
        return loss.sum() if scalar_residual else loss.ravel()
   

def regress_LBFGS(regression_obj, landscape_obj,simulated_fitness,norm,Z,P,X):
    parameter_vector = regression_obj.get_parameter_vector(Z,P,X)
    solver = jaxopt.LBFGS(fun=regression_obj.loss_function, maxiter=MAX_REGRESS_ITER)
    res = solver.run(parameter_vector, observed_fitness=simulated_fitness,norm=norm)
    return res.params

landscape_obj=landscape()


MiceG12C=pd.read_csv(r"Figure5A.csv")
MiceG12D=pd.read_csv(r"Figure5B.csv")
MiceEGFR=pd.read_csv(r"Figure5D.csv")


mergeMiceG12=pd.merge(MiceG12C,MiceG12D,on='gene',how='inner')
AllMice=pd.merge(mergeMiceG12,MiceEGFR,on="gene",how='inner')

AllMiceVals=AllMice.loc[:, ['tumor_enrichment_x', 'tumor_enrichment_y','tumor_enrichment']]
AllMiceCI=np.log(AllMice.loc[:,['CI_lower_x','CI_upper_x','CI_lower_y','CI_upper_y','CI_lower','CI_upper' ]])
AllMiceCI['sumCI_x']=(AllMiceCI['CI_upper_x']-AllMiceCI['CI_lower_x'])
AllMiceCI['sumCI_y']=(AllMiceCI['CI_upper_y']-AllMiceCI['CI_lower_y'])
AllMiceCI['sumCI']=(AllMiceCI['CI_upper']-AllMiceCI['CI_lower'])
ALLMiceNorm=AllMiceCI.loc[:,['sumCI_x','sumCI_y','sumCI']]
ALLMiceNormNP=ALLMiceNorm.to_numpy()
key, Z, P = landscape_obj.simulate_dataset(key)
X=jnp.array(1)

mice_fitness=jnp.transpose(AllMiceVals.to_numpy())
mice_fitness=jnp.log(mice_fitness)
regression_obj=RegressionProblem(landscape_obj, mice_fitness,ALLMiceNormNP)
#regressedZP=regress_LBFGS(regression_obj, landscape_obj, jnp.ravel(mice_fitness),ALLMiceNormNP,Z,P,X)

regZP=regress_LBFGS(regression_obj, landscape_obj, mice_fitness.flatten(), ALLMiceNormNP, Z, P, X)

rereconZ,rereconP,reX=regression_obj.reconstruct_ZP(regZP,2)
predFit=landscape_obj.calculate_fitness(rereconZ, rereconP,reX,key)

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

xx, yy = np.meshgrid(
    np.linspace(xmin, xmax, 200),
    np.linspace(ymin, ymax, 200)
)

# Compute fitness on grid: F = X * exp( - ((x+y)^2) / 2 )
xy_sum = xx + yy
grid_fit = np.exp(-(xy_sum**2)/2) * float(reX)


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
cont = plt.contourf(xx, yy, jnp.log(grid_fit), levels=40, cmap="viridis")
plt.colorbar(cont, label="Fitness (Log Scale)")

# Plot Z+P mutants
g12c_scatter = plt.scatter(mutants[0:28,0], mutants[0:28,1], color=gene_colors, alpha=.5, marker="8", s=70)
g12d_scatter = plt.scatter(mutants[28:56,0], mutants[28:56,1], color=gene_colors, alpha=.5, marker="^", s=70)
egfr_scatter = plt.scatter(mutants[56:,0], mutants[56:,1], color=gene_colors, alpha=.5, marker="s", s=70)

# Plot clone centers Z
z_g12c = plt.scatter(Z_np[0,0], Z_np[0,1], color=("slateblue"), edgecolor="black",
             s=120, marker="8")
z_g12d = plt.scatter(Z_np[1,0], Z_np[1,1], color=("slateblue"), edgecolor="black",
             s=120, marker="^")
z_egfr = plt.scatter(Z_np[2,0], Z_np[2,1], color=("slateblue"), edgecolor="black",
             s=120, marker="s")


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
center_handles = [
    plt.Line2D([0], [0], marker='8', color='w', markerfacecolor='slateblue', markeredgecolor='black', markersize=10),
    plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='slateblue', markeredgecolor='black', markersize=10),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='slateblue', markeredgecolor='black', markersize=10)
]
center_labels = ["Clone Center Z (G12C)", "Clone Center Z (G12D)", "Clone Center Z (EGFR)"]

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
all_handles = group_handles + center_handles + gene_handles_col1 + gene_handles_col2
all_labels = group_labels + center_labels + gene_labels_col1 + gene_labels_col2
# Plotting the combined legend
# Use 'bbox_to_anchor' to place the legend outside the plot area on the right.
# Use ncol=2 to split the gene legend into two columns, drastically reducing the height.

plt.legend(
    all_handles, 
    all_labels, 
    loc='center left',
    bbox_to_anchor=(1.2, 0.5),
    #title="Legend",
    ncol=2,  
    framealpha=0.9,
    title_fontsize=14, # Increase the legend title size
    fontsize="large" # Slightly smaller font for the gene list inside the legend
)
plt.xlabel("Dimension 1",fontsize=14)
plt.tick_params(axis='both', labelsize=14)
plt.ylabel("Dimension 2",fontsize=14)
plt.title("Mouse Fitness Landscape",fontsize=16)
# Adjust layout to make room for the wider, two-column legend on the right
plt.tight_layout()#rect=[0, 0, 0.82, 1]) 

plt.savefig(r"MtoMPlots\Mouse_fitness_landscape.pdf", dpi=300)
plt.show()
