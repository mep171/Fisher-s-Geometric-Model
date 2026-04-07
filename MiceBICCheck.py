# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 13:03:16 2026

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
MAX_REGRESS_ITER = 50000
#CONSTRAIN_ROTATION=True
#D=2
#C=13
#M=18
seed = 15

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
    def __init__(self, C=3, D=1, M=28, scale=1, CONSTRAIN_ROTATION=True, **dimensions):
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
    def __init__(self, landscape_obj, observed_fitnesses,norm,C=3, D=1, M=28, CONSTRAIN_ROTATION=True,LOG_FITNESS=True):
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
    
        #weighted_residuals = (obs_flat - pred_flat) / weight_flat
    
    # Using Huber loss on the weighted residuals
        loss = jaxopt.loss.huber_loss(obs_flat/weight_flat, pred_flat/weight_flat)
    
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

regZP=regress_LBFGS(regression_obj, landscape_obj, jnp.ravel(mice_fitness), ALLMiceNormNP, Z, P, X)
reZ1, reP1, reX1=regression_obj.reconstruct_ZP(regZP, 1)
predfit1=landscape_obj.calculate_fitness(reZ1, reP1, reX1)

#MicePredFitSeed15=pd.read_csv(r"C:\Users\Meaghan Parks\Documents\McFarlandLabProjects\PredFitBlairMouse15WithCorrectCI.csv").to_numpy()

from sklearn.metrics import r2_score

r2 = r2_score(jnp.ravel(mice_fitness), predfit1)
from sklearn.metrics import mean_absolute_error
MAE = mean_absolute_error(jnp.ravel(mice_fitness), predfit1)
import scipy.stats as stats

correlation_coefficient, p_value = stats.pearsonr(jnp.ravel(mice_fitness), jnp.ravel(predfit1))

plt.rcParams.update({'font.size': 12})

plt.scatter(jnp.ravel(mice_fitness),predfit1,c=ALLMiceNormNP.flatten(),cmap="summer",marker="o",alpha=.5)
plt.title("Regressed vs. Measured Fitness",fontsize=16)
plt.ylabel("Log Regressed Fitness",fontsize=14)
plt.xlabel("Log Observed Fitness",fontsize=14)
plt.text(-1, -1.5, "MAE= 0.1294", fontsize=12)
plt.text(-1, -1.8, "PCC = 0.8208, p < .00001  ", fontsize=12)
plt.colorbar(label='Experimental Uncertainty')
plt.plot((-2,2),(-2,2),color="black")
#plt.savefig("Reg_vs_Real_Mouse.pdf", dpi=300)

plt.show()


MicePredFitSeed15=pd.read_csv(r"C:\Users\Meaghan Parks\Documents\McFarlandLabProjects\BlairMouse111Results.csv",index_col=0).to_numpy()

parameter2=pd.read_csv(r"C:\Users\Meaghan Parks\Documents\McFarlandLabProjects\regressedZPBlairMouse302WithCorrectCI.csv").to_numpy()


def calculate_bic(min_loss, n_data_points, n_parameters):
    return n_parameters * jnp.log(n_data_points) + 2 * min_loss


loss1=(sum(jaxopt.loss.huber_loss(jnp.ravel(mice_fitness)/(jnp.ravel(ALLMiceNormNP)), predfit1.flatten()/(jnp.ravel(ALLMiceNormNP)))))

loss2=(sum(jaxopt.loss.huber_loss(jnp.ravel(mice_fitness)/(jnp.ravel(ALLMiceNormNP)), MicePredFitSeed15.flatten()/(jnp.ravel(ALLMiceNormNP)))))


bic_d2 = calculate_bic(loss2, len(jnp.ravel(mice_fitness)), (len(parameter2)-3))
bic_d1 = calculate_bic(loss1, len(jnp.ravel(mice_fitness)),len(regZP))



def weighted_correlation(x, y, w):
    """Calculates the weighted Pearson correlation coefficient."""
    def weighted_mean(v, w):
        return jnp.sum(v * w) / jnp.sum(w)

    mw_x = weighted_mean(x, w)
    mw_y = weighted_mean(y, w)

    cov_xy = jnp.sum(w * (x - mw_x) * (y - mw_y))
    cov_xx = jnp.sum(w * (x - mw_x)**2)
    cov_yy = jnp.sum(w * (y - mw_y)**2)

    return cov_xy / jnp.sqrt(cov_xx * cov_yy)

# Compare these for D=1 and D=2
# Use 1/sigma^2 as the weight (standard precision weighting)
weights = 1.0 / (ALLMiceNormNP.ravel()**2)
w_corr = weighted_correlation(mice_fitness.ravel(), MicePredFitSeed15.ravel(), weights)
print(f"Weighted Correlation: {w_corr:.4f}")