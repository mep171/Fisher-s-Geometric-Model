# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 14:07:43 2026

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


MAX_REGRESS_ITER = 10000
L2_LAMBDA = 1e-5
C_VAL = 20
M_VAL = 20
REAL_X = 25.0
GUESS_X = 1.0
NOISE_LEVEL = 0.05
D_TRUE_VALUES = [1, 2, 3, 4, 5]
D_GUESS_VALUES = [1, 2, 3, 4, 5]
SEED_SIM = 90
SEED_GUESS_LIST = [980]


def gauge_fix_posthoc(Z_pred, P_pred, Z_ref, P_ref):
    P_mean = np.mean(P_pred, axis=0)
    P_centered = P_pred - P_mean
    Z_shifted = Z_pred + P_mean

    A = np.vstack([Z_shifted, P_centered])
    B = np.vstack([Z_ref, P_ref])
    
    M = A.T @ B
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt
    
    return Z_shifted @ R, P_centered @ R

def calculate_stable_bic(mse, n_samples, k_eff):
    return n_samples * np.log(mse + 1e-10) + k_eff * np.log(n_samples)

# --- Classes ---

class Landscape:
    def __init__(self, C=20, D=2, M=20):
        self.C, self.D, self.M = C, D, M

    def simulate_dataset(self, key, noise=0.0):
        key, kz, kp, kz_noise, kp_noise = random.split(key, 5)
        Z = random.normal(kz, (self.C, self.D))
        P = random.normal(kp, (self.M, self.D))
        if noise > 0:
            Z += random.normal(kz_noise, Z.shape) * noise
            P += random.normal(kp_noise, P.shape) * noise
        return key, Z, P

    def calculate_fitness(self, Z, P, X):
        combined = Z[:, None, :] + P[None, :, :]
        dist_sq = jnp.sum(combined**2, axis=2)
        return jnp.log(jnp.abs(X)) - (dist_sq / 2.0)

class RegressionProblem:
    def __init__(self, landscape_obj, observed_fitness, norm, D_guess):
        self.ls = landscape_obj
        self.obs = observed_fitness
        self.norm = norm
        self.D = D_guess 

    def get_parameter_vector(self, Z, P, X):
        return jnp.concatenate([jnp.array([X]), jnp.ravel(Z), jnp.ravel(P)])

    def reconstruct_ZP(self, p_vec):
        z_size = self.ls.C * self.D
        p_size = self.ls.M * self.D
        X = p_vec[0]
        Z = p_vec[1 : 1 + z_size].reshape((self.ls.C, self.D))
        P = p_vec[1 + z_size : 1 + z_size + p_size].reshape((self.ls.M, self.D))
        return Z, P, X

    def loss_function(self, p_vec):
        Z, P, X = self.reconstruct_ZP(p_vec)
        pred = self.ls.calculate_fitness(Z, P, X)
        return (
            jnp.mean(((self.obs - pred) / self.norm) ** 2)
            + L2_LAMBDA * (jnp.sum(Z**2) + jnp.sum(P**2))
        )

def regress_LBFGS(prob_obj, init_Z, init_P, init_X):
    solver = jaxopt.ScipyMinimize(method="L-BFGS-B", fun=prob_obj.loss_function, maxiter=MAX_REGRESS_ITER)
    init_pv = prob_obj.get_parameter_vector(init_Z, init_P, init_X)
    res = solver.run(init_pv)
    return res.params, res.state.fun_val

# --- Plotting Functions ---

def save_individual_plots(results_list):
    df = pd.DataFrame(results_list)
    D_true_vals = sorted(df['True_Dimension'].unique())
    colors = plt.cm.plasma(np.linspace(0, 0.9, len(D_true_vals)))

    plt.figure(figsize=(9, 7))
    for i, dt in enumerate(D_true_vals):
        sub = df[df['True_Dimension'] == dt]
        plt.plot(sub['Guess_Dimension'], sub['Predicted_R2'], marker='o', 
                 markersize=8, linewidth=2, color=colors[i], label=f'$D_{{true}}={dt}$')
    plt.title(r'$R^2$ Score vs. Model Dimension $D_{model}$', fontsize=16)
    plt.xlabel("Model Dimension $D_{model}$", fontsize=14)
    plt.ylabel("Avg. $R^2$ Score", fontsize=14)
    plt.legend(title="True Dimension", loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("R2_Score_Analysis.pdf", dpi=300)
    plt.show()

    plt.figure(figsize=(9, 7))
    for i, dt in enumerate(D_true_vals):
        sub = df[df['True_Dimension'] == dt]
        plt.plot(sub['Guess_Dimension'], sub['BIC_Score'], marker='s', 
                 markersize=8, linewidth=2, linestyle='--', color=colors[i], label=f'$D_{{true}}={dt}$')
    plt.title(r'BIC Score vs. Model Dimension $D_{model}$', fontsize=16)
    plt.xlabel("Model Dimension $D_{model}$", fontsize=14)
    plt.ylabel("BIC Score (Lower is Better)", fontsize=14)
    plt.legend(title="True Dimension", loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("BIC_Score_Analysis.pdf", dpi=300)
    plt.show()


def main():
    key_sim = random.PRNGKey(SEED_SIM)
    results_list = []
    n_samples = C_VAL * M_VAL

    for D_true in D_TRUE_VALUES:
        ls_true = Landscape(C=C_VAL, D=D_true, M=M_VAL)
        key_sim, k1 = random.split(key_sim)
        _, real_Z, real_P = ls_true.simulate_dataset(k1, noise=NOISE_LEVEL)
        sim_fit = ls_true.calculate_fitness(real_Z, real_P, REAL_X)
        norm = jnp.std(sim_fit) + 1e-6

        for seed_guess in SEED_GUESS_LIST:
            key_guess = random.PRNGKey(seed_guess)
            for D_guess in D_GUESS_VALUES:
                ls_guess = Landscape(C=C_VAL, D=D_guess, M=M_VAL)
                prob = RegressionProblem(ls_guess, sim_fit, norm, D_guess)
                key_guess, k_sub = random.split(key_guess)
                _, gZ, gP = ls_guess.simulate_dataset(k_sub)

                params, _ = regress_LBFGS(prob, gZ, gP, GUESS_X)
                rZ, rP, rX = prob.reconstruct_ZP(params)
                
                
                pred_fit = ls_guess.calculate_fitness(rZ, rP, rX)
                mse = float(jnp.mean((sim_fit - pred_fit) ** 2))
                
                ss_res = jnp.sum((sim_fit - pred_fit) ** 2)
                ss_tot = jnp.sum((sim_fit - jnp.mean(sim_fit)) ** 2)
                r2 = float(1 - (ss_res / ss_tot))

                n_params = len(params)
                constraints = (D_guess * (D_guess + 1)) // 2
                eff_p = n_params - constraints
                
                bic = calculate_stable_bic(mse, n_samples, eff_p)

                results_list.append({
                    'True_Dimension': D_true,
                    'Guess_Dimension': D_guess,
                    'Predicted_R2': r2,
                    'BIC_Score': bic
                })
                print(f"Dt:{D_true} Dg:{D_guess} | R2:{r2:.3f} | BIC:{bic:.1f}")

    save_individual_plots(results_list)

if __name__ == "__main__":
    main()
