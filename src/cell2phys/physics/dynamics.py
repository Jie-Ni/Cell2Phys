import numpy as np
from src.cell2phys.config import Config


def system_dynamics(state, t, pancreas_agents, liver_agents, env_params):
    """
    Bio-Physical ODE System for Glucose-Insulin Dynamics.
    Compatible with scipy.integrate.odeint (signature: state, t, *args).

    State Vector:
        [0]       Glucose (G) in mg/dL
        [1]       Plasma Insulin (I) in uU/mL
        [2..N+1]  Beta-Cell Mass fractions for each pancreas agent
    """
    n_agents = len(pancreas_agents)

    # --- Unpack & clamp state to physiological bounds ---
    G = np.clip(state[0], Config.GLUCOSE_MIN, Config.GLUCOSE_MAX)
    I = np.clip(state[1], Config.INSULIN_MIN, Config.INSULIN_MAX)

    # ===== 1. Pancreas: Insulin Secretion & Beta-Cell Plasticity =====
    total_secretion_rate = 0.0
    beta_mass_derivatives = []

    for i, agent in enumerate(pancreas_agents):
        beta_mass = np.clip(state[2 + i], Config.BETA_MASS_MIN, Config.BETA_MASS_MAX)

        sec_rate = agent.calculate_secretion_rate(G)
        plast_rate = agent.calculate_plasticity_rate(G, beta_mass)

        total_secretion_rate += beta_mass * sec_rate
        beta_mass_derivatives.append(plast_rate)

    vol = env_params.get("volume_distribution", 10.0)
    dI_secretion = total_secretion_rate / vol

    # ===== 2. Liver: HGP & Uptake =====
    total_net_flux = 0.0
    for agent in liver_agents:
        total_net_flux += agent.calculate_hgp_rate(I, G)

    n_liver = max(len(liver_agents), 1)
    avg_liver_flux = total_net_flux / n_liver

    # ===== 3. Peripheral Uptake (muscle/fat): simplified mass-action =====
    S_peripheral = 0.05   # insulin sensitivity of peripheral tissue
    k_uptake = 0.01
    peripheral_uptake = k_uptake * G * (1.0 + S_peripheral * I)

    # ===== 4. Assemble Derivatives =====
    k_influx = env_params.get("glucose_influx", 0.0)
    dG_dt = k_influx + avg_liver_flux - peripheral_uptake

    gamma_clearance = 0.1   # insulin clearance rate (1/min)
    dI_dt = dI_secretion - gamma_clearance * I

    return [dG_dt, dI_dt] + beta_mass_derivatives
