import numpy as np
from typing import List, Any

def system_dynamics(t, state, pancreas_agents, liver_agents, env_params):
    """
    Bio-Physical ODE System for Glucose-Insulin Dynamics.
    
    State Vector y:
    [0] Glucose (G)
    [1] Plasma Insulin (I)
    [2...N+2] Beta-Cell Mass (for each agent)
    """
    
    # Unpack State
    G = state[0]
    I = state[1]
    
    # 1. Pancreas Dynamics (Secretion & Plasticity)
    total_secretion_rate = 0.0
    beta_mass_derivatives = []
    
    # Aggregate individual agent dynamics
    for i, agent in enumerate(pancreas_agents):
        # Current functional mass of this agent
        # State vector index offset by 2
        current_beta_mass = state[2 + i]
        
        # Calculate rates
        sec_rate = agent.calculate_secretion_rate(G)
        plast_rate = agent.calculate_plasticity_rate(G, current_beta_mass)
        
        # Scale secretion by beta mass (Mass * per_cell_secretion)
        total_secretion_rate += current_beta_mass * sec_rate
        
        beta_mass_derivatives.append(plast_rate)
        
    # Normalize secretion to plasma volume
    # Assuming standard volume distribution
    dI_secretion = total_secretion_rate / env_params.get('volume_distribution', 10.0)
    
    # 2. Liver Dynamics (HGP & Uptake)
    total_net_flux = 0.0
    for agent in liver_agents:
        flux = agent.calculate_hgp_rate(I, G)
        total_net_flux += flux
        
    avg_liver_flux = total_net_flux / len(liver_agents) * 5.0 # Scaling factor
    
    # 3. Peripheral Uptake (Muscle/Fat) - Simplified Mass Action for now
    # dG_peripheral = - k * G * (1 + S * I)
    S_peripheral = 0.05 # Insulin sensitivity
    k_uptake = 0.01
    peripheral_uptake = k_uptake * G * (1 + S_peripheral * I)
    
    # 4. System Derivatives
    
    # dG/dt = Influx + LiverFlux - PeripheralUptake
    k_influx = env_params.get('glucose_influx', 0.0) # e.g. from meal
    dG_dt = k_influx + avg_liver_flux - peripheral_uptake
    
    # dI/dt = Secretion - Clearance
    gamma_clearance = 0.1
    dI_dt = dI_secretion - gamma_clearance * I
    
    # Assemble derivatives
    derivs = [dG_dt, dI_dt] + beta_mass_derivatives
    
    return derivs
