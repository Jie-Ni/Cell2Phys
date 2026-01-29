import torch
import scanpy as sc
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
from scipy.integrate import odeint

from src.cell2phys.config import Config, ExecutionMode

warnings.filterwarnings('ignore')

from src.cell2phys.agents.cell import CellAgent
from src.cell2phys.agents.liver import LiverAgent
from src.cell2phys.pharmacology.mechanisms import PharmacologyEngine
from src.cell2phys.physics.dynamics import system_dynamics

def run_drug_simulation():
    print(f"💊 [Cell2Phys] Starting In Silico Clinical Trial...")
    
    # 1. Setup Cohort
    target_agents = 20 
    print(f"   Creating {target_agents} Virtual Patients...")
    
    # Updated to point to the Zenodo dataset
    data_path = os.path.join(Config.PROJECT_ROOT, "data", "bastidas_ponce_2019.h5ad")
    
    if os.path.exists(data_path):
        adata = sc.read_h5ad(data_path)
        pancreas_df = adata[adata.obs['tissue'] == 'Pancreas'].to_df()
        liver_df = adata[adata.obs['tissue'] == 'Liver'].to_df()
        
        n_p = min(len(pancreas_df), target_agents)
        n_l = min(len(liver_df), target_agents)
        pancreas_agents = [CellAgent(i, pancreas_df.iloc[i]) for i in range(n_p)]
        liver_agents = [LiverAgent(i, liver_df.iloc[i]) for i in range(n_l)]
    else:
        print(f"❌ CRITICAL ERROR: Dataset not found at: {data_path}")
        print("   The simulation requires the specific single-cell atlas to initialize agents.")
        print("   Please download 'bastidas_ponce_2019.h5ad' from Zenodo (DOI: 10.5281/zenodo.18331267)")
        print("   and place it in the 'data/' directory.")
        exit(1) # Stop execution, do not use random data

    arms = [
        {"name": "Placebo", "drug_conc": 0.0},
        {"name": "Metformin (Therapeutic)", "drug_conc": 150.0}
    ]
    
    drug_params = {"E_max": 2.0, "EC50": 100.0, "n": 1.5}

    results = {}
    
    # Simulation Parameters
    t_start = 0.0
    t_end = 200.0
    dt = 1.0 # Step size 
    t_eval = np.arange(t_start, t_end, dt)
    
    env_params = {'glucose_influx': 2.0, 'volume_distribution': 10.0}

    print("\n🚀 Running Simulation Loop (LSODA + ASC)...")
    for arm in arms:
        print(f"   ...Simulating Arm: {arm['name']}")
        
        effect_magnitude = PharmacologyEngine.hill_langmuir(
            arm['drug_conc'], drug_params['E_max'], drug_params['EC50'], drug_params['n']
        )
        treatment_modifier = 1.0 + effect_magnitude
        
        for l in liver_agents:
            l.sensitivity_factor = min(3.0, 1.0 * treatment_modifier)
            
        initial_glucose = 200.0
        initial_insulin = 10.0
        initial_beta_masses = [1.0] * len(pancreas_agents)
        
        y0 = [initial_glucose, initial_insulin] + initial_beta_masses
        
        # Pre-calculate Cognitive Updates
        for p in pancreas_agents:
            p.update_parameters(initial_glucose)
            
        sol = odeint(
            system_dynamics, 
            y0, 
            t_eval, 
            args=(pancreas_agents, liver_agents, env_params),
            mxstep=5000
        )
        
        results[arm['name']] = sol[:, 0] # Store Glucose Trace

    # 4. Data Export & Statistical Analysis
    print("\n💾 Exporting Experimental Data...")
    
    export_data = []
    
    for arm in arms:
        data_series = results[arm['name']]
        
        # Calculate Key Metrics
        start_glucose = data_series[0]
        end_glucose = data_series[-1]
        min_glucose = np.min(data_series)
        max_glucose = np.max(data_series)
        auc_glucose = np.trapz(data_series, dx=dt)
        
        # Time in Range (70-140 mg/dL)
        tir_mask = (data_series >= 70) & (data_series <= 140)
        time_in_range_percent = (np.sum(tir_mask) / len(data_series)) * 100.0
        
        stat_row = {
            "Arm": arm['name'],
            "Drug_Conc_uM": arm['drug_conc'],
            "Start_Glucose": start_glucose,
            "End_Glucose": end_glucose,
            "Min_Glucose": min_glucose,
            "Max_Glucose": max_glucose,
            "AUC_Glucose": auc_glucose,
            "Time_in_Range_Pct": time_in_range_percent
        }
        export_data.append(stat_row)
        
        # Save raw time-series
        ts_df = pd.DataFrame({
            "Time": t_eval,
            "Glucose": data_series
        })
        ts_filename = f"series_{arm['name'].replace(' ', '_')}.csv"
        ts_path = os.path.join(Config.PROJECT_ROOT, "results", "data", ts_filename)
        os.makedirs(os.path.dirname(ts_path), exist_ok=True)
        ts_df.to_csv(ts_path, index=False)
        
    # Save Summary Statistics
    summary_df = pd.DataFrame(export_data)
    summary_path = os.path.join(Config.PROJECT_ROOT, "results", "data", "trial_summary_stats.csv")
    summary_df.to_csv(summary_path, index=False)
    
    print("\n📊 Trial Summary:")
    print(summary_df.to_string(index=False))
    print(f"\n✅ Data exported to: {os.path.dirname(summary_path)}")

if __name__ == "__main__":
    run_drug_simulation()
