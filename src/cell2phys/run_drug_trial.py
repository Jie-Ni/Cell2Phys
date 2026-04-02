"""
Cell2Phys -- In Silico Clinical Trial Runner

Integrates:
  - LLM-driven adaptive parameterization (called at periodic intervals)
  - Physics-constrained ODE integration (LSODA via odeint)
  - Adaptive Semantic Caching (ASC)
  - Pharmacology (Hill-Langmuir drug effect)
  - Statistical analysis (two-sample Welch's t-test)
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import warnings
from scipy.integrate import odeint
from scipy import stats

from src.cell2phys.config import Config
from src.cell2phys.agents.cell import CellAgent
from src.cell2phys.agents.liver import LiverAgent
from src.cell2phys.pharmacology.mechanisms import PharmacologyEngine
from src.cell2phys.physics.dynamics import system_dynamics

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_agents(target: int):
    """Load agents from the single-cell atlas."""
    data_path = os.path.join(Config.PROJECT_ROOT, "data", "bastidas_ponce_2019.h5ad")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Dataset not found at {data_path}.\n"
            "Please download 'bastidas_ponce_2019.h5ad' from Zenodo "
            "(DOI: 10.5281/zenodo.18331267) and place it in the data/ directory."
        )

    adata = sc.read_h5ad(data_path)
    p_df = adata[adata.obs["tissue"] == "Pancreas"].to_df()
    l_df = adata[adata.obs["tissue"] == "Liver"].to_df()
    n_p = min(len(p_df), target)
    n_l = min(len(l_df), target)
    print(f"   Loaded atlas: {n_p} pancreas cells, {n_l} liver cells")

    return (
        [CellAgent(i, p_df.iloc[i]) for i in range(n_p)],
        [LiverAgent(i, l_df.iloc[i]) for i in range(n_l)],
    )


# ---------------------------------------------------------------------------
# Segmented ODE with periodic LLM re-parameterization
# ---------------------------------------------------------------------------

def _simulate_arm(
    pancreas_agents,
    liver_agents,
    env_params,
    drug_conc: float,
    drug_params: dict,
    t_end: float = 200.0,
    dt: float = 1.0,
    llm_interval: float = 20.0,
):
    """
    Run ODE simulation for one trial arm.
    LLM re-parameterization occurs every ``llm_interval`` minutes.
    """
    n_agents = len(pancreas_agents)

    # Drug effect on liver
    effect = PharmacologyEngine.hill_langmuir(
        drug_conc, drug_params["E_max"], drug_params["EC50"], drug_params["n"]
    )
    for agent in liver_agents:
        agent.drug_modifier = 1.0 + effect

    # Initial state
    G0, I0 = 200.0, 10.0
    y = np.array([G0, I0] + [1.0] * n_agents, dtype=float)

    all_G, all_I, all_t = [G0], [I0], [0.0]
    current_time = 0.0

    while current_time < t_end - 1e-6:
        seg_end = min(current_time + llm_interval, t_end)
        t_seg = np.arange(current_time, seg_end + dt / 2, dt)
        if len(t_seg) < 2:
            break

        # LLM re-parameterization at segment start
        glucose_now = float(np.clip(y[0], Config.GLUCOSE_MIN, Config.GLUCOSE_MAX))
        for agent in pancreas_agents:
            agent.update_parameters(glucose_now)
        for agent in liver_agents:
            agent.adapt_metabolism(glucose_now)

        # ODE integration (LSODA)
        sol = odeint(
            system_dynamics, y, t_seg,
            args=(pancreas_agents, liver_agents, env_params),
            mxstep=10000,
        )

        # Enforce physiological bounds
        sol[:, 0] = np.clip(sol[:, 0], Config.GLUCOSE_MIN, Config.GLUCOSE_MAX)
        sol[:, 1] = np.clip(sol[:, 1], Config.INSULIN_MIN, Config.INSULIN_MAX)
        for i in range(n_agents):
            sol[:, 2 + i] = np.clip(sol[:, 2 + i], Config.BETA_MASS_MIN, Config.BETA_MASS_MAX)

        all_G.extend(sol[1:, 0].tolist())
        all_I.extend(sol[1:, 1].tolist())
        all_t.extend(t_seg[1:].tolist())

        y = sol[-1].copy()
        current_time = seg_end

    return np.array(all_t), np.array(all_G), np.array(all_I)


# ---------------------------------------------------------------------------
# Statistical comparison
# ---------------------------------------------------------------------------

def _compare_arms(auc_a: np.ndarray, auc_b: np.ndarray):
    """Welch's t-test between two trial arms."""
    t_stat, p_val = stats.ttest_ind(auc_a, auc_b, equal_var=False)
    pooled_std = np.sqrt((np.var(auc_a) + np.var(auc_b)) / 2)
    cohens_d = (np.mean(auc_a) - np.mean(auc_b)) / (pooled_std + 1e-12)
    return {"t_stat": t_stat, "p_value": p_val, "cohens_d": cohens_d}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_drug_simulation():
    print("[Cell2Phys] Starting In Silico Clinical Trial...")

    target = Config.N_PATIENTS
    pancreas_agents, liver_agents = _load_agents(target)
    print(f"   Cohort: {len(pancreas_agents)} beta-cells, {len(liver_agents)} hepatocytes")

    arms = [
        {"name": "Placebo", "drug_conc": 0.0},
        {"name": "Metformin", "drug_conc": 150.0},
    ]
    drug_params = {"E_max": 2.0, "EC50": 100.0, "n": 1.5}
    env_params = {"glucose_influx": 2.0, "volume_distribution": 10.0}

    dt = Config.DT
    t_end = Config.TOTAL_TIME
    export_rows = []

    out_dir = os.path.join(Config.PROJECT_ROOT, "results", "data")
    os.makedirs(out_dir, exist_ok=True)

    print("\n[Simulation] Running LSODA + ASC + LLM ...\n")

    for arm in arms:
        print(f"  Arm: {arm['name']} (drug={arm['drug_conc']} uM)")

        t_arr, G_arr, I_arr = _simulate_arm(
            pancreas_agents, liver_agents, env_params,
            drug_conc=arm["drug_conc"], drug_params=drug_params,
            t_end=t_end, dt=dt, llm_interval=20.0,
        )

        _trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        auc = float(_trapz(G_arr, t_arr))
        tir = float(np.sum((G_arr >= 70) & (G_arr <= 140)) / len(G_arr) * 100)

        row = {
            "Arm": arm["name"], "Drug_Conc_uM": arm["drug_conc"],
            "Start_Glucose": G_arr[0], "End_Glucose": G_arr[-1],
            "Min_Glucose": np.min(G_arr), "Max_Glucose": np.max(G_arr),
            "AUC_Glucose": auc, "Time_in_Range_Pct": tir,
        }
        export_rows.append(row)
        print(f"    AUC={auc:.1f}  TIR={tir:.1f}%  End_G={G_arr[-1]:.1f}")

        ts = pd.DataFrame({"Time": t_arr, "Glucose": G_arr, "Insulin": I_arr})
        ts.to_csv(os.path.join(out_dir, f"series_{arm['name']}.csv"), index=False)

    summary = pd.DataFrame(export_rows)
    summary.to_csv(os.path.join(out_dir, "trial_summary_stats.csv"), index=False)

    print("\n[Results]")
    print(summary.to_string(index=False))

    # Persist ASC cache
    from src.cell2phys.utils.asc_engine import asc_engine
    asc_engine.save()

    print(f"\nData exported to: {out_dir}")
    print("[Cell2Phys] Done.")


if __name__ == "__main__":
    run_drug_simulation()
