import numpy as np
from src.cell2phys.utils.llm_client import brain

class LiverAgent:
    """
    Agent representing a Hepatocyte (Liver Cell).
    Receptor kinetics are modulated by predictive metabolic adaptation.
    """

    def __init__(self, agent_id, real_expression_series=None):
        self.id = agent_id
        
        # 1. Receptor Expression (Genotype)
        if real_expression_series is not None:
            self.insr_expression = self._find_gene(real_expression_series, ['INSR', 'Insr', 'Insulin_Receptor'])
        else:
            self.insr_expression = 15.0

        # 2. Metabolic State (Phenotype)
        self.hgp_base_rate = 2.0
        
        # 3. LLM-Derived Parameters (Plasticity)
        # The LLM determines the "Insulin Sensitivity Factor" based on history
        self.sensitivity_factor = 1.0 

    def _find_gene(self, genes, candidates):
        for g in candidates:
            if g in genes.index: return float(genes[g])
        return 15.0

    def adapt_metabolism(self, avg_glucose_history):
        """
        Determines insulin sensitivity based on chronic glucose exposure history.
        """
        system_prompt = "You are a Hepatocyte. Determine insulin sensitivity (0.1=Resistant, 1.0=Healthy)."
        user_prompt = f"Average glucose over last 24h was {avg_glucose_history:.1f} mg/dL. INSR expression is {self.insr_expression:.1f}."
        
        self.sensitivity_factor = brain.think_and_decide(system_prompt, user_prompt)

    def calculate_hgp_rate(self, plasma_insulin, glucose_level):
        """
        Calculates Net Glucose Flux (HGP - Uptake).
        Positive = Release; Negative = Uptake.
        """
        # --- 1. Receptor Binding ---
        Kd = 20.0
        # Bound receptors are effective only if "sensitivity_factor" is high
        bound_receptors = (self.insr_expression * self.sensitivity_factor) * plasma_insulin / (Kd + plasma_insulin + 1e-6)

        # --- 2. Signal Transduction (Inhibition of HGP) ---
        inhibition_factor = bound_receptors / (bound_receptors + 5.0)
        inhibition_factor = min(1.0, inhibition_factor)

        current_hgp = self.hgp_base_rate * (1.0 - inhibition_factor)

        # --- 3. Uptake (Mass Action) ---
        # Liver uptake is proportional to glucose concentration
        uptake_rate = 0.008 * glucose_level

        # --- 4. Net Flux ---
        # HGP adds glucose, Uptake removes it
        net_flux = current_hgp - uptake_rate
        
        return net_flux
