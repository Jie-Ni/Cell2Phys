import numpy as np
from src.cell2phys.config import Config


class LiverAgent:
    """
    Hepatocyte agent.  Receptor kinetics modulated by LLM-derived insulin sensitivity.
    """

    def __init__(self, agent_id, real_expression_series=None):
        self.id = agent_id

        if real_expression_series is not None:
            self.insr_expression = self._find_gene(
                real_expression_series, ["INSR", "Insr", "Insulin_Receptor"]
            )
        else:
            self.insr_expression = 15.0

        self.hgp_base_rate = 2.0
        self.sensitivity_factor = 1.0       # LLM-derived
        self.drug_modifier = 1.0            # set externally by pharmacology

    def _find_gene(self, genes, candidates):
        for g in candidates:
            if g in genes.index:
                val = float(genes[g])
                if np.isfinite(val):
                    return val
        return np.random.uniform(10.0, 20.0)

    def adapt_metabolism(self, avg_glucose_history: float):
        """LLM-driven insulin sensitivity adaptation based on glucose history."""
        from src.cell2phys.utils.llm_client import get_brain

        brain = get_brain()
        system_prompt = (
            "You are a Hepatocyte. Determine insulin sensitivity factor: "
            "0.1 = severely resistant, 1.0 = healthy, up to 2.0 = very sensitive."
        )
        user_prompt = (
            f"Average glucose: {avg_glucose_history:.1f} mg/dL. "
            f"INSR expression: {self.insr_expression:.1f}."
        )
        self.sensitivity_factor = brain.think_and_decide(
            system_prompt,
            user_prompt,
            low=Config.SENSITIVITY_FACTOR_MIN,
            high=Config.SENSITIVITY_FACTOR_MAX,
        )

    def calculate_hgp_rate(self, plasma_insulin: float, glucose_level: float) -> float:
        """
        Net hepatic glucose flux.  Positive = release, Negative = uptake.
        Drug effect is applied to sensitivity_factor multiplicatively.
        """
        I = max(plasma_insulin, 0.0)
        G = max(glucose_level, 0.0)

        effective_sensitivity = self.sensitivity_factor * self.drug_modifier

        # Receptor binding (Michaelis-Menten style)
        Kd = 20.0
        bound = (self.insr_expression * effective_sensitivity) * I / (Kd + I + 1e-6)

        # Signal transduction -> HGP inhibition
        inhibition = min(1.0, bound / (bound + 5.0))
        current_hgp = self.hgp_base_rate * (1.0 - inhibition)

        # Hepatic glucose uptake (mass-action)
        uptake = 0.008 * G

        return current_hgp - uptake
