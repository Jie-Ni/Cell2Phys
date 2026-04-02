import numpy as np
from src.cell2phys.config import Config


class CellAgent:
    """
    Pancreatic beta-cell agent.
    Kinetic parameters are modulated by LLM reasoning (via PhysioBrain).
    """

    def __init__(self, agent_id, real_expression_series):
        self.id = agent_id
        self.genes = real_expression_series

        # Gene expression initialization
        self.ins_expression = self._find_gene_value(["INS", "Insulin", "INSULIN"])
        self.gcg_expression = self._find_gene_value(["GCG", "Glucagon"])
        self.glut2_expression = self._find_gene_value(["SLC2A2", "GLUT2"])

        # LLM-derived parameter (updated dynamically)
        self.regulation_factor = 1.0

        # Expression-derived Hill parameters (M4 fix)
        self.Km = 80.0 + 20.0 * np.clip(self.glut2_expression / 10.0, 0.5, 2.0)
        self.h = 1.5 + 0.5 * np.clip(self.ins_expression / 15.0, 0.5, 1.5)

        # Genotype summary for LLM context
        self.genotype_summary = (
            f"INS={self.ins_expression:.1f}, GCG={self.gcg_expression:.1f}, "
            f"GLUT2={self.glut2_expression:.1f}"
        )

    def _find_gene_value(self, candidates):
        for gene in candidates:
            if gene in self.genes.index:
                val = float(self.genes[gene])
                if np.isfinite(val):
                    return val
        return np.random.uniform(5.0, 20.0)  # biologically plausible fallback

    def update_parameters(self, glucose_level: float):
        """Update regulation_factor via LLM (or demo heuristic)."""
        from src.cell2phys.utils.llm_client import get_brain

        brain = get_brain()

        system_prompt = (
            f"You are a pancreatic beta-cell with genotype: {self.genotype_summary}. "
            f"Maintain glucose homeostasis (~90 mg/dL). "
            f"Return a Regulation Factor (float): >1.0 for compensation, <1.0 for glucotoxic suppression, 1.0 for basal."
        )
        user_prompt = f"Current Glucose: {glucose_level:.1f} mg/dL."

        self.regulation_factor = brain.think_and_decide(
            system_prompt,
            user_prompt,
            low=Config.REGULATION_FACTOR_MIN,
            high=Config.REGULATION_FACTOR_MAX,
        )

    def calculate_secretion_rate(self, glucose_level: float) -> float:
        """Insulin secretion rate via Hill kinetics."""
        G = max(glucose_level, 0.0)
        Vmax = 0.05 * self.ins_expression * self.regulation_factor
        rate = Vmax * (G ** self.h) / (self.Km ** self.h + G ** self.h + 1e-12)
        return max(rate, 0.0)

    def calculate_plasticity_rate(self, glucose_level: float, beta_mass: float) -> float:
        """Beta-cell mass dynamics (glucotoxicity model)."""
        G = max(glucose_level, 0.0)
        bm = max(beta_mass, Config.BETA_MASS_MIN)

        # Proliferation (quadratic Hill, threshold ~150 mg/dL)
        k_prol = 0.001
        rate_prol = k_prol * (G ** 2) / (150.0 ** 2 + G ** 2)

        # Apoptosis (quartic Hill, sharp threshold ~250 mg/dL)
        k_death = 0.002
        rate_death = k_death * (G ** 4) / (250.0 ** 4 + G ** 4)

        return bm * (rate_prol - rate_death)
