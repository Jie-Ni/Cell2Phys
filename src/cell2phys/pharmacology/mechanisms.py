import numpy as np

class PharmacologyEngine:
    """
    Implements the mathematical models for drug-response and toxicity
    as described in the Cell2Phys manuscript.
    """
    
    @staticmethod
    def hill_langmuir(concentration: float, e_max: float, ec50: float, n: float = 1.0) -> float:
        """
        Calculates the drug effect using the Hill-Langmuir Isotherm.
        
        Equation: E(C) = E_max * (C^n / (EC50^n + C^n))
        
        Args:
            concentration (C): Drug concentration
            e_max (E_max): Maximum achievable effect (efficacy)
            ec50 (EC50): Concentration at half-maximal effect (potency)
            n (n): Hill coefficient (cooperativity)
        """
        if concentration <= 0:
            return 0.0
            
        c_n = concentration ** n
        ec50_n = ec50 ** n
        
        return e_max * (c_n / (ec50_n + c_n))

    @staticmethod
    def calculate_toxicity_score(apoptosis_rates_drug: list, apoptosis_rates_basal: list, dt: float) -> float:
        """
        Calculates the cumulative toxicity score (ToxScore).
        
        Equation: Integral(0->24h) of max(0, dBeta/dt_drug - dBeta/dt_basal) dt
        """
        tox_score = 0.0
        
        # Ensure lists are same length
        steps = min(len(apoptosis_rates_drug), len(apoptosis_rates_basal))
        
        for i in range(steps):
            rate_drug = apoptosis_rates_drug[i]
            rate_basal = apoptosis_rates_basal[i]
            
            # Only excess apoptosis counts as toxicity
            excess_death = max(0.0, rate_drug - rate_basal)
            
            tox_score += excess_death * dt
            
        return tox_score

    @staticmethod
    def predict_parameters_via_llm(brain, drug_name: str, gene_context: str):
        """
        Uses the LLM to predict EC50/E_max based on mechanism of action.
        """
        prompt = f"""
        Predict the EC50 (in uM) for the drug '{drug_name}' acting on a cell with this genetic profile: {gene_context}.
        Return ONLY the numeric value.
        """
        ec50 = brain.think_and_decide("You are a pharmacologist.", prompt)
        return ec50
