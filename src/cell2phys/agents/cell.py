import numpy as np
from src.cell2phys.utils.llm_client import brain
try:
    from src.cell2phys.utils.rag_engine import RAGController
    shared_rag = RAGController()
    RAG_AVAILABLE = True
except Exception as e:
    print(f"⚠️ RAG Engine fail: {e}")
    RAG_AVAILABLE = False


class CellAgent:
    """
    Agent representing a pancreatic beta-cell.
    Dynamics are governed by ODEs with parameters modulated by LLM reasoning.
    """

    def __init__(self, agent_id, real_expression_series):
        self.id = agent_id
        self.genes = real_expression_series
        
        # True biological initialization
        self.ins_expression = self._find_gene_value(['INS', 'Insulin', 'INSULIN', 'Gene_10'])
        self.current_context = "Basal state"
        
        # Personalized LLM Context (Genotype)
        self.genotype_summary = f"High expression of INS ({self.ins_expression:.1f})."

    def _find_gene_value(self, candidates):
        for gene in candidates:
            if gene in self.genes.index:
                return float(self.genes[gene])
        return 10.0

    def update_parameters(self, glucose_level):
        """
        Updates kinetic parameters based on current state and retrieved context.
        """
        query = f"Mechanism of beta-cell insulin secretion at {glucose_level:.1f} mg/dL glucose."
        
        if RAG_AVAILABLE:
            knowledge = shared_rag.retrieve(query, k=1)
            self.current_context = knowledge
        else:
            knowledge = "Glucose stimulates insulin secretion."

        system_prompt = f"""
        You are a pancreatic beta-cell with this genotype: {self.genotype_summary}.
        Your goal is to maintain homeostasis (90 mg/dL).
        
        Current Knowledge Context:
        {knowledge}
        
        Task: 
        Calculate the 'Regulation Factor' (a multiplier for insulin transcription).
        - If glucose is high (>120) and context suggests compensation, output > 1.0.
        - If glucose is toxic (>250) and context suggests suppression, output < 1.0.
        - If glucose is basal (90), output 1.0.
        
        Return ONLY the numeric float value.
        """
        
        user_prompt = f"Current Glucose: {glucose_level:.1f} mg/dL."
        
        # The Brain decides the factor (Cached)
        self.regulation_factor = brain.think_and_decide(system_prompt, user_prompt)

    def calculate_secretion_rate(self, glucose_level):
        """
        Calculates insulin secretion rate (dI/dt) using Hill kinetics.
        """
        # Hill Kinetics
        # Vmax depends on INS expression and recent regulation
        Vmax = 0.05 * self.ins_expression * getattr(self, 'regulation_factor', 1.0)
        Km = 90.0
        h = 2.0
        
        # Hill function: Vmax * G^h / (Km^h + G^h)
        rate = Vmax * (glucose_level**h) / (Km**h + glucose_level**h)
        return rate

    def calculate_plasticity_rate(self, glucose_level, beta_mass):
        """
        [Physics Step]
        Returns d(BetaMass)/dt based on Glucotoxicity Model.
        """
        # Proliferation (quadratic)
        k_prol = 0.0001
        rate_prol = k_prol * (glucose_level**2 / (150**2 + glucose_level**2))
        
        # Apoptosis (quartic - sharp threshold)
        k_death = 0.0002
        rate_death = k_death * (glucose_level**4 / (250**4 + glucose_level**4))
        
        return beta_mass * (rate_prol - rate_death)
