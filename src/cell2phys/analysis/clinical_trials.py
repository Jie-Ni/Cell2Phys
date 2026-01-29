import numpy as np
from scipy import stats

class ClinicalTrialAnalyzer:
    """
    Implements statistical rigorousness for In Silico Trials.
    """
    
    @staticmethod
    def calculate_power(n_per_group: int, mean_diff: float, pooled_std: float, alpha: float = 0.05) -> float:
        """
        Calculates the statistical power (1 - Beta) for a two-sample t-test 
        using the non-central t-distribution.
        
        Formula: Power = Phi( -z_{1-alpha/2} + |delta| / (sigma * sqrt(2/N)) )
        (Approximation using standard normal Z-test for large N)
        """
        if pooled_std == 0:
            return 1.0
            
        effect_size = abs(mean_diff)
        standard_error = pooled_std * np.sqrt(2 / n_per_group)
        
        z_crit = stats.norm.ppf(1 - alpha / 2)
        z_score = (effect_size / standard_error) - z_crit
        
        power = stats.norm.cdf(z_score)
        return power

    @staticmethod
    def stratify_cohort(agents: list, criterion: str = "variance"):
        """
        Stratifies the virtual cohort into "High Variance" (Standard)
        and "Low Variance" (PhysioLLM Selected) groups.
        """
        # Stratify patient cohort based on baseline variance metrics
        
        sorted_agents = sorted(agents, key=lambda a: getattr(a, 'expression_variance', 1.0))
        
        mid_point = len(sorted_agents) // 2
        low_var_group = sorted_agents[:mid_point]
        high_var_group = sorted_agents[mid_point:]
        
        return low_var_group, high_var_group
