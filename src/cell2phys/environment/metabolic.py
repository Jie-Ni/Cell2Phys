import torch
import torch.nn as nn


class MetabolicEnv(nn.Module):
    """
    Cell2Phys Core Module: Metabolic Environment.

    Implements the Bergman Minimal Model for Glucose-Insulin Dynamics.
    This module acts as the physiological constraint layer, ensuring that
    system dynamics adhere to mass conservation and kinetic laws.
    """

    def __init__(self, p1=0.02, p2=0.025, p3=0.00001, n=0.09, vol=120.0, gb=90.0, ib=10.0):
        super().__init__()

        # Register physiological parameters as constants (non-trainable buffers)
        # p1: Glucose effectiveness (1/min)
        self.register_buffer('p1', torch.tensor(p1))
        # p2: Rate of decrease of insulin action (1/min)
        self.register_buffer('p2', torch.tensor(p2))
        # p3: Insulin sensitivity
        self.register_buffer('p3', torch.tensor(p3))
        # n: Insulin clearance rate (1/min)
        self.register_buffer('n', torch.tensor(n))
        # vol: Distribution volume (dL)
        self.register_buffer('vol', torch.tensor(vol))
        # Gb: Basal glucose level (mg/dL)
        self.register_buffer('Gb', torch.tensor(gb))
        # Ib: Basal insulin level (uU/mL)
        self.register_buffer('Ib', torch.tensor(ib))

        # Interface: Total insulin secretion rate from all agents
        self.secretion_rate = 0.0

    def forward(self, t, state):
        """
        Computes the derivatives for the system of differential equations.

        Args:
            t (float): Current time.
            state (Tensor): Current state vector [Glucose, Remote_Insulin, Plasma_Insulin].

        Returns:
            Tensor: Derivatives [dG/dt, dX/dt, dI/dt].
        """
        G, X, I = state[0], state[1], state[2]

        # 1. dG/dt: Glucose Dynamics
        # Glucose is removed by insulin-independent (p1) and insulin-dependent (X) mechanisms.
        dG = -(self.p1 + X) * G + self.p1 * self.Gb

        # 2. dX/dt: Insulin Remote Action Dynamics
        # Represents the delayed effect of insulin on glucose disappearance.
        dX = -self.p2 * X + self.p3 * (I - self.Ib)

        # 3. dI/dt: Plasma Insulin Dynamics
        # Balance between endogenous secretion (from agents) and clearance (n).
        dI = -self.n * (I - self.Ib) + (self.secretion_rate / self.vol)

        return torch.stack([dG, dX, dI])