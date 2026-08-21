from dataclasses import dataclass
import numpy as np


@dataclass
class LiftingLineSolution:
    """Store the aerodynamic state and derivatives produced by a lifting-line solve."""

    # Operating state
    v_inf: float
    p: float
    r: float
    deltas: np.ndarray

    # Grid and Geometry
    y: np.ndarray
    theta: np.ndarray
    chords: np.ndarray

    # Spanwise Distributions
    alpha_geo: np.ndarray
    alpha_eff: np.ndarray
    gamma: np.ndarray
    L: np.ndarray
    D: np.ndarray
    fourier_coeffs: np.ndarray

    # Integrated Forces and Moments
    L_total: float
    D_total: float
    Mx: float
    Mz: float

    # Derivatives
    stability_derivs: np.ndarray  # 2x2 matrix: [[dMx/dp, dMx/dr], [dMz/dp, dMz/dr]]
    control_derivs: np.ndarray    # 2xN matrix: [[dMx/d_delta_i], [dMz/d_delta_i]]

    @property
    def oswald_efficiency(self) -> float:
        """Return the Oswald efficiency factor for the solved circulation distribution."""
        A = self.fourier_coeffs
        A1 = A[0]
        higher_order_terms = np.sum(np.arange(2, len(A) + 1) * A[1:] ** 2)
        return 1.0 / (1.0 + (higher_order_terms / A1**2))