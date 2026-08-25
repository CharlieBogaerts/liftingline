from dataclasses import dataclass
import numpy as np


@dataclass
class LiftingLineSolution:
    """Stores the aerodynamic state, spanwise distributions, integrated forces,

    and sensitivity derivatives produced by a lifting-line solver evaluation.

    Attributes
    ----------
    v_inf : float
        Free-stream air velocity magnitude [m/s].
    p : float
        Body roll rate [rad/s] (positive right wing down).
    r : float
        Body yaw rate [rad/s] (positive nose right).
    deltas : np.ndarray
        Array of shape `(N_controls,)` containing deflection angles [rad]
        for each control surface.

    y : np.ndarray
        Array of shape `(N_span,)` containing spanwise coordinate points [m],
        ranging from `-b/2` to `+b/2`.
    theta : np.ndarray
        Array of shape `(N_span,)` containing transformed angular coordinates [rad],
        where `y = -(b/2) * cos(theta)` with `theta` in `[0, pi]`.
    chords : np.ndarray
        Array of shape `(N_span,)` containing local chord lengths [m] along the span.

    alpha_geo : np.ndarray
        Array of shape `(N_span,)` containing total local geometric angle of attack [rad]
        including geometric pitch, twist, control deflections, and motion rate effects.
    alpha_eff : np.ndarray
        Array of shape `(N_span,)` containing effective angle of attack [rad]
        after subtracting spanwise downwash.
    gamma : np.ndarray
        Array of shape `(N_span,)` containing local circulation strength [m^2/s].
    L : np.ndarray
        Array of shape `(N_span,)` containing local sectional lift force per unit span [N/m].
    D : np.ndarray
        Array of shape `(N_span,)` containing local sectional induced drag force per unit span [N/m].
    fourier_coeffs : np.ndarray
        Array of shape `(N_coeffs,)` containing non-dimensional Fourier sine series
        coefficients `[A_1, A_2, ..., A_N]` describing the circulation distribution.

    L_total : float
        Total integrated wing lift force [N].
    D_total : float
        Total integrated wing induced drag force [N].
    Mx : float
        Total aerodynamic rolling moment [N*m] computed about the origin `y = 0`.
    Mz : float
        Total aerodynamic yawing moment [N*m] computed about the origin `y = 0`.

    stability_derivs : np.ndarray
        Array of shape `(2, 2)` containing linear rate sensitivity derivatives:
        `[[dMx/dp, dMx/dr], [dMz/dp, dMz/dr]]` in units of `[N*m / (rad/s)]`.
    control_derivs : np.ndarray
        Array of shape `(2, N_controls)` containing control effectiveness derivatives:
        `[[dMx/d_delta_1, ..., dMx/d_delta_N], [dMz/d_delta_1, ..., dMz/d_delta_N]]`
        in units of `[N*m / rad]`.
    """

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