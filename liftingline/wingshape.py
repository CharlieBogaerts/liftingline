from dataclasses import dataclass
import numpy as np


@dataclass
class ControlSurface:
    spans: np.ndarray  # Absolute spanwise positions [y_start, ..., y_end]
    delta_alpha0: np.ndarray  # Local d(alpha_0)/d(delta) shift at each spanwise point
    symmetric: bool = False  # False = Aileron mode (-left, +right), True = Flap mode (+left, +right)

    def __post_init__(self):
        self.spans = np.asarray(self.spans, dtype=float)
        self.delta_alpha0 = np.asarray(self.delta_alpha0, dtype=float)


class WingShape:

    def __init__(
        self,
        airfoil_spans,
        airfoil_chords,
        airfoil_alphas,
        controls: list[ControlSurface] = None,
    ):
        # Sort and store wing main planform arrays
        sort_idx = np.argsort(airfoil_spans)
        self.spans = np.asarray(airfoil_spans, dtype=float)[sort_idx]
        self.chords = np.asarray(airfoil_chords, dtype=float)[sort_idx]
        self.alphas = np.asarray(airfoil_alphas, dtype=float)[sort_idx]

        self.span = self.spans.max() * 2.0
        self.controls = controls if controls is not None else []
        self.nr_of_controls = len(self.controls)

    def chord(self, y):
        """Linearly interpolates chord along spanwise positions y."""
        return np.interp(
            np.abs(y), self.spans, self.chords, left=0.0, right=0.0
        )

    def alpha_geo(self, y):
        """Linearly interpolates baseline geometric angle of attack along y."""
        return np.interp(
            np.abs(y), self.spans, self.alphas, left=0.0, right=0.0
        )

    def alpha_control(self, y, cont_nr: int):
        """Calculates the local alpha_0 shift distribution for 1 radian or unit deflection.

        - Negative y (Left Wing): Deflects according to symmetry rule.
        - Positive y (Right Wing): Standard deflection direction.
        """
        if cont_nr >= self.nr_of_controls:
            return np.zeros_like(y, dtype=float)

        cs = self.controls[cont_nr]
        y_arr = np.asarray(y, dtype=float)
        y_abs = np.abs(y_arr)

        # 1. Linearly interpolate delta_alpha0 within the control surface boundaries
        alpha_shift = np.interp(y_abs, cs.spans, cs.delta_alpha0, left=0.0, right=0.0)

        # 2. Hard mask to 0 outside the control surface span range
        out_of_bounds = (y_abs < cs.spans.min()) | (y_abs > cs.spans.max())
        alpha_shift[out_of_bounds] = 0.0

        # 3. Apply wing sign conventions
        if cs.symmetric:
            # Flap behavior: Symmetric (+1 left, +1 right)
            return alpha_shift
        else:
            # Aileron behavior: Anti-symmetric (+1 left, -1 right)
            return np.where(y_arr >= 0, -alpha_shift, alpha_shift)