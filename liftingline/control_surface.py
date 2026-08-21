from dataclasses import dataclass
import numpy as np


@dataclass
class ControlSurface:
    spans: np.ndarray           # Absolute spanwise positions [y_start, ..., y_end]
    delta_alpha0: np.ndarray    # Local d(alpha_0)/d(delta) shift at each spanwise point
    symmetric: bool = False     # False = Aileron mode (-left, +right), True = Flap mode (+left, +right)

    def __post_init__(self):
        self.spans = np.asarray(self.spans, dtype=float)
        self.delta_alpha0 = np.asarray(self.delta_alpha0, dtype=float)