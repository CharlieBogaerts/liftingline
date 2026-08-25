import numpy as np
from liftingline.control_surface import ControlSurface



class WingShape:
    def __init__(
        self,
        airfoil_spans,
        airfoil_chords,
        airfoil_alphas,
        controls: list[ControlSurface] = None,
    ):
        # Convert raw inputs to NumPy arrays first
        spans_raw = np.asarray(airfoil_spans, dtype=float)
        chords_raw = np.asarray(airfoil_chords, dtype=float)
        alphas_raw = np.asarray(airfoil_alphas, dtype=float)

        # Validate raw NumPy arrays
        invalid_spans = np.where(spans_raw < 0.0)[0]
        if invalid_spans.size > 0:
            raise ValueError(
                f"Span station positions must be non-negative (y >= 0).\n"
                f"  Invalid Indices: {invalid_spans.tolist()}\n"
                f"  Invalid Values:  {spans_raw[invalid_spans].tolist()}"
            )

        invalid_chords = np.where(chords_raw <= 0.0)[0]
        if invalid_chords.size > 0:
            raise ValueError(
                f"All airfoil chords must be strictly positive (c > 0).\n"
                f"  Invalid Indices: {invalid_chords.tolist()}\n"
                f"  Invalid Values:  {chords_raw[invalid_chords].tolist()}"
            )

        # Sort arrays by span location
        sort_idx = np.argsort(spans_raw)
        self.spans = spans_raw[sort_idx]
        self.chords = chords_raw[sort_idx]
        self.alphas = alphas_raw[sort_idx]
        
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

    def surface_area(self):
        """Calculates the total wing surface area using trapezoidal integration."""
        return np.trapezoid(self.chords, self.spans) * 2.0

    def aspect_ratio(self):
        """Calculates the wing aspect ratio."""
        return self.span**2 / self.surface_area()