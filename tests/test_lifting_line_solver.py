import sys
import unittest
import numpy as np

# Adjust module path if necessary
sys.path.append("/home/charlie/Documents/Python3/MyPackages/liftingline")

import liftingline as ll




class TestEllipticalWing(unittest.TestCase):
    """Test suite comparing numerical results against Prandtl's classical 

    analytical solutions for an ideal elliptical wing.
    """

    def setUp(self):
        """Builds the elliptical wing and solver instance before each test."""
        self.span = 10.0
        self.root_chord = 2.0
        self.alpha_deg = 5.0
        self.alpha_rad = np.radians(self.alpha_deg)
        self.v_inf = 10.0
        self.rho = 1.225
        self.q_inf = 0.5 * self.rho * self.v_inf**2

        # Semi-span coordinates from 0 to b/2
        n_points = 201
        spans = np.linspace(0.0, self.span / 2.0, n_points)

        # Elliptical chord distribution: c(y) = c0 * sqrt(1 - (2y/b)^2)
        eta = spans / (self.span / 2.0)
        chords = self.root_chord * np.sqrt(np.maximum(0.0, 1.0 - eta**2))
        chords = np.maximum(chords, 0.0001)  # Ensure minimum chord length to avoid singularities
        alphas = np.full_like(spans, self.alpha_rad)

        self.wing = ll.WingShape(spans, chords, alphas)
        self.solver = ll.LiftingLineSolver(self.wing, nr_of_coefs=80)

    def test_lift_and_drag_coefficients(self):
        """Validates C_L and C_Di against Prandtl's analytical formulas:

        C_L = (2 * pi * alpha) / (1 + 2/AR)
        C_Di = C_L^2 / (pi * AR)
        """
        sol = self.solver.solve(v_inf=self.v_inf, rho=self.rho)

        S = self.wing.surface_area()
        AR = self.wing.aspect_ratio()

        # Analytical solutions
        CL_analytical = (2.0 * np.pi * self.alpha_rad) / (1.0 + 2.0 / AR)
        CDi_analytical = (CL_analytical**2) / (np.pi * AR)

        # Numerical solutions
        CL_numerical = sol.L_total / (self.q_inf * S)
        CDi_numerical = sol.D_total / (self.q_inf * S)

        np.testing.assert_allclose(
            CL_numerical,
            CL_analytical,
            rtol=1e-2,
            err_msg="Lift coefficient CL deviated from analytical solution.",
        )
        np.testing.assert_allclose(
            CDi_numerical,
            CDi_analytical,
            rtol=1e-2,
            err_msg="Induced drag coefficient CDi deviated from analytical solution.",
        )

    def test_fourier_coefficients(self):
        """Verifies that only the A1 Fourier term is non-zero.

        For an elliptical wing, higher harmonic terms (A3, A5, ...) must vanish.
        """
        sol = self.solver.solve(v_inf=self.v_inf)
        A = sol.fourier_coeffs

        # A[0] is A1; A[1:] represent higher-order terms
        np.testing.assert_allclose(
            A[1:],
            0.0,
            atol=1e-3,
            err_msg="Higher-order Fourier coefficients should be zero for an elliptical wing.",
        )

    def test_3d_lift_curve_slope(self):
        """Validates 3D lift curve slope dCL/dalpha against Prandtl's exact formula:

        C_L_alpha = a0 / (1 + a0 / (pi * AR))
        """
        sol = self.solver.solve(v_inf=self.v_inf, rho=self.rho)

        S = self.wing.surface_area()
        AR = self.wing.aspect_ratio()
        a0 = 2.0 * np.pi  # 2D thin airfoil lift curve slope

        # Numerical lift-curve slope
        CL_numerical = sol.L_total / (self.q_inf * S)
        CL_alpha_numerical = CL_numerical / self.alpha_rad

        # Exact analytical lift-curve slope for an elliptical wing
        CL_alpha_analytical = a0 / (1.0 + a0 / (np.pi * AR))

        np.testing.assert_allclose(
            CL_alpha_numerical,
            CL_alpha_analytical,
            rtol=1e-2,
            err_msg="Elliptical wing 3D lift slope deviated from exact analytical formula.",
        )

class TestRectangularWing(unittest.TestCase):

    def setUp(self):
        """Sets up an Aspect Ratio 10 rectangular wing."""
        self.span = 10.0
        self.chord = 1.0
        self.alpha_deg = 5.0
        self.alpha_rad = np.radians(self.alpha_deg)
        self.v_inf = 10.0
        self.rho = 1.225
        self.q_inf = 0.5 * self.rho * self.v_inf**2

        n_points = 201
        spans = np.linspace(0.0, self.span / 2.0, n_points)
        chords = np.full_like(spans, self.chord)
        alphas = np.full_like(spans, self.alpha_rad)

        self.wing = ll.WingShape(spans, chords, alphas)
        self.solver = ll.LiftingLineSolver(self.wing, nr_of_coefs=100)

    def test_fourier_symmetry(self):
        """Verifies that symmetric flight produces only odd Fourier coefficients (A1, A3, A5...).

        Even terms (A2, A4, A6...) must be zero.
        """
        sol = self.solver.solve(v_inf=self.v_inf)
        A = sol.fourier_coeffs

        # A[1::2] selects even modes A2, A4, A6 (0-indexed position 1, 3, 5...)
        even_coefficients = A[1::2]

        np.testing.assert_allclose(
            even_coefficients,
            0.0,
            atol=1e-12,
            err_msg="Non-zero even Fourier coefficients found in a symmetric flight condition.",
        )

    def test_downwash_distribution_shape(self):
        """Validates that effective alpha_eff decreases toward the wingtips

        because downwash (alpha_i) increases near the tips on a rectangular planform.
        """
        sol = self.solver.solve(v_inf=self.v_inf)

        # alpha_eff = alpha_geo - alpha_i
        # Center section (midspan) should have HIGHER effective alpha than near the tip
        mid_idx = len(sol.alpha_eff) // 2
        tip_idx = 10  # Near the tip boundary

        self.assertGreater(
            sol.alpha_eff[mid_idx],
            sol.alpha_eff[tip_idx],
            "Downwash should be higher near tips, causing effective alpha to drop toward the tips.",
        )

    def test_induced_drag_correction_factor_delta(self):
        """Verifies that the induced drag parameter delta = sum(n * (An/A1)^2) for n=3,5...

        falls within the known Glauert range (~0.08 to 0.09 for AR=10).
        """
        sol = self.solver.solve(v_inf=self.v_inf)
        A = sol.fourier_coeffs

        # n_llt = [1, 2, 3, 4, 5, ...]
        # Slicing from index 2 with step 2 extracts n = 3, 5, 7...
        n_odd = self.solver.n_llt_lst[2::2]
        A_odd = A[2::2]

        delta = np.sum(n_odd * (A_odd / A[0]) ** 2)

        # For an AR = 10 flat rectangular wing, Glauert's theory yields delta ~ 0.086
        self.assertGreater(
            delta, 0.075, "Induced drag correction factor delta is too small."
        )
        self.assertLess(
            delta, 0.095, "Induced drag correction factor delta is too large."
        )





if __name__ == "__main__":
    unittest.main()