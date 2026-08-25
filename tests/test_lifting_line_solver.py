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


class TestRatePerturbations(unittest.TestCase):
    def setUp(self):
        """Sets up a baseline rectangular wing solver."""
        self.span = 10.0
        self.chord = 1.0
        self.alpha_rad = np.radians(5.0)
        self.v_inf = 10.0
        self.rho = 1.225

        spans = np.linspace(0.0, self.span / 2.0, 101)
        chords = np.full_like(spans, self.chord)
        alphas = np.full_like(spans, self.alpha_rad)

        self.wing = ll.WingShape(spans, chords, alphas)
        self.solver = ll.LiftingLineSolver(self.wing, nr_of_coefs=80)

    def test_roll_rate_p_against_finite_difference(self):
        """Validates dMx/dp and dMz/dp using central finite difference."""
        p_base = 0.0
        dp = 1e-3  # Small step size for linear perturbation

        # Evaluate at p + dp and p - dp
        sol_plus = self.solver.solve(v_inf=self.v_inf, rho=self.rho, p=p_base + dp, eval_derivs=False)
        sol_minus = self.solver.solve(v_inf=self.v_inf, rho=self.rho, p=p_base - dp, eval_derivs=False)

        # Central finite difference derivatives
        dMx_dp_fd = (sol_plus.Mx - sol_minus.Mx) / (2.0 * dp)
        dMz_dp_fd = (sol_plus.Mz - sol_minus.Mz) / (2.0 * dp)

        # Analytical derivatives from solver at base point
        sol_base = self.solver.solve(v_inf=self.v_inf, rho=self.rho, p=p_base, eval_derivs=True)
        dMx_dp_ana = sol_base.stability_derivs[0, 0]  # [0, 0] is dMx/dp
        dMz_dp_ana = sol_base.stability_derivs[1, 0]  # [1, 0] is dMz/dp

        # Check analytical vs finite difference
        np.testing.assert_allclose(dMx_dp_ana, dMx_dp_fd, rtol=1e-3)
        np.testing.assert_allclose(dMz_dp_ana, dMz_dp_fd, rtol=1e-3)

        # Verify physical sign (roll damping must be negative: dMx/dp < 0)
        self.assertLess(dMx_dp_ana, 0.0, "Roll rate damping dMx/dp must be negative.")

    def test_yaw_rate_r_against_finite_difference(self):
        """Validates dMx/dr and dMz/dr using central finite difference."""
        r_base = 0.0
        dr = 1e-3

        sol_plus = self.solver.solve(v_inf=self.v_inf, rho=self.rho, r=r_base + dr, eval_derivs=False)
        sol_minus = self.solver.solve(v_inf=self.v_inf, rho=self.rho, r=r_base - dr, eval_derivs=False)

        dMx_dr_fd = (sol_plus.Mx - sol_minus.Mx) / (2.0 * dr)
        dMz_dr_fd = (sol_plus.Mz - sol_minus.Mz) / (2.0 * dr)

        sol_base = self.solver.solve(v_inf=self.v_inf, rho=self.rho, r=r_base, eval_derivs=True)
        dMx_dr_ana = sol_base.stability_derivs[0, 1]  # [0, 1] is dMx/dr
        dMz_dr_ana = sol_base.stability_derivs[1, 1]  # [1, 1] is dMz/dr

        np.testing.assert_allclose(dMx_dr_ana, dMx_dr_fd, rtol=1e-3)
        np.testing.assert_allclose(dMz_dr_ana, dMz_dr_fd, rtol=1e-3)


class TestSolveEquilibrium(unittest.TestCase):
    def setUp(self):
        """Builds a wing solver with an asymmetric aileron ControlSurface for equilibrium testing."""
        self.span = 10.0
        self.chord = 1.0
        self.alpha_rad = np.radians(5.0)
        self.v_inf = 20.0
        self.rho = 1.225

        spans = np.linspace(0.0, self.span / 2.0, 101)
        chords = np.full_like(spans, self.chord)
        alphas = np.full_like(spans, self.alpha_rad)

        # Define an aileron ControlSurface over the outboard span (y = 2.5 to 5.0)
        # symmetric=False applies anti-symmetric deflection (+left, -right)
        aileron = ll.ControlSurface(
            spans=np.array([2.5, 5.0]),
            delta_alpha0=np.array([1.0, 1.0]),  # Unit effectiveness (1 rad/rad)
            symmetric=False,
        )

        self.wing = ll.WingShape(
            airfoil_spans=spans,
            airfoil_chords=chords,
            airfoil_alphas=alphas,
            controls=[aileron],
        )
        self.solver = ll.LiftingLineSolver(self.wing, nr_of_coefs=80)

    def test_equilibrium_symmetric_baseline(self):
        """Verifies that an unperturbed symmetric wing yields p=0 and r=0 at equilibrium."""
        sol = self.solver.solve_equilibrium(v_inf=self.v_inf, rho=self.rho)

        self.assertAlmostEqual(sol.p, 0.0, places=6)
        self.assertAlmostEqual(sol.r, 0.0, places=6)
        np.testing.assert_allclose(sol.Mx, 0.0, atol=1e-5)
        np.testing.assert_allclose(sol.Mz, 0.0, atol=1e-5)

    def test_solve_equilibrium_with_control_input(self):
        """Verifies zero net moments after trimming with control surface deflections."""
        deltas = [np.radians(5.0)]

        sol_trim = self.solver.solve_equilibrium(
            v_inf=self.v_inf,
            rho=self.rho,
            deltas=deltas,
            enforce_yaw_equilibrium=True,
        )

        # Moments must be trimmed to 0.0
        np.testing.assert_allclose(sol_trim.Mx, 0.0, atol=1e-5)
        np.testing.assert_allclose(sol_trim.Mz, 0.0, atol=1e-5)
        self.assertNotAlmostEqual(sol_trim.p, 0.0, places=4)

    def test_yaw_equilibrium_enforced_vs_constrained(self):
        """Compares equilibrium roll rate when r is free vs when r is constrained to 0.0."""
        deltas = [np.radians(5.0)]

        sol_free = self.solver.solve_equilibrium(
            v_inf=self.v_inf,
            rho=self.rho,
            deltas=deltas,
            enforce_yaw_equilibrium=True,
        )

        sol_constrained = self.solver.solve_equilibrium(
            v_inf=self.v_inf,
            rho=self.rho,
            deltas=deltas,
            enforce_yaw_equilibrium=False,
        )

        self.assertEqual(sol_constrained.r, 0.0)
        np.testing.assert_allclose(sol_constrained.Mx, 0.0, atol=1e-5)
        self.assertNotAlmostEqual(sol_free.r, 0.0, places=5)
        self.assertNotEqual(sol_free.p, sol_constrained.p)


if __name__ == "__main__":
    unittest.main()