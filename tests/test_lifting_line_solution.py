import sys
import unittest
import numpy as np

# Adjust module path if necessary
sys.path.append("/home/charlie/Documents/Python3/MyPackages/liftingline")

import liftingline as ll


class TestLiftingLineSolution(unittest.TestCase):
    def test_oswald_efficiency(self):
        chord = 0.1
        combos = [
            [1.5, 20, 1.02],
            [0.5, 20, 1.02],
            [1.5, 50, 1.40],
            [5.0, 30, 0.9],
        ]
        for combo in combos:
            span, v_inf, rho = combo
            AR = span / chord
            S = span * chord
            spans = np.array([0, span / 2])
            chords = np.array([chord, chord])
            alphas = np.array([3, 1])

            wingshape = ll.WingShape(spans, chords, alphas)
            solver = ll.LiftingLineSolver(wingshape)

            # Modern functional API: solver returns a LiftingLineSolution object
            sol = solver.solve(v_inf=v_inf)

            L, D = sol.L_total, sol.D_total
            Cl = L / (0.5 * rho * v_inf**2 * S)
            Cd = D / (0.5 * rho * v_inf**2 * S)

            e_theory = Cl**2 / (np.pi * AR * Cd)
            e_solver = sol.oswald_efficiency

            self.assertAlmostEqual(e_solver, e_theory, delta=1e-3)