import unittest
import numpy as np
import sys
sys.path.append('/home/charlie/Documents/Python3/MyPackages/liftingline')
import liftingline as ll


class TestControlSurface(unittest.TestCase):

    def setUp(self):
        self.aileron = ll.ControlSurface(
            spans=[3.0, 6.0],
            delta_alpha0=[-0.4, -0.2],
            symmetric=False,
        )
        self.flap = ll.ControlSurface(
            spans=[0.0, 3.0],
            delta_alpha0=[-0.8, -0.8],
            symmetric=True,
        )

    def test_post_init_conversion(self):
        """Verify inputs are correctly converted to numpy arrays."""
        self.assertIsInstance(self.aileron.spans, np.ndarray)
        self.assertIsInstance(self.aileron.delta_alpha0, np.ndarray)
        self.assertEqual(self.aileron.spans.dtype, float)


class TestWingShape(unittest.TestCase):

    def setUp(self):
        # 0 to 6m span wing with taper and twist
        self.spans = [0.0, 3.0, 6.0]
        self.chords = [2.0, 1.5, 1.0]
        self.alphas = [0.1, 0.05, 0.0]  # radians

        self.aileron = ll.ControlSurface(
            spans=[3.0, 6.0],
            delta_alpha0=[-0.4, -0.2],
            symmetric=False,
        )
        self.flap = ll.ControlSurface(
            spans=[0.0, 2.0],
            delta_alpha0=[-0.5, -0.5],
            symmetric=True,
        )

        self.wing = ll.WingShape(
            airfoil_spans=self.spans,
            airfoil_chords=self.chords,
            airfoil_alphas=self.alphas,
            controls=[self.aileron, self.flap],
        )

    def test_initialization_and_sorting(self):
        """Ensure wing spans are properly sorted and total span is calculated."""
        # Unsorted inputs should end up sorted
        unsorted_wing = ll.WingShape(
            airfoil_spans=[6.0, 0.0, 3.0],
            airfoil_chords=[1.0, 2.0, 1.5],
            airfoil_alphas=[0.0, 0.1, 0.05],
        )
        np.testing.assert_array_equal(unsorted_wing.spans, [0.0, 3.0, 6.0])
        np.testing.assert_array_equal(unsorted_wing.chords, [2.0, 1.5, 1.0])
        self.assertEqual(self.wing.span, 12.0)  # max_span * 2
        self.assertEqual(self.wing.nr_of_controls, 2)

    def test_chord_interpolation(self):
        """Test linear interpolation of chord lengths on both wing halves."""
        y_query = np.array([-4.5, 0.0, 1.5, 6.0, 7.0])
        # Expected:
        # y=4.5 -> mid of 1.5 and 1.0 = 1.25
        # y=0.0 -> 2.0
        # y=1.5 -> mid of 2.0 and 1.5 = 1.75
        # y=6.0 -> 1.0
        # y=7.0 -> 0.0 (out of bounds)
        expected = np.array([1.25, 2.0, 1.75, 1.0, 0.0])
        np.testing.assert_allclose(self.wing.chord(y_query), expected)

    def test_alpha_geo_interpolation(self):
        """Test linear interpolation of geometric angle of attack."""
        y_query = np.array([-3.0, 0.0, 4.5])
        # Expected:
        # y=-3.0 -> 0.05
        # y= 0.0 -> 0.10
        # y= 4.5 -> mid of 0.05 and 0.00 = 0.025
        expected = np.array([0.05, 0.10, 0.025])
        np.testing.assert_allclose(self.wing.alpha_geo(y_query), expected)

    def test_aileron_antisymmetric_behavior(self):
        """Verify anti-symmetric aileron deflection (-1 right wing, +1 left wing)."""
        # Cont 0 is Aileron [spans 3.0 to 6.0]
        y_query = np.array([-4.5, -1.0, 0.0, 1.0, 4.5])
        # Mid-aileron deflection magnitude is -0.3
        # Left wing (-4.5): +0.3
        # Inboard/out of bounds (-1.0, 0, 1.0): 0.0
        # Right wing (+4.5): -(-0.3) = +0.3 (or opposite sign depending on convention)
        res = self.wing.alpha_control(y_query, cont_nr=0)

        self.assertAlmostEqual(res[0], -0.3)  # Left wing (+alpha_shift)
        self.assertEqual(res[1], 0.0)  # Outside control surface
        self.assertEqual(res[2], 0.0)
        self.assertEqual(res[3], 0.0)
        self.assertAlmostEqual(res[4], 0.3)  # Right wing (-alpha_shift)

    def test_flap_symmetric_behavior(self):
        """Verify symmetric flap deflection (+1 left wing, +1 right wing)."""
        # Cont 1 is Flap [spans 0.0 to 2.0], delta_alpha0 = -0.5
        y_query = np.array([-1.0, 0.0, 1.0, 3.0])
        res = self.wing.alpha_control(y_query, cont_nr=1)

        self.assertAlmostEqual(res[0], -0.5)  # Left wing
        self.assertAlmostEqual(res[1], -0.5)  # Root
        self.assertAlmostEqual(res[2], -0.5)  # Right wing
        self.assertEqual(res[3], 0.0)  # Outside flap bounds

    def test_invalid_control_index(self):
        """Requesting an out-of-bounds control surface index should return zeros."""
        y_query = np.array([0.0, 2.0, 4.0])
        res = self.wing.alpha_control(y_query, cont_nr=99)
        np.testing.assert_array_equal(res, np.zeros_like(y_query))


if __name__ == "__main__":
    unittest.main()