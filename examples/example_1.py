import sys
sys.path.append('/home/charlie/Documents/Python3/MyPackages/liftingline')
import liftingline as ll

# 1. Define Main Wing Planform
wing_spans = [0.0, 3.0, 6.0]
chords = [1.5, 1.2, 0.6]
alphas = [0.05, 0.04, 0.02]  # ~2.8 deg twist at root tapering to ~1.1 deg at tip

# 2. Define Control Surfaces
aileron = ll.ControlSurface(
    spans=[3.5, 5.8],  # Spans from y=3.5 to y=5.8
    delta_alpha0=[-0.4, -0.25],  # Tapered control effectiveness along span
    symmetric=False,  # Roll control (-1 right, +1 left)
)

flap = ll.ControlSurface(
    spans=[0.5, 3.0],  # Spans from y=0.5 to y=3.0
    delta_alpha0=[-0.8, -0.8],  # Constant flap deflection shift
    symmetric=True,  # Symmetric lift control
)

# 3. Instantiate WingShape
wing = ll.WingShape(wing_spans, chords, alphas, controls=[aileron, flap])

# Integrated seamless execution in LiftingLineSolver
solver = ll.LiftingLineSolver(wing, v_inf=30.0)