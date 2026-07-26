import numpy as np
from matplotlib import pyplot as plt

import sys
sys.path.append('C:/Users/charl/My Drive/Documents/Python3/MyPackages/liftingline')
import liftingline as ll

spans = np.array([0, 1])
chords = np.array([.1, .04])
alphas = np.array([5, 5])
v_inf = 30

wingshape = ll.WingShape(spans, chords, alphas)
solver = ll.LiftingLineSolver(wingshape, v_inf)

plt.plot(solver.y, solver.L)
plt.plot(solver.y, solver.D)
plt.show()
