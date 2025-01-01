import sys
sys.path.append('..')
from systems import TwoDimDoubleIntegratorNominal

import numpy as np

system = TwoDimDoubleIntegratorNominal(padding=5)

state0 = np.array([230.0, 0.0, 100.0, 0.0])
action = np.array([30.0, 30.0])
DT = 0.1
w = 240
h = 120
feature_map = np.zeros((h, w, 3))
state_list = []
for i in range(100):
    state1 = system.dynamics(state0, action, DT, feature_map)
    state_list.append(state1)
    state0 = state1.copy()
    
import matplotlib.pyplot as plt
plt.plot([s[0] for s in state_list], [s[2] for s in state_list])
plt.plot(state_list[0][0], state_list[0][2], 'ro')
plt.plot(state_list[-1][0], state_list[-1][2], 'gx')
plt.plot([5, 5, 235, 235, 5], [5, 115, 115, 5, 5])
plt.plot([0, 0, 240, 240, 0], [0, 120, 120, 0, 0])

plt.savefig('test_systems_plot.png')
