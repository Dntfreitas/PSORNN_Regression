from swarm import Swarm
import numpy as np

iterations = 5000

data_set = 1
# 1 - winequality-red.csv

hidden_units = 4

swarm = Swarm(swarm_topology="Toroid", c1=0.5, c2=0.3, w=0.9,
              iterations=iterations, n_particles=24, velocity_str=3, precision=10 ** (-5), data_set, hidden_units)

swarm.run()
