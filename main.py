from swarm import Swarm

iterations = 500

data_set = 1
# 1 - winequality-red.csv

# Star + 24 + K + linear w

hidden_units = 4

swarm = Swarm(swarm_topology="Ring", c1=2, c2=2,
              iterations=iterations, n_particles=24, precision=10 ** (-5), data_set=data_set,
              hidden_units=hidden_units)

swarm.run()
