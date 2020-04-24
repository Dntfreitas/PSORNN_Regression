import numpy as np

import operators as ops
from topologies.topology import Topology


class Classic(Topology):

    def __init__(self):
        super(Classic, self).__init__()

    def compute_gbest(self, swarm):

        if np.min(swarm.pbest_cost) < swarm.best_cost:
            # Get the particle position with the lowest pbest_cost
            # and assign it to be the best_pos
            best_pos = swarm.pbest_pos[np.argmin(swarm.pbest_cost)]
            best_cost = np.min(swarm.pbest_cost)
        else:
            # Just get the previous best_pos and best_cost
            best_pos, best_cost = swarm.best_pos, swarm.best_cost

        return best_pos, best_cost

    def compute_velocity(self, swarm):
        return ops.compute_velocity(swarm)

    def compute_position(self, swarm):
        return ops.compute_position(swarm)
