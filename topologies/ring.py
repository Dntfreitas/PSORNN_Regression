import numpy as np

import operators as ops
from topologies.topology import Topology


class Ring(Topology):

    def __init__(self):
        super(Ring, self).__init__()

    def compute_gbest(self, swarm):
        if self.neighbor_idx is None:
            self.neighbor_idx = ring_nei(swarm.n_particles)

        idx_min = np.array(
            [
                swarm.pbest_cost[self.neighbor_idx[i]].argmin()
                for i in range(len(self.neighbor_idx))
            ]
        )
        best_neighbor = np.array(
            [
                self.neighbor_idx[i][idx_min[i]]
                for i in range(len(self.neighbor_idx))
            ]
        ).astype(int)

        # Obtain best cost and position
        best_cost = np.min(swarm.pbest_cost[best_neighbor])
        best_pos = swarm.pbest_pos[best_neighbor]

        return best_pos, best_cost

    def compute_velocity(self, swarm):
        return ops.compute_velocity(swarm)

    def compute_position(self, swarm):
        return ops.compute_position(swarm)


def ring_nei(n_particles):
    neighborhood = []
    for identification in range(n_particles):
        neighbor_left = identification - 1 if identification > 0 else n_particles - 1
        neighbor_right = identification + 1 if identification < n_particles - 1 else 0
        neighbors = np.array([neighbor_left, neighbor_right])
        neighborhood.append(neighbors.astype(int))
    return np.asanyarray(neighborhood)
