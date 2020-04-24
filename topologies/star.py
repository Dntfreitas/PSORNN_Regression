import random

import numpy as np

import operators as ops
from topologies.topology import Topology


class Star(Topology):

    def __init__(self):
        super(Star, self).__init__()

    def compute_gbest(self, swarm):
        if self.neighbor_idx is None:
            self.neighbor_idx = star_nei(swarm.n_particles)

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


def star_nei(n_particles):
    neighborhood = []
    central_particle = random.randint(0, n_particles - 1)
    for identification in range(n_particles):
        if identification != central_particle:
            neighbors = np.array([central_particle])
        else:
            neighbors = np.arange(n_particles)
            neighbors = neighbors[neighbors != central_particle]
        neighborhood.append(neighbors.astype(int))
    return np.asanyarray(neighborhood)
