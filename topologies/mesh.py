import numpy as np

import operators as ops
from topologies.topology import Topology


class Mesh(Topology):

    def __init__(self):
        super(Mesh, self).__init__()

    def compute_gbest(self, swarm):
        if self.neighbor_idx is None:
            self.neighbor_idx = mesh_nei(swarm.n_particles)

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


def mesh_nei(n_particles):
    neighborhood = []
    n_lines = 3
    particles_p_line = n_particles / n_lines
    for identification in range(n_particles):
        row = int(identification / particles_p_line)
        col = identification % particles_p_line
        n1 = identification - particles_p_line if row > 0 else np.nan
        n2 = identification + 1 if col < particles_p_line - 1 else np.nan
        n3 = identification + particles_p_line if row < n_lines - 1 else np.nan
        n4 = identification - 1 if col > 0 else np.nan
        neighbors = np.array([n1, n2, n3, n4])
        neighbors = neighbors[~np.isnan(neighbors)]
        neighborhood.append(neighbors.astype(int))
    return np.asanyarray(neighborhood)
