import numpy as np
from pyswarms.backend.topology import Topology

import operators as ops


class Toroid(Topology):

    def __init__(self, static=True):
        super(Toroid, self).__init__(static)

    def compute_gbest(self, swarm, **kwargs):
        try:

            if self.neighbor_idx is None:
                self.neighbor_idx = toroid_nei(swarm.n_particles)

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

        except AttributeError:
            self.rep.logger.exception(
                "Please pass a Swarm class. You passed {}".format(type(swarm))
            )
            raise
        else:
            return best_pos, best_cost

    def compute_velocity(self, swarm):
        return ops.compute_velocity(swarm)

    def compute_position(self, swarm):
        return ops.compute_position(swarm)


def toroid_nei(n_particles):
    neighborhood = []
    n_lines = 3
    particles_p_line = n_particles / n_lines
    for identification in range(n_particles):
        row = int(identification / particles_p_line)
        col = identification % particles_p_line
        n1 = identification - particles_p_line if row > 0 else identification + ((n_lines - 1) * particles_p_line)
        n2 = identification + 1 if col < particles_p_line - 1 else identification - particles_p_line + 1
        n3 = identification + particles_p_line if row < n_lines - 1 else identification - (particles_p_line * row)
        n4 = identification - 1 if col > 0 else -1 + particles_p_line + particles_p_line * row
        neighbors = np.array([n1, n2, n3, n4])
        neighborhood.append(neighbors.astype(int))
    return np.asanyarray(neighborhood)
