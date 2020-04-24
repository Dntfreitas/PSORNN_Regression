import numpy as np

import ann.choice as choice
import arch
import operators
import util


class Swarm:

    def __init__(self, swarm_topology, c1, c2, iterations, n_particles, precision, data_set,
                 hidden_units):
        """
        **** ATTRIBUTES INITIALIZATION ****
        """
        self.topology = arch.topology(swarm_topology)  # Topology
        self.options = {'c1': c1, 'c2': c2, 'iteration': 0, 'max_iteration': iterations,
                        'precision': precision, 'bounds': np.nan}  # Parameters
        self.iterations = iterations  # Number of iterations
        self.n_particles = n_particles  # Number of particles
        self.time_elapsed = np.nan  # Time elapsed
        self.f = choice.getAnn(data_set)
        self.f.set_n_hidden(hidden_units)
        self.dimensions = self.f.get_dimension()
        self.center = 2.4 * self.f.get_n_inputs()
        max_bound = 30 * np.ones(self.dimensions)
        min_bound = - max_bound
        self.options['bounds'] = (min_bound, max_bound)
        """
        **** PARTICLES INITIALIZATION ****
        """
        self.swarm = util.create_swarm(
            n_particles=self.n_particles, dimensions=self.dimensions, center=self.center, options=self.options)

    def run(self):
        max_iterations = self.options['max_iteration']
        precision = self.options['precision']
        precision_stop = False

        while (self.options['iteration'] < max_iterations or self.options['max_iteration'] == -1) and \
                not precision_stop:
            # Step 1: Compute current fitness
            self.swarm.current_cost = self.f.compute(
                self.swarm.position)  # Compute current cost

            # Step 2: Compute and update current personal best
            self.swarm.pbest_cost = self.f.compute(self.swarm.pbest_pos)
            self.swarm.pbest_pos, self.swarm.pbest_cost = operators.compute_pbest(
                self.swarm)

            # Part 2: Update global best
            self.swarm.best_pos, self.swarm.best_cost = self.topology.compute_gbest(
                self.swarm)

            # Step 4: Update position and velocity for the next iteration
            self.swarm.velocity = self.topology.compute_velocity(self.swarm)
            self.swarm.position = self.topology.compute_position(self.swarm)

            # Increment the number of iterations
            self.options['iteration'] += 1

            print("Best cost:", self.swarm.best_cost)

            if self.swarm.best_cost <= precision:
                precision_stop = True

        # Test swarm
        best_weights = self.swarm.pbest_pos[self.swarm.pbest_cost.argmin()]
        mse_test, r2 = self.f.test(best_weights)
        # Report results
        results = {'weights': best_weights,
                   'train_cost': self.swarm.best_cost, 'test_cost': mse_test, 'r2': r2}
        print(results)
        return results
