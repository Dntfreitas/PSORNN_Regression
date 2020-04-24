import numpy as np


def create_swarm(n_particles, architecture, bounds):

    lb, ub = bounds
    min_bounds = np.repeat(
        np.array(lb)[np.newaxis, :], n_particles, axis=0
    )
    max_bounds = np.repeat(
        np.array(ub)[np.newaxis, :], n_particles, axis=0
    )
    pos = center * np.random.uniform(
        low=min_bounds, high=max_bounds, size=(n_particles, dimensions)
    )

    velocity = generate_velocity(n_particles, dimensions, clamp=clamp)


def generate_velocity(n_particles, dimensions):

    min_velocity, max_velocity = (0, 1)

    velocity = (max_velocity - min_velocity) * \
        np.random.random_sample(size=(n_particles, dimensions)) + min_velocity

    return velocity
