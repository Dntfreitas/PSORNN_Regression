import numpy as np

from swarms import Swarm


def generate_swarm(n_particles, dimensions, center):
    pos = center * np.random.uniform(
        low=-1, high=1, size=(n_particles, dimensions))
    return pos


def generate_velocity(n_particles, dimensions):
    min_velocity, max_velocity = (0, 1)
    velocity = (max_velocity - min_velocity) * np.random.random_sample(
        size=(n_particles, dimensions)
    ) + min_velocity
    return velocity


def create_swarm(n_particles, dimensions, center, options):
    position = generate_swarm(n_particles, dimensions, center)
    velocity = generate_velocity(n_particles, dimensions)
    return Swarm(position, velocity, options=options)
