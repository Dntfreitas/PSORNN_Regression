# Import standard library

# Import modules
import math

import numpy as np


def compute_pbest(swarm):
    # Infer dimensions from positions
    dimensions = swarm.dimensions
    # Create a 1-D and 2-D mask based from comparisons
    mask_cost = swarm.current_cost < swarm.pbest_cost
    mask_pos = np.repeat(mask_cost[:, np.newaxis], dimensions, axis=1)
    # Apply masks
    new_pbest_pos = np.where(~mask_pos, swarm.pbest_pos, swarm.position)
    new_pbest_cost = np.where(
        ~mask_cost, swarm.pbest_cost, swarm.current_cost
    )

    return new_pbest_pos, new_pbest_cost


def compute_velocity(swarm):
    # Prepare parameters
    swarm_size = swarm.position.shape
    c1 = swarm.options["c1"]
    c2 = swarm.options["c2"]
    max_iterations = swarm.options["max_iteration"]
    current_iteration = swarm.options["iteration"]
    # Compute for cognitive and social terms
    c1 = 2.05
    c2 = 2.05
    c = c1 + c2

    cognitive = (
            c1
            * np.random.uniform(0, 1, swarm_size)
            * (swarm.pbest_pos - swarm.position)
    )
    social = (
            c2
            * np.random.uniform(0, 1, swarm_size)
            * (swarm.best_pos - swarm.position)
    )
    # Non-Linear
    w = 0.9 - (0.5 * current_iteration) / max_iterations
    # velocity = (w * swarm.velocity) + cognitive + social

    k = 2 / abs(2 - c - math.sqrt(c ** 2 - 4 * c))
    velocity = k * (swarm.velocity + cognitive + social)

    return velocity


def compute_position(swarm):
    bounds = swarm.options['bounds']
    temp_position = swarm.position.copy()
    temp_position += swarm.velocity

    if bounds is not None:
        temp_position = nearest(temp_position, bounds)

    position = temp_position
    return position


def compute_objective_function(swarm, objective_func):
    return objective_func(swarm.position)


def nearest(position, bounds):
    lb, ub = bounds
    bool_greater = position > ub
    bool_lower = position < lb
    new_pos = np.where(bool_lower, lb, position)
    new_pos = np.where(bool_greater, ub, new_pos)
    return new_pos
