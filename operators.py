# Import standard library
import math
import random
from functools import partial

# Import modules
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
    n = 1.2
    w = 0.9 + 0.5 * (1 - current_iteration / max_iterations) ** n
    velocity = (w * swarm.velocity) + cognitive + social

    return velocity


def compute_position(swarm):

    new_position = swarm.position.copy()
    new_position += swarm.velocity

    return new_position


def compute_objective_function(swarm, objective_func):

    return objective_func(swarm.position)
