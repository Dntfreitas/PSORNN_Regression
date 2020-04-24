import abc


class Topology(abc.ABC):
    def __init__(self):
        self.neighbor_idx = None

    @abc.abstractmethod
    def compute_gbest(self, swarm):
        """Compute the best particle of the swarm and return the cost and
        position"""
        raise NotImplementedError("Topology::compute_gbest()")

    @abc.abstractmethod
    def compute_position(self, swarm):
        """Update the swarm's position-matrix"""
        raise NotImplementedError("Topology::compute_position()")

    @abc.abstractmethod
    def compute_velocity(self, swarm):
        """Update the swarm's velocity-matrix"""
        raise NotImplementedError("Topology::compute_velocity()")
