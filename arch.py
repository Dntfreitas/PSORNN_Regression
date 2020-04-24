from topologies.classic import Classic
from topologies.mesh import Mesh
from topologies.pyramid import Pyramid
from topologies.random import Random
from topologies.ring import Ring
from topologies.star import Star
from topologies.toroid import Toroid


def topology(parm):
    if parm == "Ring":
        return Ring()
    elif parm == "Star":
        return Star()
    elif parm == "Mesh":
        return Mesh()
    elif parm == "Classic":
        return Classic()
    elif parm == "Random":
        return Random()
    elif parm == "Pyramid":
        return Pyramid()
    elif parm == "Toroid":
        return Toroid()
