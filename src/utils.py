###
# Utils functions.
#
###
import matplotlib.pyplot as plt 
from numba import njit 
import numpy as np


@njit
def route_distance(route: np.ndarray, distances: np.ndarray):
    """ Computes the distance of a route.

        Args:
            route: sequence of int, representing a route
            distances: distance matrix

        Returns:
            c: float, total distance travelled in route.
    """
    c = 0
    for i in range(1, len(route)):
        c += distances[route[i-1], route[i]]
    return c


def trace_progress(values: list, true_best: float):
    """ Trace progress of q learning. 
    
        Args:
            values: list of tour lenghts over qlearning iterations
            true_best: true optimal value for corresponding instance

        Returns:
            matplotlib figure        
    """
    pass