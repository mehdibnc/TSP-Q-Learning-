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

def jlj():
    pass 

def compute_greedy_route(Q_table: np.ndarray):
    """ Computes greedy route based on Q values """
    N = Q_table.shape[0]
    mask = np.array([True]*N)
    route = np.zeros((N,))
    mask[0] = False
    for i in range(1, N):
        # Iteration i : choosing ith city to visit, knowing the past
        current = route[i-1]
        next_visit = np.argmax(Q_table[int(current), mask])
        # update mask and route
        mask[next_visit] = False
        route[i] = next_visit
    return route 


def trace_progress(values: list, true_best: float):
    """ Trace progress of q learning. 
    
        Args:
            values: list of tour lenghts over qlearning iterations
            true_best: true optimal value for corresponding instance

        Returns:
            matplotlib figure        
    """
    pass

