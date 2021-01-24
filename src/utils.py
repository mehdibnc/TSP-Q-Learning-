###
# Utils functions.
#
###
import matplotlib.pyplot as plt 
from numba import njit 
import numpy as np



def load_data():
    """ Loads TSP instances
    
        Returns:
            data: dict, contains distance matrix and optimal value of tour.
    
    """
    dist_matrix_15 = np.loadtxt("../data/tsp_15_291.txt") #15 cities and min cost = 291
    dist_matrix_26 = np.loadtxt("../data/tsp_26_937.txt")
    dist_matrix_17 = np.loadtxt("../data/tsp_17_2085.txt")
    dist_matrix_42 = np.loadtxt("../data/tsp_42_699.txt")
    dist_matrix_48 = np.loadtxt("../data/tsp_48_33523.txt")

    data = {15:(dist_matrix_15, 291),
            17:(dist_matrix_17, 2085),
            26:(dist_matrix_26, 937),
            42:(dist_matrix_42, 699),
            48:(dist_matrix_48, 33523)}
    return data 



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




