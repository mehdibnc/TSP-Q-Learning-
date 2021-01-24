###
# Utils functions.
#
###
import matplotlib.pyplot as plt 
import numpy as np
from prettytable import PrettyTable



def load_data():
    """ Loads TSP instances
    
        Returns:
            data: dict, contains distance matrix and optimal value of tour.
    
    """

    dist_matrix_15 = np.loadtxt("data/tsp_15_291.txt") #15 cities and min cost = 291
    dist_matrix_26 = np.loadtxt("data/tsp_26_937.txt")
    dist_matrix_17 = np.loadtxt("data/tsp_17_2085.txt")
    dist_matrix_42 = np.loadtxt("data/tsp_42_699.txt")
    dist_matrix_48 = np.loadtxt("data/tsp_48_33523.txt")

    data = {15:(dist_matrix_15, 291),
            17:(dist_matrix_17, 2085),
            26:(dist_matrix_26, 937),
            42:(dist_matrix_42, 699),
            48:(dist_matrix_48, 33523)}
    return data 




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
        c += distances[int(route[i-1]), int(route[i])]
    c += distances[int(route[-1]), int(route[0])]
    return c


def custom_argmax(Q_table: np.ndarray, row: int, mask: np.ndarray):
    """ Compute argmax over one row of Q table on unmasked colunms.

        Args:
            Q_table: Q values table
            row: id of row over which to compute argmax
            mask: columns to mask

        Returns:
            argmax: value of argmax
    
    """
    argmax = 0
    max_v = -np.inf 
    idx = np.arange(Q_table.shape[1])
    np.random.shuffle(idx)
    for i in idx:
        if not(mask[i]):
            continue
        if Q_table[row, i] > max_v:
            argmax = i
            max_v = Q_table[row, i]
    return argmax


def compute_greedy_route(Q_table: np.ndarray):
    """ Computes greedy route based on Q values """
    N = Q_table.shape[0]
    mask = np.array([True]*N)
    route = np.zeros((N,))
    mask[0] = False
    for i in range(1, N):
        # Iteration i : choosing ith city to visit, knowing the past
        current = route[i-1]
        next_visit = custom_argmax(Q_table, int(current), mask) 
        # update mask and route
        mask[next_visit] = False
        route[i] = next_visit
    return route 


def trace_progress(values: list, true_best: float, tag: str):
    """ Trace progress. 
        Figure is save in ../figures/
        Args:
            values: list of tour lenghts over qlearning iterations
            true_best: true optimal value for corresponding instance

        Returns:
            None       
    """
    plt.figure(figsize=(19,7))
    plt.plot(values, label='Tour distance')
    plt.hlines(true_best, xmin=0, xmax=len(values), color='r', label='True best')
    plt.title(tag)
    plt.legend()
    plt.savefig(f'figures/Distance_evolution_{tag}')


def write_overall_results(res: dict, 
                          data: dict,
                          tag: str):
    """ Saves prettytable to summarize results
    
        Args:
            res: contains 
            data: problem data, key is number of cities, value tuple (distances, best tour value)
            tag: tag name to add to file name
        
        Returns:
            None
    
    """
    table = PrettyTable()
    table.field_names = ["Number of cities", "Tour distance QLearning", "Best distance"]
    for c in res:
        table.add_row([c, res[c], data[c][1]])
    with open(f'Results{tag}.txt', 'w') as f:
        f.write(str(table))
    

