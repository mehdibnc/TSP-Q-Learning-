from numba import njit 
import numpy as np 
import sys 
print('oooooo')
from utils import compute_greedy_route, route_distance


def eps_greedy_update(Q_table: np.ndarray,
                      distances: np.ndarray,
                      epsilon: float, 
                      gamma: float, 
                      lr: float, 
                      lbda: float):
    """ """
    pass

def QLearning(Q_table: np.ndarray,
              distances: np.ndarray,
              epsilon: float, 
              gamma: float, 
              lr: float, 
              lbda: float, 
              epochs: int = 100):
    """ Q Learning 
    
        Args:


        Returns:

    
    
    """
    N = Q_table.shape[0]
    mask = np.array([True]*N)
    route = np.zeros((N,))
    for ep in range(epochs):
        epsilon = epsilon * (1-lbda)
        greedy_route = compute_greedy_route(Q_table)
        greedy_cost = route_distance(greedy_route, distances)
        if (ep+1)%100 == 0:
            print(f"Episode {ep}. Epsilon {epsilon}. Current greedy cost {greedy_cost}.")
        mask[0] = False
        for i in range(1, N):
            # Iteration i : choosing ith city to visit, knowing the past
            possible = np.where(mask==True)[0]
            current = route[i-1]
            if len(possible) == 1:
                next_visit = possible[0]
                reward = -distances[int(current), int(next_visit)] + 200
                # Get max Q from new state
                max_next = 0
            else:
                u = np.random.random()
                if u < epsilon:
                    # random choice amongst possible
                    next_visit = np.random.choice(possible)
                else:
                    next_visit = np.argmax(Q_table[int(current), mask])
                # update mask and route
                mask[next_visit] = False
                route[i] = next_visit
                reward = -distances[int(current), int(next_visit)]
                # Get max Q from new state
                max_next = np.max(Q_table[int(next_visit), mask])
            # updating Q
            Q_table[int(current), int(next_visit)] = Q_table[int(current), int(next_visit)] + lr * (reward + gamma * max_next - Q_table[int(current), int(next_visit)])
        
        # resetting route and mask for next episode
        route[:] = 0
        mask[:] = True
    return Q_table
