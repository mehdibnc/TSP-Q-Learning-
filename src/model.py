import numpy as np
from numba import njit
from src.utils import compute_value_of_q_table, custom_argmax


@njit
def eps_greedy_update(
    Q_table: np.ndarray,
    distances: np.ndarray,
    mask: np.ndarray,
    route: np.ndarray,
    epsilon: float,
    gamma: float,
    lr: float,
    N: int,
):
    """Updates Q table using epsilon greedy.

    Args:
        Q_table (np.ndarray): Input Q table.
        distances (np.ndarray): Distance matrix describing the TSP instance.
        mask (np.ndarray): Boolean mask giving which cities to ignore (already visited).
        route (np.ndarray): Route container.
        epsilon (float): exploration parameter for epsilon greedy.
        gamma (float): weight for future reward.
        lr (float): learning rate for q updates.

    Returns:
        np.ndarray: Updated Q table.
    """
    mask[0] = False
    next_visit = 0
    reward = 0
    for i in range(1, N):
        # Iteration i : choosing ith city to visit
        possible = np.where(mask == True)[0]
        current = route[i - 1]
        if len(possible) == 1:
            next_visit = possible[0]
            reward = -distances[int(current), int(next_visit)]
            # Reward for finishing the route
            max_next = -distances[int(next_visit), int(route[0])]
        else:
            u = np.random.random()
            if u < epsilon:
                # random choice amongst possible
                next_visit = np.random.choice(possible)
            else:
                next_visit, _ = custom_argmax(Q_table, int(current), mask)
            # update mask and route
            mask[next_visit] = False
            route[i] = next_visit
            reward = -distances[int(current), int(next_visit)]
            # Get max Q from new state
            _, max_next = custom_argmax(Q_table, int(next_visit), mask)
        # updating Q
        Q_table[int(current), int(next_visit)] = Q_table[
            int(current), int(next_visit)
        ] + lr * (reward + gamma * max_next - Q_table[int(current), int(next_visit)])
    return Q_table


@njit
def QLearning(
    Q_table: np.ndarray,
    distances: np.ndarray,
    epsilon: float,
    gamma: float,
    lr: float,
    epochs: int = 100,
):
    """Performs simple Q learning algorithm, epsilon greedy, to learn
        a solution to the TSP.

    Args:
        Q_table (np.ndarray): Initial Q table.
        distances (np.ndarray): Distance matrix describing the TSP instance.
        epsilon (float): exploration parameter for epsilon greedy.
        gamma (float): weight for future reward.
        lr (float): learning rate for q updates.
        epochs (int, optional): Number of iterations. Defaults to 100.

    Returns:
        np.ndarray: Q table obtained after training..
        list: contains greedy distances for each epoch.
    """
    N = Q_table.shape[0]
    CompQ_table = Q_table.copy()
    mask = np.array([True] * N)
    route = np.zeros((N,))
    cache_distance_best = np.zeros((epochs,))
    cache_distance_comp = np.zeros((epochs,))
    for ep in range(epochs):
        CompQ_table = eps_greedy_update(
            CompQ_table, distances, mask, route, epsilon, gamma, lr, N
        )
        # update Q table only if best found so far is improved
        greedy_cost = compute_value_of_q_table(Q_table, distances)
        greedy_cost_comp = compute_value_of_q_table(CompQ_table, distances)
        cache_distance_best[ep] = greedy_cost
        cache_distance_comp[ep] = greedy_cost_comp
        if greedy_cost_comp < greedy_cost:
            Q_table[:, :] = CompQ_table[:, :]
        # resetting route and mask for next episode
        route[:] = 0
        mask[:] = True
    return Q_table, cache_distance_best, cache_distance_comp
