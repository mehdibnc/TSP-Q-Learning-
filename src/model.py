import numpy as np
from numba import njit
from src.utils import compute_greedy_route, custom_argmax, route_distance


def eps_greedy_update(
    Q_table: np.ndarray,
    distances: np.ndarray,
    mask: np.ndarray,
    route: np.ndarray,
    epsilon: float,
    gamma: float,
    lr: float,
    lbda: float,
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
        lbda (float): decay factor for epsilon.

    Returns:
        np.ndarray: Updated Q table.
    """
    mask[0] = False
    N = Q_table.shape[0]
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
                next_visit = custom_argmax(
                    Q_table, int(current), mask
                )  # np.argmax(Q_table[int(current), mask])
            # update mask and route
            mask[next_visit] = False
            route[i] = next_visit
            reward = -distances[int(current), int(next_visit)]
            # Get max Q from new state
            max_next = np.max(Q_table[int(next_visit), mask])
        # updating Q
        Q_table[int(current), int(next_visit)] = Q_table[
            int(current), int(next_visit)
        ] + lr * (reward + gamma * max_next - Q_table[int(current), int(next_visit)])
    return Q_table


def QLearning(
    Q_table: np.ndarray,
    distances: np.ndarray,
    epsilon: float,
    gamma: float,
    lr: float,
    lbda: float,
    epochs: int = 100,
    verbose: bool = False,
):
    """Performs simple Q learning algorithm, epsilon greedy, to learn
        a solution to the TSP.

    Args:
        Q_table (np.ndarray): Initial Q table.
        distances (np.ndarray): Distance matrix describing the TSP instance.
        epsilon (float): exploration parameter for epsilon greedy.
        gamma (float): weight for future reward.
        lr (float): learning rate for q updates.
        lbda (float): decay factor for epsilon.
        epochs (int, optional): Number of iterations. Defaults to 100.
        verbose (bool, optional): Whether to print progress. Defaults to False.

    Returns:
        np.ndarray: Q table obtained after training..
        list: contains greedy distances for each epoch.
    """
    N = Q_table.shape[0]
    mask = np.array([True] * N)
    route = np.zeros((N,))
    cache_distance = []
    for ep in range(epochs):
        epsilon *= 1 - lbda
        greedy_route = compute_greedy_route(Q_table)
        greedy_cost = route_distance(greedy_route, distances)
        cache_distance.append(greedy_cost)
        if verbose and (ep + 1) % 100 == 0:
            print(
                f"Episode {ep}. Epsilon {epsilon}. Current greedy cost {greedy_cost}."
            )
        Q_table = eps_greedy_update(
            Q_table, distances, mask, route, epsilon, gamma, lr, lbda
        )
        # resetting route and mask for next episode
        route[:] = 0
        mask[:] = True
    return Q_table, cache_distance
