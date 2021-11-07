import time

import numpy as np
from src.model import QLearning
from src.utils import (compute_greedy_route, load_data, route_distance,
                       trace_progress, write_overall_results)

EPOCHS = 4000
LEARNING_RATE = 0.2
GAMMA = 0.95
EPSILON = 0.1


def main():
    """Q Learning method is ran on each benchmark instance.
    Figures monitoring progress are saved in figures/
    """
    # Loading test instances
    data = load_data()
    start = time.time()
    # Running QLearning on each instance
    res = dict()
    for c in data:
        Q_table = np.zeros((c, c))
        Q_table, cache_distance_best, cache_distance_comp = QLearning(
            Q_table,
            data[c][0],
            epsilon=EPSILON,
            gamma=GAMMA,
            lr=LEARNING_RATE,
            epochs=EPOCHS,
        )

        # Saving evaluation figure
        trace_progress(
            cache_distance_comp,
            data[c][1],
            f"{c}_Cities_Best_distance_{data[c][1]}_Agent_exploration",
        )
        trace_progress(
            cache_distance_best,
            data[c][1],
            f"{c}_Cities_Best_distance_{data[c][1]}_Best_solution_found",
        )
        # Final result for this instance
        greedy_route = compute_greedy_route(Q_table)
        greedy_cost = route_distance(greedy_route, data[c][0])
        res[c] = greedy_cost
    print(f"Time to run : {round(time.time() - start, 3)}")
    # Overall final results
    write_overall_results(res, data, "_no_hp_tuning")


if __name__ == "__main__":
    main()
