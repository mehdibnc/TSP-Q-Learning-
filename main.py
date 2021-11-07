import numpy as np
from src import model, utils


def main():
    """Q Learning method is ran on each benchmark instance.
    Figures monitoring progress are saved in figures/
    """
    # Loading test instances
    data = utils.load_data()

    # Running QLearning on each instance
    res = dict()
    for c in data:
        Q_table = np.zeros((c, c))
        Q_table, cache_distance = model.QLearning(
            Q_table, data[c][0], epsilon=1, gamma=0.9, lr=0.2, lbda=0.001, epochs=4000
        )

        # Saving evaluation figure
        utils.trace_progress(
            cache_distance, data[c][1], f"{c} Cities, Best distance {data[c][1]}"
        )

        # Final result for this instance
        greedy_route = utils.compute_greedy_route(Q_table)
        greedy_cost = utils.route_distance(greedy_route, data[c][0])
        res[c] = greedy_cost

    # Overall final results
    utils.write_overall_results(res, data, "_no_hp_tuning")


if __name__ == "__main__":
    main()
