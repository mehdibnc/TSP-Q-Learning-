# Solving the Traveling Salesman Problem using Q-Learning


This repository explores a simple approach to applying a Q Learning algorithm to solve the 
Traveling Salesman Problem (TSP). 


## Dependencies

Dependencies are managed using poetry.

## Data

The algorithm is tested on a few instances in `data` these were downloaded [here](https://people.sc.fsu.edu/~jburkardt/datasets/tsp/tsp.html).

## Use

Running `poetry run python3 main.py` will run QLearning model on TSP instances and save figure in /figures showing progression in tour length over iterations, and a table summarizing result. No hyper params has been done so far, results are in `Results_no_hp_params.txt`.

## Comments

This is a first model to attempt to solve TSP using Q Learning, the model could be refined, hyper parameters tuned and more research put into defining a good RL-based model. Some tests are implemented, run `poetry run pytest` to run tests.


For an instance with 48 cities, the next figures shows the evolution of the cost of a greedy route as the Q table is updated.

![alt text](https://github.com/mehdibnc/TSP-Q-Learning-/blob/master/figures/Distance_evolution_48%20Cities%2C%20Best%20distance%2033523.png) 


Figures suggest several areas of improvements, for example the epsilon decay could be adaptive since we see that tour length sometimes reaches optimal value to increase again because of exploration. 


Author : Mehdi Bennaceur
