# Solving the Traveling Salesman Problem using Q-Learning


This repository explores a simple approach to applying a Q Learning algorithm to solve the 
Traveling Salesman Problem (TSP). Results and details are presented in a jupyter notebook `TSP.ipynb`.
Python 3.8.5.


## Requirements

`Python 3.8.5` is used. Requirements needed : `pip install -r requirements.txt`.

## Data

The algorithm is tested on a few instances in `data` these were downloaded [here](https://people.sc.fsu.edu/~jburkardt/datasets/tsp/tsp.html).

## Use

Running `main` will run QLearning model on TSP instances and save figure in /figures showing progression in tour length over iterations, and a table summarizing result. No hyper params has been done so far, results are in `Results_no_hp_params.txt`.

## Comments

This is a first model to attempt to solve TSP using Q Learning, the model could be refined, hyper parameters tuned and more research put into defining a good RL-based model.

Figures suggest several areas of improvements, for example the epsilon decay could be adaptive since we see that tour length sometimes reaches optimal value to increase again because of exploration. 


Author : Mehdi Bennaceur
