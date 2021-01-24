from src import utils, model
import numpy as np 

def main():
    """ """
    #Loading test instances
    data = utils.load_data()

    #Running QLearning 
    Q_table = np.zeros((15,15))
    model.QLearning(Q_table,
                    data[15][0],
                    epsilon=1,
                    gamma=0.88,
                    lr=0.1,
                    lbda=0.0001,
                    epochs=1000)

    #Saving evaluation figure



if __name__ == '__main__':
    main()
