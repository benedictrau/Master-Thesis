from Env_predict_stock import InventorySystem

import numpy as np
import pandas as pd
import random


# Set whether to display on screen (slows model)
DISPLAY_ON_SCREEN = False
# Time step between actions
TIME_STEP = 1
# Simulation duration
SIM_DURATION = 100
# Training episodes
EPISODES = 10000

### Simulation Parameter ###
START_INVENTORY = 20
MEAN_DEMAND_SIZE = 15
SIGMA_DEMAND_SIZE = 3
DEMAND_DEVIATION_BOUNDARY = 0.7
INVISIBLE_DEMAND_SIZE = 0.3
BATCH_SIZE = 20
DEVIATION_DIRECTION = 0.7


filename1 = 'Data/x_new.csv'
filename2 = 'Data/y_new.csv'

def Simulation_Environment():

    sim = InventorySystem(
        sim_duration = SIM_DURATION,
        time_step = TIME_STEP,
        mean_demand_size= MEAN_DEMAND_SIZE,
        sigma_demand_size = SIGMA_DEMAND_SIZE,
        start_inventory = START_INVENTORY,
        demand_deviation_boundary = DEMAND_DEVIATION_BOUNDARY,
        invisible_demand_size = INVISIBLE_DEMAND_SIZE,
        batch_size = BATCH_SIZE,
        deviation_direction = DEVIATION_DIRECTION
        )

    ## 6.1 - Set up and start the training loop ##
    run = 0
    all_steps = 0
    continue_learning = True

    system_level = []
    last_stock_count = []
    act_stock_class = []
    act_stock = []

    while continue_learning:
        # Count run
        run += 1
        decide = 0
        # Reset sim env
        state, stock_class, actual_stock = sim.reset()
        system_level.append(state[0])
        last_stock_count.append(state[1])
        act_stock_class.append(stock_class)
        act_stock.append(actual_stock)

        while True:

            all_steps +=1
            # take a random action to train the model
            i = random.random()
            if decide == 0:
                action = np.random.randint(0, 1)
            elif decide == 1:
                action = np.random.randint(2, 11)
            else:
                if i >= 0.4:
                    action = 0
                elif i >= 0.2:
                    action = 1
                else:
                    action = np.random.randint(2, 11)


            state, stock_class, actual_stock, terminal, info = sim.step(action)

            system_level.append(state[0])
            last_stock_count.append(state[1])
            act_stock_class.append(stock_class)
            act_stock.append(actual_stock)

            if state[0] >= 130:
                decide = 0
            elif state[0] <= 0:
                decide = 1
            else:
                decide = 2

            # Display the simulation results of each run if needed (slows down the simulation)
            if DISPLAY_ON_SCREEN:
                sim.render()

            # Actions to take at the end of gaming episode
            if terminal:
                print("\n")
                print(f'Run: {run}, ', end='')

                # Check for end of learning. This will interrupt the training
                if run == EPISODES:
                    continue_learning = False
                # End episode loop
                break

    x = pd.DataFrame()
    x['system_level'] = system_level
    #x['actual_stock_level'] = act_stock
    x['last_stock_count'] = last_stock_count

    y = pd.DataFrame()
    y['actual_stock_class'] = act_stock_class


    x.to_csv(filename1, index=False)
    y.to_csv(filename2, index=False)


# Run model and return last run results by day
last_run = Simulation_Environment()
