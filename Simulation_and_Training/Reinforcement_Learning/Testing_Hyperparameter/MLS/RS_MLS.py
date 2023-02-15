## PENDING PARTS REPLACED BY PENDING ORDERS [0,1]

###########################
### 1 - Import packages ###
###########################

from SimulateAndLearn.RL.Sim_Env import InventorySystem
from PredictStock import XGB
from MLS import order_policy

import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import time

# Set whether to display on screen (slows model)
DISPLAY_ON_SCREEN = False
# Time step between actions
TIME_STEP = 1

### Simulation Parameter ###
START_INVENTORY = 20
MEAN_DEMAND_SIZE = 15
SIGMA_DEMAND_SIZE = 3
COST_PER_ITEM = 2
COST_PER_ORDER = 10
COST_RATE_SHORTAGE = 3
COST_RATE_HOLDING = 0.1
COST_INVENTORY_CHECK = 10
DEMAND_DEVIATION_BOUNDARY = 0.7
INVISIBLE_DEMAND_SIZE = 0.3
BATCH_SIZE_ORDERS = 20
DEVIATION_DIRECTION = 0.7



features = 1
action_space = 12

class NN(nn.Module):

    def __init__(self, features, action_space, neurons_per_layer):
        dropout_rate = 0
        super(NN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(features, neurons_per_layer),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(neurons_per_layer, neurons_per_layer),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(neurons_per_layer, neurons_per_layer),
            nn.LeakyReLU(),
            nn.Linear(neurons_per_layer, action_space)
        )

    def forward(self, x):
        return self.net(x)


def get_action(class_probability, prediction, string, mod, neurons_per_layer):

    model = NN(features, action_space, neurons_per_layer)
    model.load_state_dict(torch.load(string))
    model.eval()

    global action

    if mod == "MLS":
        state = prediction
        state = np.array([[state]])
        q_values = model.forward(torch.FloatTensor(state))
        action = np.argmax(q_values.detach().numpy()[0])

    elif mod == "QMDP":

        class_probability = np.around(class_probability, 2)
        relevant_classes = np.where(class_probability > 0)
        relevant_classes = relevant_classes[1]
        expected_rewards_per_action = np.zeros((1, action_space))

        for x in np.nditer(relevant_classes.T):
            tensor = np.array(np.array([[x]]))
            pred = model.forward(torch.FloatTensor(tensor))
            predicted_action = np.argmax(pred.detach().numpy()[0])
            best_q = pred[0][predicted_action].detach().item()
            weighted_pred = best_q * class_probability[0][x].item()
            expected_rewards_per_action[0][predicted_action] += weighted_pred

        # multiply with -1 to select the index with the highest q-value
        expected_rewards_per_action *= -1
        action = np.argmax(expected_rewards_per_action[0])


    return action



def predict(system_stock, last_stock_count):

    class_prob = XGB.class_probability(system_stock, last_stock_count)
    prediction = XGB.predict(system_stock, last_stock_count)
    return class_prob, prediction


############################################
### 6 - Define results plotting function ###
############################################
def order_policy_eval(string, test_sim_dur, test_epochs, mod, neurons_per_layer):

    global total_reward
    sim = InventorySystem(
        sim_duration = test_sim_dur,
        time_step = TIME_STEP,
        mean_demand_size = MEAN_DEMAND_SIZE,
        sigma_demand_size = SIGMA_DEMAND_SIZE,
        start_inventory = START_INVENTORY,
        cost_per_item = COST_PER_ITEM,
        cost_per_order = COST_PER_ORDER,
        cost_rate_shortage = COST_RATE_SHORTAGE,
        cost_rate_holding = COST_RATE_HOLDING,
        cost_inventory_check = COST_INVENTORY_CHECK,
        demand_deviation_boundary = DEMAND_DEVIATION_BOUNDARY,
        invisible_demand_size= INVISIBLE_DEMAND_SIZE,
        batch_size= BATCH_SIZE_ORDERS,
        deviation_direction=DEVIATION_DIRECTION
        )

    ## 6.1 - Set up and start the training loop ##
    run = 0
    continue_learning = True

    # Set up dicts to store the results of the different simulation runs
    results_run = []
    results_score = []
    results_fraction_of_satisfied_demand = []
    results_fraction_of_satisfied_orders = []

    while continue_learning:

        # Count run
        run += 1
        # Reset sim env
        state = sim.reset()
        system_stock = state[0]
        last_stock_count = state[1]

        class_prob, prediction = predict(system_stock, last_stock_count)

        # Set up trackers
        inventory_system = []
        inventory_actual = []
        rewards = []
        taken_actions = []
        inventory_action = []
        total_reward = 0


        while True:

            action = get_action(class_prob, prediction, string, mod, neurons_per_layer)
            class_prob, prediction = predict(system_stock, last_stock_count)

            taken_actions.append(action)
            inventory_action.append([state[0], action])

            ## Play action ##
            # Get the results from the simulation with the respective action taken
            state_next, reward, terminal, info, KPI = sim.step(action)
            total_reward += reward

            # Store the simulation results
            inventory_system.append(state_next[0])
            rewards.append(reward)
            taken_actions.append(action)

            system_stock = state_next[0]
            last_stock_count = state_next[1]


            # Display the simulation results of each run if needed (slows down the simulation)
            if DISPLAY_ON_SCREEN:
                sim.render()

            # Actions to take at the end of gaming episode
            if terminal:

                fraction_of_satisfied_demand = KPI[0]
                fraction_of_satisfied_orders = KPI[1]

                # Add sim results after one simulation run to the results lists
                results_run.append(run)
                results_score.append(total_reward)
                results_fraction_of_satisfied_demand.append(fraction_of_satisfied_demand)
                results_fraction_of_satisfied_orders.append(fraction_of_satisfied_orders)

                # Get total reward
                total_reward = np.sum(rewards)

                # Check for end of learning. This will interrupt the training
                if run == test_epochs:
                    continue_learning = False

                # End episode loop
                break


    # Create data frame to store simulation results
    sim_details = pd.DataFrame()
    sim_details['run'] = results_run
    sim_details['reward'] = results_score
    sim_details['satisfied_demand'] = results_fraction_of_satisfied_demand
    sim_details['satisfied_orders'] = results_fraction_of_satisfied_orders

    #print(sim_details)
    return sim_details



def evaluation(mod, iterations):

    # Initialize lists to store the results
    total_results_lr = []
    total_results_gamma = []
    total_results_batch_size = []
    total_results_replay = []
    total_results_dropout = []
    total_results_neurons = []
    total_results_average_reward = []
    total_results_sigma_reward = []
    total_results_average_satisfied_orders = []
    total_results_average_satisfied_demand = []

    global best_result
    # initialize best results with an awful value to make sure it is updated after the first run
    best_result = -100000
    global best_result_sigma
    global best_lr
    global best_gamma
    global best_batch
    global best_dropout
    global best_neurons
    global best_replay
    run = 0


    for i in range(iterations):
        run += 1

        print("------------------------")
        print("Start Run: "+str(i))

        # Parameter lists where value are chosen from randomly (1440 possible combinations)
        learning_rate_list = np.linspace(0.0001, 0.002, num=20)
        gamma_list = np.linspace(0.60, 0.99, num=12)
        batch_size_list = [16, 32, 64, 128]
        replay_start_list = np.linspace(500, 4000, num=8)
        dropout_list = [0.2, 0.3, 0.4, 0.5, 0.6]
        neurons_list = [16, 32, 64, 128]

        # Train and Test Episodes and Simulation duration
        train_epochs = 50
        train_sim_dur = 100
        test_epochs = 20
        test_sim_dur = 200

        # Chose the parameters randomly out of the lists
        learning_rate_choice = random.choice(learning_rate_list)
        print(f'Learning_rate: {learning_rate_choice}')
        total_results_lr.append(learning_rate_choice)

        gamma_choice = random.choice(gamma_list)
        print(f'Gamma: {gamma_choice}')
        total_results_gamma.append(gamma_choice)

        batch_size_choice = random.choice(batch_size_list)
        print(f'Batch size: {batch_size_choice}')
        total_results_batch_size.append(batch_size_choice)

        replay_choice = random.choice(replay_start_list)
        print(f'Replay start size: {replay_choice}')
        total_results_replay.append(replay_choice)

        dropout_choice = random.choice(dropout_list)
        print(f'Dropout rate: {dropout_choice}')
        total_results_dropout.append(dropout_choice)

        neurons_choice = random.choice(neurons_list)
        print(f'Neurons per layer: {neurons_choice}')
        total_results_neurons.append(neurons_choice)



        # !!! Needs to be updated !!!
        string = "/Users/benedictrau/Documents/GitHub/Masterarbeit/SimulateAndLearn/"+\
                 "RL/Testing_Hyperparameter/MLS/results_NN_RS/"+str(mod)+\
                 "_lr_"+str(learning_rate_choice)+\
                 "_gamma_"+str(gamma_choice)+\
                 "_batch_"+str(batch_size_choice)+\
                 "_replay_"+str(replay_choice)+\
                 "_dropout_"+str(dropout_choice)+\
                 "_neurons_"+str(neurons_choice)+".pt"

        #print("Train")
        order_policy(learning_rate = learning_rate_choice, gamma = gamma_choice, train_sim_dur=train_sim_dur,
                     train_epochs=train_epochs, batch_size = batch_size_choice, replay_start_size=replay_choice,
                     string = string, dropout_rate= dropout_choice, neurons_per_layer=neurons_choice)

        #print("Test")
        results = order_policy_eval(string, test_sim_dur=test_sim_dur, test_epochs=test_epochs, mod=mod,
                                    neurons_per_layer=neurons_choice)

        total_results_average_reward.append(results["reward"].mean())
        total_results_sigma_reward.append(results["reward"].std())
        total_results_average_satisfied_orders.append(results["satisfied_orders"].mean())
        total_results_average_satisfied_demand.append(results["satisfied_demand"].mean())

        print(f'average test reward: {results["reward"].mean()}')


        if results["reward"].mean() > best_result:
            best_result = results["reward"].mean()
            best_result_sigma = results["reward"].std()
            best_lr = learning_rate_choice
            best_gamma = gamma_choice
            best_batch = batch_size_choice
            best_dropout = dropout_choice
            best_neurons = neurons_choice
            best_replay = replay_choice


    total_results = pd.DataFrame()
    total_results['lr'] = total_results_lr
    total_results['gamma'] = total_results_gamma
    total_results['batch_size'] = total_results_batch_size
    total_results['dropout_rate'] = total_results_dropout
    total_results['neurons_per_layer'] = total_results_neurons
    total_results['replay_start_size'] = total_results_replay
    total_results['avg_satisfied_orders'] = total_results_average_satisfied_orders
    total_results['avg_satisfied_demand'] = total_results_average_satisfied_demand
    total_results['sigma_reward_training'] = total_results_sigma_reward
    total_results['avg_reward_training'] = total_results_average_reward

    total_results.to_excel("results_NN_FS/Results_" + str(run) + ".xlsx", sheet_name="Results")

    print("------------------------")
    print("RESULTS")
    # show all columns
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(total_results)

    print("Best parameters found after "+str(iterations)+" iterations with:")
    print(f'lr: {best_lr}')
    print(f'gamma: {best_gamma}')
    print(f'batch size: {best_batch}')
    print(f'dropout: {best_dropout}')
    print(f'neurons: {best_neurons}')
    print(f'replay: {best_replay}')
    print(f'avg_reward: {best_result}')
    print(f'reward_std: {best_result_sigma}')


start_proc = time.process_time()

run = evaluation(mod="MLS", iterations=2)

end_proc = time.process_time()
print('Required time: {:5.3f}s'.format(end_proc-start_proc))

