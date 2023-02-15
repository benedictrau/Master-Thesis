## PENDING PARTS REPLACED BY PENDING ORDERS [0,1]

###########################
### 1 - Import packages ###
###########################

from SimulateAndLearn.RL.Sim_Env import InventorySystem
from PredictStock import RF
from DualModeControl import order_policy

import numpy as np
import pandas as pd
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
COST_INVENTORY_CHECK = 5
DEMAND_DEVIATION_BOUNDARY = 0.7
INVISIBLE_DEMAND_SIZE = 0.3
BATCH_SIZE_ORDERS = 20
DEVIATION_DIRECTION = 0.7



features = 1
action_space = 12

class NN(nn.Module):

    def __init__(self, features, action_space, neurons_per_layer=64):
        super(NN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(features, neurons_per_layer),
            nn.LeakyReLU(),
            nn.Linear(neurons_per_layer, neurons_per_layer),
            nn.LeakyReLU(),
            nn.Linear(neurons_per_layer, neurons_per_layer),
            nn.LeakyReLU(),
            nn.Linear(neurons_per_layer, action_space)
        )

    def forward(self, x):
        return self.net(x)


def get_action(class_probability, prediction, string, mod):

    model = NN(features, action_space, neurons_per_layer=64)
    model.load_state_dict(torch.load(string))
    model.eval()

    global action

    if mod == "DMC":
        state = prediction
        state = np.array([[state]])
        q_values = model.forward(torch.FloatTensor(state))
        action = np.argmax(q_values.detach().numpy()[0])



    return action



def predict(system_stock, last_stock_count):

    class_prob = RF.class_probability(system_stock, last_stock_count)
    prediction = RF.predict(system_stock, last_stock_count)
    return class_prob, prediction


############################################
### 6 - Define results plotting function ###
############################################
def order_policy_eval(string, test_sim_dur, test_epochs, mod):

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


            action = get_action(class_prob, prediction, string, mod)
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

                # Clear print row content
                #print('------------------------')
                #clear_row = '\r' + ' ' * 79 + '\r'
                #print(clear_row, end='')
                #print(f'Run: {run}, ', end='')
                #print(f'Total reward: {total_reward:4.1f}')
                #print(f'Actions_taken: {taken_actions}')
                #print(f'Inventory System: {inventory_system}')
                #print(f'Inventory Actual: {inventory_actual}')
                #print(f'fraction_of_satisfied_demand: {fraction_of_satisfied_demand}')
                #print(f'fraction_of_satisfied_orders: {fraction_of_satisfied_orders}')

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


def get_parameters(parameter):
    # default_parameter
    default_lr = 0.00025
    default_gamma = 0.99
    default_memory_size = 10000
    default_batch_size = 32
    default_replay_start_size = 500
    default_synch_target_steps = 100
    default_exp_max = 1
    default_exp_min = 0.1
    default_exp_decay = 0.99
    default_threshold = 0.5

    # Parameter settings to be tested
    learning_rate_list = [0.000125, 0.0005]
    gamma_list = [0.8, 0.6]
    memory_size_list = [5000, 7500]
    batch_size_list = [16, 64]
    replay_start_size_list = [250, 1000]
    synch_target_steps_list = [50, 200]
    exp_max_list = [1, 1]
    exp_min_list = [0.05, 0.2]
    exp_decay_list = [0.8, 0.6]
    threshold_list = [0, 1]

    global list

    # get the parameter list for the parameter to be evaluated
    if parameter == "lr":
        list = learning_rate_list
    elif parameter == "gamma":
        list = gamma_list
    elif parameter == "memory_size":
        list = memory_size_list
    elif parameter == "batch_size":
        list = batch_size_list
    elif parameter == "replay_start_size":
        list = replay_start_size_list
    elif parameter == "synch_target_steps":
        list = synch_target_steps_list
    elif parameter == "exp_max":
        list = exp_max_list
    elif parameter == "exp_min":
        list = exp_min_list
    elif parameter == "exp_decay":
        list = exp_decay_list
    elif parameter == "threshold":
        list = threshold_list

    # initialize the parameter values by there default values
    lr = default_lr
    gamma = default_gamma
    memory_size = default_memory_size
    batch_size = default_batch_size
    replay_start_size = default_replay_start_size
    synch_target_steps = default_synch_target_steps
    exp_max = default_exp_max
    exp_min = default_exp_min
    exp_decay = default_exp_decay
    threshold = default_threshold

    # return the parameter values and the list of the parameter to be evaluated
    return lr, gamma, memory_size, batch_size, replay_start_size, synch_target_steps, \
           exp_max, exp_min, exp_decay, threshold, list



def get_default():

    lr, gamma, memory_size, batch_size, replay_start_size, synch_target_steps, \
    exp_max, exp_min, exp_decay, threshold, list = get_parameters(parameter="lr")

    train_epochs = 50
    train_sim_dur = 200
    test_epochs = 50
    test_sim_dur = 200

    # get benchmark
    default_string = "/Users/benedictrau/Documents/GitHub/Masterarbeit/SimulateAndLearn/" \
                     "RL/Testing_Hyperparameter/DMC/results_NN_FS/default.pt"

    print("------------------------")
    print("Default values")

    order_policy(learning_rate=lr, gamma=gamma, memory_size=memory_size, batch_size=batch_size,
                 replay_start_size=replay_start_size, synch_target_steps=synch_target_steps,
                 exp_max=exp_max, exp_min=exp_min, exp_decay=exp_decay, string=default_string,
                 train_epochs=train_epochs, train_sim_dur=train_sim_dur, threshold=threshold)

    default_results = order_policy_eval(default_string, test_sim_dur=test_sim_dur, test_epochs=test_epochs, mod="DMC")


    print(f'-> average_reward: {default_results["reward"].mean()}')
    print(f'-> sigma reward: {default_results["reward"].std()}')
    print(f'-> satisfied demand: {default_results["satisfied_demand"].mean()}')
    print(f'-> satisfied orders: {default_results["satisfied_orders"].mean()}')



def evaluation(mod, varied_parameter):

    # Initialize lists to store the results
    total_results_lr = []
    total_results_gamma = []
    total_results_memory_size = []
    total_results_batch_size = []
    total_results_replay_start_size = []
    total_results_synch_target_steps = []
    total_results_exp_min = []
    total_results_exp_max = []
    total_results_exp_decay = []
    total_results_threshold = []
    total_results_average_reward = []
    total_results_sigma_reward = []
    total_results_average_satisfied_orders = []
    total_results_average_satisfied_demand = []

    lr, gamma, memory_size, batch_size, replay_start_size, synch_target_steps, \
    exp_max, exp_min, exp_decay, threshold, list = get_parameters(parameter=varied_parameter)
    run = 0

    for i in list:

        if varied_parameter == "lr":
            lr = i
        elif varied_parameter == "gamma":
            gamma = i
        elif varied_parameter == "memory_size":
            memory_size = i
        elif varied_parameter == "batch_size":
            batch_size = i
        elif varied_parameter == "replay_start_size":
            replay_start_size = i
        elif varied_parameter == "synch_target_steps":
            synch_target_steps = i
        elif varied_parameter == "exp_max":
            exp_max = i
        elif varied_parameter == "exp_min":
            exp_min = i
        elif varied_parameter == "exp_decay":
            exp_decay = i
        elif varied_parameter == "threshold":
            threshold = i

        run += 1

        train_epochs = 50
        train_sim_dur = 200
        test_epochs = 50
        test_sim_dur = 200

        print("------------------------")
        print("Start Run: "+str(run))
        print(str(varied_parameter)+": "+str(i))

        string = "/Users/benedictrau/Documents/GitHub/Masterarbeit/SimulateAndLearn/"+\
                 "RL/Testing_Hyperparameter/DMC/results_NN_FS/"+str(mod)+"_"+str(varied_parameter)+\
                 "_"+str(i)+".pt"

        order_policy(learning_rate = lr, gamma = gamma, train_sim_dur=train_sim_dur, threshold=threshold,
                     train_epochs=train_epochs, batch_size = batch_size, exp_max=exp_max, exp_min = exp_min,
                     exp_decay = exp_decay, string = string, replay_start_size=replay_start_size,
                     synch_target_steps=synch_target_steps, memory_size=memory_size)

        #print("Test")
        results = order_policy_eval(string, test_sim_dur=test_sim_dur, test_epochs=test_epochs, mod=mod)

        total_results_average_reward.append(results["reward"].mean())
        total_results_sigma_reward.append(results["reward"].std())
        total_results_lr.append(lr)
        total_results_gamma.append(gamma)
        total_results_memory_size.append(memory_size)
        total_results_batch_size.append(batch_size)
        total_results_replay_start_size.append(replay_start_size)
        total_results_synch_target_steps.append(synch_target_steps)
        total_results_exp_min.append(exp_min)
        total_results_exp_max.append(exp_max)
        total_results_exp_decay.append(exp_decay)
        total_results_threshold.append(threshold)
        total_results_average_satisfied_orders.append(results["satisfied_orders"].mean())
        total_results_average_satisfied_demand.append(results["satisfied_demand"].mean())


    total_results = pd.DataFrame()
    total_results['lr'] = total_results_lr
    total_results['gamma'] = total_results_gamma
    total_results['memory_size'] = total_results_memory_size
    total_results['batch_size'] = batch_size
    total_results['replay_start_size'] = replay_start_size
    total_results['synch_target_steps'] = synch_target_steps
    total_results['exp_max'] = exp_max
    total_results['exp_min'] = exp_min
    total_results['exp_decay'] = exp_decay
    total_results['threshold'] = threshold
    total_results['avg_satisfied_orders'] = total_results_average_satisfied_orders
    total_results['avg_satisfied_demand'] = total_results_average_satisfied_demand
    total_results['sigma_reward_training'] = total_results_sigma_reward
    total_results['avg_reward_training'] = total_results_average_reward

    total_results.to_excel("results_NN_FS/Results_" + str(varied_parameter)+".xlsx", sheet_name="Results")

    print("------------------------")
    print("RESULTS")
    # show all columns
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(total_results)



start_proc = time.process_time()



varied_parameters = ['lr', 'gamma']

#varied_parameters = ['lr', 'gamma', 'memory_size', 'batch_size', 'replay_start_size',
#                     'synch_target_steps', 'exp_min', 'exp_decay', 'threshold]

get_default()
for param in varied_parameters:
    run = evaluation(mod="DMC", varied_parameter=param)


end_proc = time.process_time()
print('Required time: {:5.3f}s'.format(end_proc-start_proc))




