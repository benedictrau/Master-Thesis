###########################
### 1 - Import packages ###
###########################

from Simulation_and_Training.Reinforcement_Learning.Sim_Env import InventorySystem
from EOQ import EOQ_Calculation
from QR import QR_Calculation
from RL_NN import get_action
from PredictStock import XGB

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


# get the prediction of the current stock class and the probabilities of being in each class
def predict(system_stock, last_stock_count):

    class_prob = XGB.class_probability(system_stock, last_stock_count)
    prediction = XGB.predict(system_stock, last_stock_count)
    return class_prob, prediction


# Plot the deviation and action taken
def plot_results(run, deviation, action, action_taken):

    # plot to detect freezing
    fig, ax1 = plt.subplots()

    ax1.plot(run, deviation, color="blue")
    ax1.set_xlabel("simulation time [time units]")
    ax1.set_ylabel("stock deviation [SKU]")
    ax1.set_ylim(-30, 30)
    ax1.set_xlim(0, 200)

    ax2 = ax1.twinx()
    ax2.set_ylabel("action taken")
    ax2.scatter(action_taken, action, color="red")
    ax2.set_ylim(0, 12)
    ax1.grid(which='major', color='#DDDDDD', linewidth=0.8)
    # Show the minor grid as well. Style it in very light gray as a thin,
    # dotted line.
    ax2.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    # Make the minor ticks and gridlines show.
    ax1.minorticks_on()

    plt.show()



# plot to determine the transient time
def plot_transient_time(step_list, df, window, mod):


    # First, calculate the mean system inventory at each simulation step
    df['mean'] =df.mean(axis=1)
    # Then, calculate the rolling average
    df['rolling_average'] = df["mean"].rolling(window, min_periods=1).mean()

    plt.plot(step_list, df['rolling_average'])
    #plt.title("Moving average with window w = "+str(window) + " for mod "+str(mod))
    plt.ylabel("moving average Y [SKU] ")
    plt.xlabel("simulation time [time units]")
    plt.ylim(0, 60)
    plt.xlim(0,200)
    #plt.grid()
    plt.grid(which='major', color='#DDDDDD', linewidth=0.8)
    # Show the minor grid as well. Style it in very light gray as a thin,
    # dotted line.
    plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
    # Make the minor ticks and gridlines show.
    plt.minorticks_on()
    plt.show()


# plot to show the stock deviation
def plot_deviation_stock(run, system_stock, actual_stock, deviation, actions, actions_taken):


    ax = plt.subplot(211)
    ax.plot(run, system_stock, color="green", label="system stock")
    ax.plot(run, actual_stock, color="blue", label="actual stock")
    #plt.axis('equal')
    ax.legend(frameon=True, loc="lower right")

    ax.set_xlabel("simulation time [time units]")
    ax.set_ylabel("inventory [SKU]")
    ax.xlim(0, 200)

    ax1 = plt.subplot(212)
    ax2 = ax1.twinx()
    ax1.set_xlabel("simulation time [time units]")
    ax1.set_ylabel("inventory deviation [SKU]")
    ax1.plot(run, deviation, color="blue", label="inventory deviation")
    ax2.set_ylabel("action taken")
    ax2.scatter(actions_taken, actions, color="red", label="actions")
    ax2.set_ylim(0, 12)
    ax2.legend()
    ax1.legend()
    ax1.xlim(0,200)

    plt.tight_layout()
    plt.show()



# function used to conduct a simulation run
def order_policy(mod, sim_dur, sim_episodes, neurons_per_layer, string, audit_frequency, transient_time=0,
                 plot_action=False, plot_transient = False, plot_dev_stock = False, window=0):

    # initialize the parameters used in the simulation environment
    global total_reward
    sim = InventorySystem(
        sim_duration = sim_dur,
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
        batch_size = BATCH_SIZE_ORDERS,
        invisible_demand_size=INVISIBLE_DEMAND_SIZE,
        deviation_direction=DEVIATION_DIRECTION
        )

    ## 6.1 - Set up and start the training loop ##
    run = 0
    continue_learning = True
    action_space = sim.actions

    # Set up dicts to store the results of the different simulation runs
    results_run = []
    results_score = []
    results_fraction_of_satisfied_demand = []
    results_fraction_of_satisfied_orders = []
    results_average_inventory = []

    if plot_transient == True:
        df = pd.DataFrame()

    while continue_learning:

        # Count run
        run += 1
        step = 1
        sim_step = 0

        # Reset sim env before starting a new simulation run
        state = sim.reset()
        system_stock = state[0]
        last_stock_count = state[1]

        # Set up trackers
        step_list = []
        inventory_system = []
        inventory_actual = []
        deviation = []
        rewards = []
        taken_actions = []
        inventory_action = []
        actions = []
        actions_taken = []
        total_reward = 0


        while True:

            step += 1

            global action

            # get the action depending on the policy (mod) used
            if mod == "MLS":
                class_prob, prediction = predict(system_stock, last_stock_count)

                action = get_action(class_prob, prediction, mod, string, neurons_per_layer, THRESHOLD=0.5)

            elif mod == "QMDP":
                class_prob, prediction = predict(system_stock, last_stock_count)

                action = get_action(class_prob, prediction, mod, string, neurons_per_layer, THRESHOLD=0.5)

            elif mod == "DMC":
                class_prob, prediction = predict(system_stock, last_stock_count)

                action = get_action(class_prob, prediction, mod, string, neurons_per_layer, THRESHOLD=0.5)

            elif mod == "EOQ":
                if system_stock <= MEAN_DEMAND_SIZE:
                    action = EOQ_Calculation(COST_PER_ORDER = COST_PER_ORDER,
                                             COST_PER_ITEM = COST_PER_ITEM,
                                             MEAN_DEMAND_SIZE = MEAN_DEMAND_SIZE,
                                             COST_RATE_HOLDING = COST_RATE_HOLDING,
                                             BATCH_SIZE_ORDERS = BATCH_SIZE_ORDERS,
                                             SIM_DURATION = sim_dur,
                                             action_space = action_space)
                else:
                    action = 0

                if last_stock_count == audit_frequency:
                    action += 1



            elif mod == "QR":
                selected_action, ROP = QR_Calculation(COST_PER_ITEM = COST_PER_ITEM,
                                                      COST_PER_ORDER  =COST_PER_ORDER,
                                                      MEAN_DEMAND_SIZE = MEAN_DEMAND_SIZE,
                                                      SIGMA_DEMAND_SIZE = SIGMA_DEMAND_SIZE,
                                                      COST_RATE_HOLDING = COST_RATE_HOLDING,
                                                      COST_RATE_SHORTAGE = COST_RATE_SHORTAGE,
                                                      BATCH_SIZE_ORDERS = BATCH_SIZE_ORDERS,
                                                      SIM_DURATION = sim_dur,
                                                      action_space = action_space)

                if system_stock <= ROP:
                    action = selected_action

                else:
                    action = 0

                if last_stock_count == audit_frequency:
                    action += 1



            taken_actions.append(action)
            inventory_action.append([state[0], action])
            if action > 0:
                actions.append(action)
                actions_taken.append(step)

            ## Play action ##
            # Get the results from the simulation with the respective action taken
            state_next, reward, terminal, info, KPI = sim.step(action)

            # if the transient time is passed start storing the results
            if step >= transient_time:
                sim_step += 1
                total_reward += reward

                # Store the simulation results
                inventory_system.append(state_next[0])
                inventory_actual.append(state_next[2])
                rewards.append(reward)
                taken_actions.append(action)
                step_list.append(sim_step)
                deviation.append(state_next[0] - state_next[2])

            system_stock = state_next[0]
            last_stock_count = state_next[1]


            # Display the simulation results of each run if needed (slows down the simulation)
            if DISPLAY_ON_SCREEN:
                sim.render()

            # Actions to take at the end of gaming episode
            if terminal:

                # get the KPIs
                fraction_of_satisfied_demand = KPI[0]
                fraction_of_satisfied_orders = KPI[1]
                satisfied_demand = KPI[2]
                missed_demand = KPI[3]
                satisfied_orders = KPI[4]
                missed_orders = KPI[5]

                # calculate the average inventory
                average_inventory = np.mean(inventory_actual)

                # Clear print row content
                #print('------------------------')
                clear_row = '\r' + ' ' * 79 + '\r'
                print(clear_row, end='')
                print(f'Run: {run}', end='')
                #print(f'Total reward: {total_reward:4.1f}')

                # Add sim results after one simulation run to the results lists
                results_run.append(run)
                results_score.append(total_reward)
                results_fraction_of_satisfied_demand.append(fraction_of_satisfied_demand)
                results_fraction_of_satisfied_orders.append(fraction_of_satisfied_orders)
                results_average_inventory.append(average_inventory)

                # Get total reward
                total_reward = np.sum(rewards)

                # if the boolean variable plot_transient is set to True a dataframe is used to store the system inventory
                if plot_transient == True:
                    df.insert(loc=run-1, column=str(run), value=inventory_system)



                # Check for end of learning. This will interrupt the training
                if run == sim_episodes:
                    continue_learning = False

                    # depending on the variables plots are created
                    if plot_action == True:
                        plot_results(step_list, deviation, actions, actions_taken)

                    if plot_transient == True:
                        plot_transient_time(step_list, df, window, mod)
                        #df[run] = inventory_system

                    if plot_dev_stock == True:
                        plot_deviation_stock(step_list, inventory_system, inventory_actual, deviation, actions, actions_taken)

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


# get the string where the neural net is stored depending on the mod selected
def get_string(mod):

    global string_RL
    global neurons_per_layer

    if mod == "MLS":
        neurons_per_layer = 64
        #string_RL = "/Users/benedictrau/Documents/GitHub/Masterarbeit/SimulateAndLearn/RL/Testing_Hyperparameter/MLS/results_NN_Eva/MLS_lr_0.001_gamma_0.95_batch_16_exp_min_0.01_exp_decay_0.995.pt"
        string_RL = "/Users/benedictrau/Documents/GitHub/Masterarbeit/SimulateAndLearn/RL/results_NN/MLS_lr_0.0006000000000000001_gamma_0.9545454545454546_batch_128_replay_2000.0_dropout_0.2_neurons_64.pt"

    elif mod == "QMDP":
        neurons_per_layer = 128
        string_RL = "/Users/benedictrau/Documents/GitHub/Masterarbeit/SimulateAndLearn/RL/results_NN/QMDP_lr_0.0007000000000000001_gamma_0.8836363636363636_batch_16_replay_1000.0_dropout_0.3_neurons_128.pt"

    elif mod == "DMC":
        neurons_per_layer = 64
        string_RL = "/Users/benedictrau/Documents/GitHub/Masterarbeit/SimulateAndLearn/RL/results_NN/DMC_lr_0.00030000000000000003_gamma_0.99_batch_64_replay_2000.0_dropout_0.2_neurons_64_threshold_0.3.pt"

    else:
        neurons_per_layer = 0
        string_RL =""

    return string_RL, neurons_per_layer


# function to determine the transient time
def det_transient_time(mod, sim_dur, sim_episodes, audit_frequency, window):

    # depending on the mod get the string where the neural net is saved
    string_RL, neurons_per_layer = get_string(mod)
    order_policy(mod, sim_dur, sim_episodes, neurons_per_layer=neurons_per_layer, string=string_RL,
                 audit_frequency=audit_frequency, plot_transient = True, window=window, plot_action=False)

#det_transient_time(mod="DMC",
#                   sim_dur=200,
#                   sim_episodes=10,
#                   audit_frequency=203,
#                   window = 100)





# To run with one policy
def plot_deviation_inventory(mod, sim_dur, sim_episodes=1, audit_frequency=200):

    string_RL, neurons_per_layer = get_string(mod)
    order_policy(mod, sim_dur, sim_episodes=1, neurons_per_layer=neurons_per_layer, string=string_RL,
                 audit_frequency = audit_frequency, plot_dev_stock = True)

#plot_deviation_inventory(mod="EOQ",
#                        sim_dur = 200,
#                        audit_frequency=200)


# Function to print actions and stock deviation
def one_run(mod, sim_dur, audit_frequency, sim_episodes, transient_time=0):

    string_RL, neurons_per_layer = get_string(mod)
    print("---------------------")
    print(f'start mod: {mod}')

    sim_dur = sim_dur + transient_time

    result = order_policy(mod, sim_dur, sim_episodes=sim_episodes, neurons_per_layer=neurons_per_layer, string=string_RL,
                          audit_frequency = audit_frequency, plot_action = True, transient_time=transient_time)

    mean_reward = result["reward"].mean()
    std_reward = result["reward"].std()
    print(result)

    print(f'mod: {mod}')
    print(f'mean reward: {mean_reward}')
    print(f'std reward: {std_reward}')



#one_run(mod="EOQ",
#        sim_dur=200,
#        audit_frequency=14)


# Plot the confidence interval
def plot_confidence_interval(x, mean, sigma, horizontal_line_width=0.25, color='#2187bb'):
    print(f'mean: {mean}')
    print(f'sigma: {sigma}')
    confidence_interval = 1.96 * sigma
    left = x - horizontal_line_width / 2
    top = mean - confidence_interval
    right = x + horizontal_line_width / 2
    bottom = mean + confidence_interval
    print(f'top: {top}')
    print(f'bottom: {bottom}')
    plt.plot([x, x], [top, bottom], color=color)
    plt.plot([left, right], [top, top], color=color)
    plt.plot([left, right], [bottom, bottom], color=color)
    plt.plot(x, mean, 'o', color='#f44336')
    plt.ylabel("reward")
    #plt.xlabel("Policy")



# Function to conduct the simulation study and plot the confidence interval
def confidence_interval():


    df = pd.DataFrame()
    df["mod"] = ["MLS", "QMDP", "DMC", "EOQ", "EOQ", "QR", "QR"]
    df["sim_episodes"] = [82, 111, 100, 255, 306, 102, 41]
    df["audit_freq"] = [250, 250, 250, 250, 14, 250, 30]
    df["transient_time"] = [50, 50, 25, 0, 80, 0, 75]

    run = []
    run_mod = []

    for i in range(len(df["mod"])):
        mod = df.at[i, "mod"]
        sim_episodes = df.at[i, "sim_episodes"]
        audit_frequency = df.at[i, "audit_freq"]
        transient_time = df.at[i, "transient_time"]

        sim_dur = 200 + transient_time

        string_RL, neurons_per_layer = get_string(mod)

        print("-------------")
        print(f'mod: {mod}')

        result = order_policy(mod = mod, sim_dur=sim_dur, sim_episodes=sim_episodes,
                              neurons_per_layer=neurons_per_layer,
                              string = string_RL, audit_frequency=audit_frequency,
                              transient_time=transient_time)


        plot_confidence_interval(x=i+1, mean=result["reward"].mean(),
                                 sigma=result["reward"].std())

        run.append(i+1)
        if audit_frequency < 200:
            run_mod.append(mod + " + Audit")
        else:
            run_mod.append(mod)

    plt.xticks(run, run_mod)
    plt.ylabel("reward")
    plt.title("Confidence Interval")
    plt.tight_layout()
    plt.show()


#confidence_interval()



# Function to determine the number of replications required
def prerun():

    df = pd.DataFrame()
    df["mod"] = ["MLS", "QMDP", "DMC", "EOQ", "EOQ", "QR", "QR"]
    df["sim_episodes"] = [20, 20, 20, 20, 20, 20, 20]
    df["audit_freq"] = [250, 250, 250, 250, 14, 250, 30]
    df["transient_time"] = [50, 50, 25, 0, 80, 0, 75]
    #df["mod"] = ["MLS"]
    #df["sim_episodes"] = [10]
    #df["audit_freq"] = [250]
    #df["transient_time"] = [50]

    for i in range(len(df["mod"])):
        mod = df.at[i, "mod"]
        sim_episodes = df.at[i, "sim_episodes"]
        audit_frequency = df.at[i, "audit_freq"]
        transient_time = df.at[i, "transient_time"]

        one_run(mod=mod, sim_dur=200, audit_frequency=audit_frequency, sim_episodes=sim_episodes, transient_time=transient_time)

#prerun()