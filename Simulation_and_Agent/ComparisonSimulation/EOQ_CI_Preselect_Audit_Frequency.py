###########################
### 1 - Import packages ###
###########################

from SimulateAndLearn.RL.Sim_Env import InventorySystem
from SimulateAndLearn.Simulate.EOQ import EOQ_Calculation


import numpy as np
import pandas as pd
import time
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



############################################
### 6 - Define results plotting function ###
############################################
def order_policy_eval(mod, audit_frequency, sim_dur, episodes):

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
    action_space = sim.action_size
    all_actions = sim.actions

    # Set up dicts to store the results of the different simulation runs
    results_run = []
    results_score = []
    results_fraction_of_satisfied_demand = []
    results_fraction_of_satisfied_orders = []
    results_average_inventory = []

    while continue_learning:

        # Count run
        run += 1
        # Reset sim env
        state = sim.reset()
        system_stock = state[0]
        last_stock_count = state[1]


        # Set up trackers
        inventory_system = []
        rewards = []
        taken_actions = []
        inventory_action = []
        total_reward = 0
        step = 0


        while True:

            global action
            step += 1

            # order parts (MUSS NICHT IN JEDER ITERATION ABGERUFEN WERDEN, dann muss man aber audit anders implementieren)
            if mod == "EOQ":
                if system_stock <= MEAN_DEMAND_SIZE:
                    action = EOQ_Calculation(COST_PER_ORDER=COST_PER_ORDER, COST_PER_ITEM= COST_PER_ITEM,
                                             MEAN_DEMAND_SIZE=MEAN_DEMAND_SIZE, COST_RATE_HOLDING=COST_RATE_HOLDING,
                                             BATCH_SIZE_ORDERS=BATCH_SIZE_ORDERS, SIM_DURATION=sim_dur,
                                             action_space=all_actions)
                else:
                    action = 0

            # Audit frequency
            if last_stock_count == audit_frequency:
                action += 1


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
                if run == episodes:
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



def plot_results(audit_frequency, average_reward, lower_bound, upper_bound):

    #plt.plot(audit_frequency, average_reward)
    fig, ax = plt.subplots()
    ax.plot(audit_frequency, average_reward)
    ax.fill_between(audit_frequency, lower_bound, upper_bound, color='b', alpha=.1)
    plt.ylabel("reward")
    plt.xlabel("audit frequency")
    plt.tight_layout()
    plt.show()


def sim_study(iterations, mod):

    # Initialize lists to store the results
    total_results_audit_frequency = []
    total_results_average_reward = []
    total_results_lower_bound = []
    total_results_upper_bound = []
    total_results_sigma_reward = []
    total_results_average_satisfied_orders = []
    total_results_average_satisfied_demand = []

    global best_result
    # initialize best results with an awful value to make sure it is updated after the first run
    best_result = -100000
    global best_result_sigma
    global best_audit_frequency

    for i in range(1, iterations):

        print("------------------------")
        print("Start Run: " + str(i))
        print(f'Audit frequency: {i}')
        total_results_audit_frequency.append((i))

        #
        test_epochs = 250
        test_sim_dur = 200

        results = order_policy_eval(mod=mod, audit_frequency=i,
                                    sim_dur=test_sim_dur, episodes=test_epochs)


        # Get the results from the simulation run
        total_results_average_reward.append(results["reward"].mean())
        total_results_sigma_reward.append(results["reward"].std())
        total_results_lower_bound.append(results["reward"].mean()-1.64 * results["reward"].std())
        total_results_upper_bound.append(results["reward"].mean()+1.64 * results["reward"].std())
        total_results_average_satisfied_orders.append(results["satisfied_orders"].mean())
        total_results_average_satisfied_demand.append(results["satisfied_demand"].mean())


        # If the average reward is better than the previous policies update the best results
        if results["reward"].mean() > best_result:
            best_result = results["reward"].mean()
            best_result_sigma = results["reward"].std()
            best_audit_frequency = i

    total_results = pd.DataFrame()
    total_results['audit_frequency'] = total_results_audit_frequency
    total_results['avg_satisfied_orders'] = total_results_average_satisfied_orders
    total_results['avg_satisfied_demand'] = total_results_average_satisfied_demand
    total_results['sigma_reward_training'] = total_results_sigma_reward
    total_results['lower_bound'] = total_results_lower_bound
    total_results['upper_bound'] = total_results_upper_bound
    total_results['avg_reward_training'] = total_results_average_reward

    print("------------------------")
    print("RESULTS")
    # show all columns
    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(total_results)

    print("Best parameters found after " + str(iterations) + " iterations with:")
    print(f'audit_frequency: {best_audit_frequency}')
    print(f'avg_reward: {best_result}')
    print(f'reward_std: {best_result_sigma}')

    total_results.to_excel("Results_Excel/" + mod + "_CI.xlsx", sheet_name="Results")

    plot_results(audit_frequency=total_results_audit_frequency, average_reward = total_results_average_reward,
                 lower_bound=total_results_lower_bound, upper_bound=total_results_upper_bound)


start_proc = time.process_time()

run = sim_study(iterations=200, mod="QR")

end_proc = time.process_time()
print('Required time: {:5.3f}s'.format(end_proc-start_proc))
