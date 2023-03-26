# Import packages #
from SimulateAndLearn.RL.Sim_Env import InventorySystem

import numpy as np
import pandas as pd
import random
from PredictStock import XGB

import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque


# Define Model Parameters #
# Set whether to display on screen (slows model)
DISPLAY_ON_SCREEN = False
# Maximum number of game steps (state, action, reward, next state) to keep in memory
MEMORY_SIZE = 20000
# Number of game steps to play before starting training (all random actions)
REPLAY_START_SIZE = 200
# Time step between actions
TIME_STEP = 1
# Number of steps between policy -> target network update
SYNC_TARGET_STEPS = 200
# Exploration rate (epsilon) is probability of choosing a random action
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.1
# Reduction in epsilon with each game step
EXPLORATION_DECAY = 0.999
# Threshold to decide whether an action should be performed in order to maximize the reward or to reduce uncertainty
THRESHOLD = 0.5

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

### Prediction ###
PREDICTOR = 'XGB'

### Filename ###
RESULT_NAME = 'DMC'




# Define DQN #
class DQN(nn.Module):

    def __init__(self, observation_space, action_space, dropout_rate,
                 exploration_max, neurons_per_layer):

        # Starting with the maximal exploration rate when initializing the NN
        self.exploration_rate = exploration_max
        self.action_space = action_space

        # NN with two hidden layers
        # Sigmoid activation function
        # nn.Linear means we have a linear transformation of the incoming data [nn.Linear(in_feature, out_feature)]
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_space, neurons_per_layer),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(neurons_per_layer, neurons_per_layer),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(neurons_per_layer, neurons_per_layer),
            nn.LeakyReLU(),
            nn.Linear(neurons_per_layer, action_space)
        )

    def act(self, class_prob, threshold):

        # Decide if the taken action is random or determined by the NN according to the decreasing exploration rate
        if np.random.rand() < self.exploration_rate:
            action = random.randrange(self.action_space)
        else:
            # Calculate the entropie first
            entropie = 0
            class_probability = np.around(class_prob, 2)
            # Summing up the entropie using a for-loop
            for x in np.nditer(class_probability.T):
                if x != 0:
                    entropie += -np.log10(x)*x

            # Get relevant classes with a probability larger than zero
            relevant_classes = np.where(class_probability > 0)
            relevant_classes = relevant_classes[1]
            # Create an empty vector to store the votes in
            voting_vector = np.zeros((1, self.action_space))

            # Using a for-loop to get the votes of each relevant class
            for x in np.nditer(relevant_classes.T):
                x = np.array(np.array([[x]]))
                q_values = self.net(torch.FloatTensor(x))
                predicted_action = np.argmax(q_values.detach().numpy()[0])
                voting_vector[0][predicted_action] += 1

            action = np.argmax(voting_vector[0])

            # if the entropie is larger than the threshold add 1 to let the system perform a stock count to reduce uncertainty
            if entropie > threshold:
                if action % 2 == 0:
                    action += 1

        return action

    def forward(self, x):
        return self.net(x)

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))



def optimize(policy_net, target_net, memory, gamma, batch_size, exp_min, exp_decay):
    # policy network to predict best action (= best Q)
    # target network to provide target of Q for the selected next action
    BATCH_SIZE = batch_size
    EXPLORATION_MIN = exp_min
    EXPLORATION_DECAY = exp_decay
    # No optimization if there isn't enough data stored in the memory to sample a batch
    if len(memory) < BATCH_SIZE:
        return

    # Reduction of the exploration rate
    policy_net.exploration_rate *= EXPLORATION_DECAY
    # Ensure the exploration rate is at least the min exploration rate
    policy_net.exploration_rate = max(EXPLORATION_MIN, policy_net.exploration_rate)

    # Sample a random batch from memory
    batch = random.sample(memory, BATCH_SIZE)

    # create a dict to store all MSE calculated within one batch
    MSE = []
    step_MSE = 0

    for state, action, reward, state_next, terminal in batch:

        # Get the predicted rewards for the current state
        state_action_values = policy_net(torch.FloatTensor(state))

        # Get target Q for policy net update
        # Q = The expected future rewards discounted over time


        if not terminal:
            # 1. Get the same tensor as we receive it out of the policy net
            expected_state_action_values = policy_net(torch.FloatTensor(state))
            # Detach next state values from gradients to prevent updates
            expected_state_action_values = expected_state_action_values.detach()
            # Get next action with best Q from the policy net
            policy_next_state_values = policy_net(torch.FloatTensor(state_next))
            policy_next_state_values = policy_next_state_values.detach()
            best_action = np.argmax(policy_next_state_values[0].numpy())
            # Get the next s
            next_state_action_values = target_net(torch.FloatTensor(state_next))
            # Use detach again to prevent target net gradients being updated
            next_state_action_values = next_state_action_values.detach()
            best_next_q = next_state_action_values[0][best_action].numpy()
            # Get the next Q containing the reward and the discounted best next Q
            update_q = reward + (gamma * best_next_q)
            # Only the tensor that contains the action will be updated with the information out of the NN
            expected_state_action_values[0][action] = update_q


        else:
            continue


        # Set net to training mode
        policy_net.train()
        # Reset net gradients
        policy_net.optimizer.zero_grad()
        # calc loss with MSE
        loss_v = nn.MSELoss()(state_action_values, expected_state_action_values)
        # Back-propagate loss
        loss_v.backward()
        # Convert torch tensor to integer
        loss = loss_v.item()
        # Update network gradients
        policy_net.optimizer.step()
        # Append the calculated losses of the mini-batch to the MSE dict
        MSE.append(loss)
        # calculate the average loss of the batch to hand it over
        step_MSE = np.mean(MSE)


    return step_MSE



# Define memory class #
class Memory():

    """
    Replay memory used to train model.
    Limited length memory (using deque, double ended queue from collections).
      - When memory full deque replaces the oldest data with newest.
    Holds, state, action, reward, next state, and episode done.
    """

    def __init__(self, memory_size):
        """Constructor method to initialise replay memory"""
        self.memory = deque(maxlen=memory_size)

    def remember(self, state, action, reward, next_state, terminal):
        # add the SARNTR the memory
        self.memory.append((state, action, reward, next_state, terminal))




def predict(system_stock, last_stock_count):

    global class_prob, prediction


    if PREDICTOR == 'XGB':
        class_prob = XGB.class_probability(system_stock, last_stock_count)
        prediction = XGB.predict(system_stock, last_stock_count)

    return class_prob, prediction




# Define results plotting function #
def order_policy(learning_rate, gamma, train_sim_dur, train_epochs,  batch_size,
                 string, threshold=THRESHOLD, neurons_per_layer=32, dropout_rate=0.4,
                 exp_max=EXPLORATION_MAX, exp_min=EXPLORATION_MIN,
                 exp_decay=EXPLORATION_DECAY, replay_start_size=REPLAY_START_SIZE,
                 synch_target_steps=SYNC_TARGET_STEPS, memory_size=MEMORY_SIZE):


    global total_reward
    sim = InventorySystem(
        sim_duration = train_sim_dur,
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

    # Get observation and action size
    observation_space = 1
    action_space = sim.action_size
    dr = dropout_rate
    n_p_l = neurons_per_layer
    exploration_max = exp_max

    # Set up policy and target nets
    policy_net = DQN(observation_space, action_space, exploration_max=exploration_max,
                     neurons_per_layer=n_p_l, dropout_rate=dr)
    target_net = DQN(observation_space, action_space, exploration_max=exploration_max,
                     neurons_per_layer=n_p_l, dropout_rate=dr)

    # Set loss function and optimizer
    policy_net.optimizer = optim.Adam(params=policy_net.parameters(), lr=learning_rate)

    # Copy weights from policy_net to target
    target_net.load_state_dict(policy_net.state_dict())

    # Set target net to eval rater than training mode. We do not train the target net (it
    # is updated in intervals from the policy net)
    target_net.eval()

    # Set up the memory
    memory = Memory(memory_size=memory_size)

    ## 6.1 - Set up and start the training loop ##
    run = 0
    all_steps = 0
    continue_learning = True

    # Set up dicts to store the results of the different simulation runs
    results_run = []
    results_exploration = []
    results_score = []
    results_avg_MSE = []
    results_satisfied_orders = []
    results_satisfied_demand = []

    while continue_learning:

        # Count run
        run += 1
        # Reset sim env
        state = sim.reset()
        system_stock = state[0]
        last_stock_count = state[1]

        class_prob, predicted_class = predict(system_stock, last_stock_count)
        memory_state = np.array([[prediction]])

        # Set up trackers
        inventory = []
        rewards = []
        taken_actions = []
        total_MSE = []
        inventory_action = []
        total_reward = 0

        while True:

            all_steps +=1
            policy_net.eval()

            action = policy_net.act(class_prob, threshold=threshold)

            taken_actions.append(action)
            inventory_action.append([state[0], action])

            ## Play action ##
            # Get the results from the simulation with the respective action taken
            state_next, reward, terminal, info, KPI = sim.step(action)
            total_reward += reward

            # Store the simulation results
            inventory.append(state_next[0])
            rewards.append(reward)
            taken_actions.append(action)

            next_system_stock = state_next[0]
            next_last_stock_count = state_next[1]

            next_class_prob, next_stock_class = predict(next_system_stock, next_last_stock_count)
            memory_next_state = np.array([[next_stock_class]])

            # Display the simulation results of each run if needed (slows down the simulation)
            if DISPLAY_ON_SCREEN:
                sim.render()

            ## Add S/A/R/S/T/run to memory ##
            # Record state, action, reward, state_next, terminal and run to memory
            memory.remember(memory_state, action, reward, memory_next_state, terminal)
            # Update the state
            state = state_next
            class_prob = next_class_prob
            memory_state = memory_next_state

            # Update target net after a certain number of steps
            if len(memory.memory) > replay_start_size:
                #print(f'memory size: {len(memory.memory)}, ', end='')

                # Update policy net
                run_MSE = optimize(policy_net, target_net, memory.memory, gamma, batch_size,
                                   exp_min, exp_decay)
                #print(f'run_MSE: {run_MSE}')
                if run_MSE != None:
                    total_MSE.append(run_MSE)
                #print(f'all steps: {all_steps}, ', end='')

                # Update target net periodically
                if all_steps % synch_target_steps == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            # Actions to take at the end of gaming episode
            if terminal:
                # Get exploration rate
                exploration = policy_net.exploration_rate

                if len(total_MSE) == train_sim_dur:
                    average_MSE = np.mean(total_MSE)
                else:
                    average_MSE = 0

                fraction_of_satisfied_demand = KPI[0]
                fraction_of_satisfied_orders = KPI[1]


                # Add sim results after one simulation run to the results lists
                results_run.append(run)
                results_exploration.append(exploration)
                results_score.append(total_reward)
                results_avg_MSE.append(average_MSE)
                results_satisfied_orders.append(fraction_of_satisfied_orders)
                results_satisfied_demand.append(fraction_of_satisfied_demand)

                # Safe Model with the best reward
                total_reward = np.sum(rewards)

                # Check for end of learning. This will interrupt the training
                if run == train_epochs:
                    continue_learning = False

                # End episode loop
                break



    # Create data frame to store simulation results
    sim_details = pd.DataFrame()
    sim_details['run'] = results_run
    sim_details['exploration'] = results_exploration
    sim_details['reward'] = results_score
    sim_details['avgMSE'] = results_avg_MSE
    sim_details['satisfied_demand'] = results_satisfied_demand
    sim_details['satisfied_orders'] = results_satisfied_orders


    # Save the model
    torch.save(policy_net.state_dict(), string)

