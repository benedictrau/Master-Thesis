import torch
import torch.nn as nn
import numpy as np

features = 1
action_space = 12

# Define the neural network
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


# function used to get the action using a policy trained by a RL agent
def get_action(class_probability, prediction, mod, string, neurons_per_layer, THRESHOLD):

    global action

    # depending on the policy the neural net is initialized and the action is determined
    if mod == "MLS":
        features = 1
        action_space = 12
        model = NN(features, action_space, neurons_per_layer)
        model.load_state_dict(torch.load(string))
        model.eval()

        state = np.array([[prediction]])
        q_values = model.net(torch.FloatTensor(state))
        action = np.argmax(q_values.detach().numpy()[0])


    elif mod == "QMDP":
        features = 1
        action_space = 12
        model = NN(features, action_space, neurons_per_layer)
        model.load_state_dict(torch.load(string))
        model.eval()

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


    elif mod == "DMC":
        features = 1
        action_space = 12
        model = NN(features, action_space, neurons_per_layer)
        model.load_state_dict(torch.load(string))
        model.eval()

        entropie = 0
        class_probability = np.around(class_probability, 2)
        for x in np.nditer(class_probability.T):
            if x != 0:
                entropie += -np.log10(x) * x

        # Get relevant classes with a probability larger than zero
        relevant_classes = np.where(class_probability > 0)
        relevant_classes = relevant_classes[1]
        # Create an empty vector to store the votes in
        voting_vector = np.zeros((1, action_space))

        # Using a for-loop to get the votes of each relevant class
        for x in np.nditer(relevant_classes.T):
            x = np.array(np.array([[x]]))
            q_values = model.net(torch.FloatTensor(x))
            predicted_action = np.argmax(q_values.detach().numpy()[0])
            voting_vector[0][predicted_action] += 1

        action = np.argmax(voting_vector[0])

        # if the entropie is larger than the threshold increase the action value by one to let the system perform a stock count to reduce uncertainty
        if entropie > THRESHOLD:
            if action % 2 == 0:
                action += 1


    return action