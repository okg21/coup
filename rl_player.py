import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import copy
import time

from player import *
from game import *

ROLE_TO_I = {'Duke' : 0, 'Assassin': 1, 'Captain': 2, 'Ambassador': 3, 'Contessa': 4}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def state_to_input(game_state, history, name, role_to_i=ROLE_TO_I):
    """
    Returns a one-hot-encoding of the game_state and history to be used as the input of the model. game_state and history
    are used to get the information required as input.

    input will be a Tensor of shape (10 + 11n)

    The first 10 entries will be in 2 groups of 5, representing the cards the player has
    The next n entries will be the number of coins each player has (normalized by dividing by 12, the max number of possible coins)
    The next 10n entries will be in 2n groups of 5, representing the dead cards each player has revealed
    """
    players, deck, player_cards, player_deaths, player_coins, current_player = game_state['players'], game_state['deck'], game_state['player_cards'], game_state['player_deaths'], game_state['player_coins'], game_state['current_player']
    our_cards = player_cards[name]
    n = len(player_deaths.keys())
    player_names = list(player_deaths.keys())

    # initialize input of zeros
    input = torch.zeros(10 + 11 * n)

    # fill first 10 entries with information about our_cards
    if len(player_cards[name]) > 0:
        input[role_to_i[our_cards[0]]] = 1
    if len(player_cards[name]) > 1:
        input[role_to_i[our_cards[1]] + 5] = 1

    # fill next n entries with information about player_coins
    for i in range(n):
        player_name = player_names[i]
        input[10 + i] = player_coins[player_name] / 12

    # fill next 10n entries with information about player_deaths
    for i in range(n):
        player_name = player_names[i]
        if len(player_deaths[player_name]) > 0:
            input[10 + n + 10 * i + role_to_i[player_deaths[player_name][0]]] = 1
        if len(player_deaths[player_name]) > 1:
            input[10 + n + 10 * i + 5 + role_to_i[player_deaths[player_name][1]]] = 1

    return input.float().to(device)

def output_to_action(output, game_state, name):
    """
    Returns the action represented by the largest value of the actions encoded in output. If this action is
    not possible, return the next highest action such that it is possible.
    
    Let n = #players (> 1). Then,

    0                   -> Income
    1                   -> Foreign Aid
    2                   -> Tax
    3                   -> Exchange
    4, ..., 2 + n       -> Steal
    3 + n, ..., 1 + 2n  -> Assassinate
    2 + 2n, ..., 3n     -> Coup

    If i > 3, then the reciever is the player corresponding to index (i - 4) mod (n - 1).
    Otherwise, the reciever is name.
    """
    possible_actions = generate_all_action(game_state['current_player'], game_state['players'], game_state['player_coins'], game_state['player_cards'])
    n = len(game_state['player_deaths'].keys())

    list_of_players = list([p_name for p_name in game_state['player_deaths'].keys() if p_name != name])
    i_to_player = {i : list_of_players[i] for i in range(len(list_of_players))}

    i_to_type = {0: 'Income', 1: 'Foreign Aid', 2: 'Tax', 3: 'Exchange'}
    for i in range(4, 3 + n):
        i_to_type[i] = 'Steal'
    for i in range(3 + n, 2 + 2 * n):
        i_to_type[i] = 'Assassinate'
    for i in range(2 + 2 * n, 1 + 3 * n):
        i_to_type[i] = 'Coup'

    action = None
    while action not in possible_actions:
        i = torch.argmax(output).item()
        output[i] = -1 * float('inf')
        type = i_to_type[i]
        if i > 3:
            reciever = i_to_player[(i - 4) % (n - 1)]
        else:
            reciever = name
        action = (name, reciever, type)
    return action

def action_to_index(action, game_state, name):
    n = len(game_state['player_deaths'].keys())
    type_to_i = {'Income': 0, 'Foreign Aid': 1, 'Tax': 2, 'Exchange': 3, 'Steal': 4, 'Assassinate': 3 + n, 'Coup': 2 + 2 * n}
    
    list_of_players = list([p_name for p_name in game_state['player_deaths'].keys() if p_name != name])
    player_to_i = {list_of_players[i] : i for i in range(len(list_of_players))}

    i_0 = type_to_i[action[2]]
    if i_0 > 3:
        i = i_0 + player_to_i[action[1]]
        return i
    else:
        return i_0

class QLearningAgent:
    def __init__(self, state_dim, action_dim, learning_rate, gamma, name, is_main,
                 target_update_freq=100, epsilon_decay=0.99, epsilon_min=0.01
                 ,h_dim=128, h_layers=2, tau=0.01, buffer_size=1000000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau = tau
        self.name = name
        self.is_main = is_main
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.num_param_updates = 0
        self.target_update_freq = target_update_freq
        
        #creating the model
        self.model = QNetwork(state_dim, action_dim, h_dim, h_layers).to(device)
        self.target_model = QNetwork(state_dim, action_dim, h_dim, h_layers).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        if is_main:
            self.replay_buffer = deque(maxlen=buffer_size)
            self.priorities = []

        self.list_of_actions = []
        self.did_action_lie = []

        #for reward normalization
        self.total_reward = 0
        self.count = 0
        self.mean_reward = 0
        self.var_reward = 0

    def update_reward_stats(self, reward):
        self.total_reward += reward
        self.count += 1
        new_mean = self.total_reward / self.count
        self.var_reward = ((self.var_reward * (self.count - 1)) + (reward - self.mean_reward) * (reward - new_mean)) / self.count
        self.mean_reward = new_mean

    def normalize_reward(self, reward):
        if self.var_reward > 0:
            normalized_reward = (reward - self.mean_reward) / (self.var_reward ** 0.5)
        else:
            normalized_reward = reward - self.mean_reward
        return normalized_reward



    def get_action(self, state, name, epsilon):
        game_state, history = state[0], state[1]

        state = state_to_input(game_state, history, name)

        action_values = self.model.forward(state)

        if torch.rand(1) < epsilon:
            possible_actions = generate_all_action(game_state['current_player'], game_state['players'], game_state['player_coins'], game_state['player_cards'])
            action = random.choice(possible_actions)
        else:
            action = output_to_action(action_values, game_state, name)
        return action

    def update_batch(self, states, next_states, names, actions, rewards, dones, indices=None):
        # Convert lists of states, next_states, etc., into batch tensors
        state_tensors = [state_to_input(game_state, history, name) for (game_state, history), name in zip(states, names)]
        next_state_tensors = [state_to_input(game_state, history, name) for (game_state, history), name in zip(next_states, names)]

        # Convert lists into PyTorch tensors
        state_batch = torch.stack(state_tensors).to(device)
        next_state_batch = torch.stack(next_state_tensors).to(device)
        action_batch = torch.tensor([action_to_index(action, game_state, name) for action, (game_state, _), name in zip(actions, states, names)], device=device)
        reward_batch = torch.tensor(rewards, device=device)
        done_batch = torch.tensor(dones, device=device)
        
        # Pass the batches through the Q-network
        predicted_values = self.model.forward(state_batch)
        predicted_next_values = self.target_model.forward(next_state_batch)

        # Select the Q-values for the chosen actions
        q_values = predicted_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)

        # Calculate the target Q-values
        max_next_q_values = torch.max(predicted_next_values, 1)[0]
        target_q_values = reward_batch + self.gamma * max_next_q_values * (1 - done_batch.float())

        # Calculate TD error
        td_errors = target_q_values - q_values
        
        #clip the td errors
        td_errors = torch.clamp(td_errors, -1, 1)


        # Update priorities in replay buffer
        if indices is not None:
            for idx, td_error in zip(indices, td_errors.cpu().detach().numpy()):
                self.priorities[idx] = abs(td_error) + 1e-5  # Add a small value to avoid zero priority


        # Update the Q-values using gradient descent
        self.optimizer.zero_grad()
        loss = td_errors.pow(2).mean()
        loss.backward()
        self.optimizer.step()
        
        self.num_param_updates += 1

        # Periodically update the target network by Q network to target Q network
        if self.num_param_updates % self.target_update_freq == 0:
            self.soft_update()
            
    def soft_update(self):
        for target_param, main_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * main_param.data + (1.0 - self.tau) * target_param.data)
            
    def replay_experience(self, batch_size, name):
        # Compute probabilities for each experience
        priorities_sum = sum(self.priorities)
        probabilities = [priority / priorities_sum for priority in self.priorities]

        # Sample experiences based on their probabilities
        indices = np.random.choice(range(len(self.replay_buffer)), size=batch_size, p=probabilities)
        batch = [self.replay_buffer[idx] for idx in indices]

        # Unpack the experiences
        states, actions, rewards, next_states, dones = zip(*batch)

        # Update the Q-network with the batched experiences
        self.update_batch(states, next_states, [name] * batch_size, actions, rewards, dones, indices)

    def add_experience(self, state, action, reward, next_state, done):
        # Add the experience to the replay buffer
        max_priority = max(self.priorities, default=1)
        self.replay_buffer.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

def load_model(n, path):
    model = QNetwork(10 + 11 * n, 1 + 3 * n)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, h_dim, h_layers=1):
        super(QNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.inter_layers = []
        
        self.fc_in = nn.Linear(state_dim, h_dim)
        for i in range(h_layers):
            self.inter_layers.append(nn.Linear(h_dim, h_dim).to(device))
        self.fc_out = nn.Linear(h_dim, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc_in(x))
        for layer in self.inter_layers:
            x = torch.relu(layer(x))
        x = self.fc_out(x)
        return x

def rltraining_decision(game_state, history, name, agent): #be careful not calling this from the main agent, since it needs to explore
    if agent.is_main:
        action = agent.get_action((game_state, history), name, agent.epsilon)
    else:
        action = agent.get_action((game_state, history), name, 0)    
        
    return action


class Environment():
    def __init__(self, name, players):
        self.name = name
        self.players = players
        
        self.game = Game(self.players)


    def step(self, reward_dict):
        """
        Return (next_state, reward, done).
        
        next_state = (next_game_state, next_history)
        reward = self.calculate_reward(state, next_state)
        done = True if agent wins / loses
        """
        game_state = self.game.game_state
        history = self.game.history
        state = (game_state.copy(), history.copy())

        action = self.game.simulate_turn()        
        i = len(self.game.game_state['players']) - 1

        while i > 0 and self.game.game_state['current_player'].name != self.name:
            _ = self.game.simulate_turn()
            i -= 1   

        next_game_state = self.game.game_state
        next_history = self.game.history
        next_state = (next_game_state.copy(), next_history.copy())

        reward = self.calculate_reward(state, next_state, reward_dict)

        done = all(self.name == p for p in [p.name for p in next_game_state['players']]) or self.name not in [p.name for p in next_game_state['players']]

        return (action, next_state, reward, done)

    def get_main_agent(self):
        return self.players[0].agent
    
    
    def calculate_reward(self, state, next_state, reward_dict):
        """
        Calculate the reward from going from state to next_state. 

        + 1 per change in amount of owned coins
        + 10 per change in amount of opponent's cards
        - 10 if you lose a card
        + 5 if 2 of the same card is diversified via exchange
        + 200 if win
        - 200 if lose
        """
        #parse reward dict
        COIN_VALUE = reward_dict['COIN_VALUE']
        CARD_VALUE = reward_dict['CARD_VALUE']
        WIN_VALUE = reward_dict['WIN_VALUE']
        CARD_DIVERSITY_VALUE = reward_dict['CARD_DIVERSITY_VALUE']
        
        game_state, history = state
        next_game_state, next_history = next_state

        reward = 0

        change_in_coins = next_game_state['player_coins'][self.name] - game_state['player_coins'][self.name]
        reward += COIN_VALUE * change_in_coins

        change_in_opponents_cards = sum([len(next_game_state['player_cards'][p]) for p in next_game_state['player_cards'].keys() if p != self.name]) - sum([len(game_state['player_cards'][p]) for p in game_state['player_cards'].keys() if p != self.name])
        change_in_owned_cards = len(next_game_state['player_cards'][self.name]) - len(game_state['player_cards'][self.name])

        reward += -1 * CARD_VALUE * change_in_opponents_cards
        reward += CARD_VALUE * change_in_owned_cards

        change_in_diversity = len(set(next_game_state['player_cards'][self.name])) - len(set(game_state['player_cards'][self.name]))
        reward += CARD_DIVERSITY_VALUE * change_in_diversity

        if all(self.name == p for p in [p.name for p in next_game_state['players']]):
            reward += WIN_VALUE
        elif self.name not in [p.name for p in next_game_state['players']]:
            reward += -1 * WIN_VALUE

        #reward normalization
        agent = self.get_main_agent()
        agent.update_reward_stats(reward)
        normalized_reward = agent.normalize_reward(reward)
        
        return normalized_reward

    def reset(self):
        """
        Reset the game to an initial game_state and clear the history. Return the initial_state.

        initial_state = (initial_game_state, initial_history)
        """
        self.game = Game(self.players)

        initial_game_state = self.game.game_state
        initial_history = self.game.history
        initial_state = (initial_game_state, initial_history)

        return initial_state
