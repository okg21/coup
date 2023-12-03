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
    def __init__(self, state_dim, action_dim, learning_rate, gamma, name, is_main, epsilon_decay=0.99, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.name = name
        
        self.model = QNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.is_main = is_main
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        if is_main:
            self.replay_buffer = deque(maxlen=10000)

        self.list_of_actions = []
        self.did_action_lie = []

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


    def update(self, state, next_state, name, action, reward, done):
        game_state, history = state[0], state[1]
        next_game_state, next_history = next_state[0], next_state[1]

        state_tensor = state_to_input(game_state, history, name)  
        next_state_tensor = state_to_input(next_game_state, next_history, name)  

        # Pass the states through the Q-network to get the predicted Q-values
        predicted_values = self.model.forward(state_tensor)
        predicted_next_values = self.model.forward(next_state_tensor)

        # Get the Q-value of the chosen action
        q_value = predicted_values[action_to_index(action, game_state, name)]

        # Calculate the target Q-value using the Bellman equation
        if done:
            target_q_value = reward
        else:
            target_q_value = reward + self.gamma * torch.max(predicted_next_values)

        # Calculate the TD error
        td_error = target_q_value - q_value

        # Update the Q-value using gradient descent
        self.optimizer.zero_grad()
        loss = td_error.pow(2).mean()
        loss.backward()
        self.optimizer.step()

    def replay_experience(self, batch_size, name):
        # Sample a batch of experiences from the replay buffer
        batch = random.sample(self.replay_buffer, batch_size)

        # Update the Q-network with the sampled experiences
        for experience in batch:
            state, action, reward, next_state, done = experience
            self.update(state, next_state, name, action, reward, done)

    def add_experience(self, state, action, reward, next_state, done):
        # Add the experience to the replay buffer
        self.replay_buffer.append((state, action, reward, next_state, done))

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

def load_model(n, path):
    model = QNetwork(10 + 11 * n, 1 + 3 * n)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
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

        return reward

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
