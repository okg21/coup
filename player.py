from utils import *

import random

class Player():
    def __init__(self, name, funcs, agent=None):
        self.name = name

        # player logic
        self.decision_fn = funcs['decision_fn']
        self.block_fn = funcs['block_fn']
        self.dispose_fn = funcs['dispose_fn']
        self.keep_fn = funcs['keep_fn']
        self.agent = agent

    def decision(self, game_state, history):
        """
        Return an action of the form (sender, reciever, type) based on the game_state and history of the game. 
        
        sender = this player (always)
        reciever = any player
        type = Unblockable, Duke, Assassin, Captain, Ambassador, Contessa
        """
        if self.agent is None:
            return self.decision_fn(game_state, history, self.name)
        else:
            return self.decision_fn(game_state, history, self.name, self.agent, self.agent.history_length)
    
    def block(self, action, game_state, history, action_is_block=False):
        """
        Return a block of the form (blocker, attempt_block, lie_or_counter) based on the action, game_state, and history of the game.
        
        blocker = this player (always)
        attempt_block = boolean value
        lie_or_counter = boolean value; True if calling the action a lie, False if claiming a counter role
        """
        return self.block_fn(action, game_state, history, self.name, action_is_block=action_is_block)
    
    def dispose(self, game_state, history):
        """
        Return the index of the card from self.cards to remove given the fact that a card must be removed based on the game_state and history of the game.
        """
        return self.dispose_fn(game_state, history, self.name)
    
    def keep(self, cards, game_state, history):
        """
        Return the indices of the cards to keep from cards based on the game_state and history of the game.
        """
        return self.keep_fn(cards, game_state, history, self.name)
    

# -- Random Ronald -- #
def random_decision(game_state, history, name):
    return random.choice(generate_all_action(game_state['current_player'], game_state['players'], game_state['player_coins'], game_state['player_cards']))

def random_block(action, game_state, history, name, action_is_block=False):
    if action_is_block:
        if action[2]:
            action = (action[0], name, 'Lie_Block')
        else:
            action = (action[0], name, 'Role_Block')
    return random.choice(generate_all_blocks(name, action))

def random_dispose(game_state, history, name):
    return random.randint(0, len(game_state['player_cards'][name]) - 1)

def random_keep(cards, game_state, history, name):
    return random.sample(list(range(len(cards))), k=len(cards)-2)

RANDOM_FUNCS = {'decision_fn': random_decision, 'block_fn': random_block, 'dispose_fn': random_dispose, 'keep_fn': random_keep}

# -- Truth Teller Tim -- #
def truth_decision(game_state, history, name):
    player_cards = game_state['player_cards'][name]
    possible_actions = generate_all_action(game_state['current_player'], game_state['players'], game_state['player_coins'], game_state['player_cards'])
    truth_actions = [action for action in possible_actions if action[2] not in ACTION_SENDER.keys() or ACTION_SENDER[action[2]] in player_cards]
    return random.choice(truth_actions)

def truth_block(action, game_state, history, name, action_is_block=False):
    player_cards = game_state['player_cards'][name]
    if action_is_block:
        return (name, False, None)
    else:
        if ROLE_BLOCKABLE[action[2]] and not did_block_1_lie(action[2], player_cards):
            return (name, True, False)
        else:
            return (name, False, None)

TRUTH_FUNCS = {'decision_fn': truth_decision, 'block_fn': truth_block, 'dispose_fn': random_dispose, 'keep_fn': random_keep}

# -- Greedy Garrett -- #

def greedy_decision(game_state, history, name):
    player_cards = game_state['player_cards'][name]
    possible_actions = generate_all_action(game_state['current_player'], game_state['players'], game_state['player_coins'], game_state['player_cards'])
    greedy_actions = [action for action in possible_actions if action[2] == 'Income' or action[2] == 'Foreign Aid' or action[2] == 'Tax' or action[2] == 'Assassinate' or action[2] == 'Coup']
    return random.choice(greedy_actions)

def greedy_block(action, game_state, history, name, action_is_block=False):
    player_cards = game_state['player_cards'][name]
    if action_is_block:
        return (name, False, None)
    else:
        if action[1] == name:
            if ROLE_BLOCKABLE[action[2]] and not did_block_1_lie(action[2], player_cards):
                return (name, True, False)
            elif LIE_BLOCKABLE[action[2]]:
                return (name, True, True)
            elif ROLE_BLOCKABLE[action[2]]:
                return (name, True, False)
        return (name, False, None)

GREEDY_FUNCS = {'decision_fn': greedy_decision, 'block_fn': greedy_block, 'dispose_fn': random_dispose, 'keep_fn': random_keep}

# -- Income Isabel -- #

def income_decision(game_state, history, name):
    player_cards = game_state['player_cards'][name]
    possible_actions = generate_all_action(game_state['current_player'], game_state['players'], game_state['player_coins'], game_state['player_cards'])
    income_actions = [action for action in possible_actions if action[2] == 'Income' or action[2] == 'Assassinate']
    if income_actions:
        return random.choice(income_actions)
    else:
        return random.choice(possible_actions)
    
def income_block(action, game_state, history, name, action_is_block=False):
    player_cards = game_state['player_cards'][name]
    if action_is_block:
        return (name, False, None)
    else:
        if action[1] == name and ROLE_BLOCKABLE[action[2]] and not did_block_1_lie(action[2], player_cards):
            return (name, True, False)
    return (name, False, None)

INCOME_FUNCS = {'decision_fn': income_decision, 'block_fn': income_block, 'dispose_fn': random_dispose, 'keep_fn': random_keep}

# -- User Ulysses -- #
def user_decision(game_state, history, name):
    print("It's your turn, {}".format(name))
    print("Available actions:")
    possible_actions = generate_all_action(game_state['current_player'], game_state['players'], game_state['player_coins'], game_state['player_cards'])
    for i, action in enumerate(possible_actions):
        print("{}. {}".format(i + 1, action))
    
    while True:
        choice = input("Choose an action (enter the number): ")
        try:
            index = int(choice) - 1
            if index >= 0 and index < len(possible_actions):
                return possible_actions[index]
        except ValueError:
            pass

        print("Invalid choice. Please try again.")

def user_block(action, game_state, history, name, action_is_block=False): 
    if not action_is_block:
        print("Do you want to block? (Y/N): ")
        while True:
            choice = input().lower()
            if choice == 'y':
                print("Do you block by calling bluff or counter action (0/1): ")
                while True:
                    choice = input()
                    if choice == '0':
                        return (name, True, True)
                    elif choice == '1':
                        return (name, True, False)
                    else:
                        print("Invalid choice. Please enter 0 or 1.")
            elif choice == 'n':
                return (name, False, None)
            else:
                print("Invalid choice. Please enter Y or N.")
    elif not action[2]:
        print("Do you want to call their bluff? (Y/N)")
        while True:
            choice = input().lower()
            if choice == 'y':
                return (name, True, True)
            elif choice == 'n':
                return (name, False, None)
            else:
                print("Invalid choice. Please enter Y or N.")
    else:     
        return (name, False, None)

def user_dispose(game_state, history, name):
    print("Your cards: {}".format(game_state['player_cards'][name]))
    print("Choose a card to dispose (enter the index): ")
    while True:
        choice = input()
        try:
            index = int(choice)
            if index >= 0 and index < len(game_state['player_cards'][name]):
                return index
        except ValueError:
            pass

        print("Invalid choice. Please try again.")

def user_keep(cards, game_state, history, name):
    print("Your cards: {}".format(cards))
    print("Choose the indices of the cards to keep (enter as comma-separated values): ")
    while True:
        choice = input()
        indices = [int(idx) for idx in choice.split(",") if idx.strip().isdigit()]
        if len(indices) == len(game_state['player_cards'][name]) and all(idx >= 0 and idx < len(cards) for idx in indices):
            return indices

        print("Invalid choice. Please try again.")

USER_FUNCS = {'decision_fn': user_decision, 'block_fn': user_block, 'dispose_fn': user_dispose, 'keep_fn': user_keep}