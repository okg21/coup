from utils import *
import random

# BUGS: payments might not go through if an action is blocked, other edge cases might be weird and still need to be checked

class Game():
    def __init__(self, players, debug=False):
        self.game_state = self.initial_gamestate(players)
        self.history = []
        self.debug = debug

    def initial_gamestate(self, players):
        assert len(players) <= 6

        deck = ROLES * 3
        random.shuffle(deck)

        player_cards = {}
        player_deaths = {}
        player_coins = {}
        for player in players:
            player_cards[player.name] = [deck.pop(), deck.pop()]
            player_deaths[player.name] = []
            player_coins[player.name] = 2

        current_player = players[0]

        game_state = {'players': players,
                      'deck': deck,
                      'player_cards': player_cards,
                      'player_deaths': player_deaths,
                      'player_coins': player_coins,
                      'current_player': current_player}
        
        return game_state
    
    def unpack_gamestate(self):
        return self.game_state['players'], self.game_state['deck'], self.game_state['player_cards'], self.game_state['player_deaths'], self.game_state['player_coins'], self.game_state['current_player']
    
    def simulate_turn(self):
        players, deck, player_cards, player_deaths, player_coins, current_player = self.unpack_gamestate()
        action, block_1, block_2 = (None, None, None), (None, None, None), (None, None, None)

        action = current_player.decision(self.game_state, self.history)
        
        for player in [p for p in players if p.name != current_player.name]:
            potential_block_1 = player.block(action, self.game_state, self.history)
            if potential_block_1[1]:
                block_1 = potential_block_1
                break
        
        if block_1[0]:
            for player in [p for p in players if p.name != block_1[0]]:
                potential_block_2 = player.block(block_1, self.game_state, self.history, action_is_block=True)
                if potential_block_2[1]:
                    block_2 = potential_block_2
                    break

        turn = (action, block_1, block_2)
        self.history.append((self.game_state, turn))

        if self.debug:
            self.display_game_state()
            self.display_turn(turn)

        self.apply_game_logic(turn, players, player_cards, player_deaths)
        return action

    def apply_game_logic(self, turn, players, player_cards, player_deaths):
        action, block_1, block_2 = turn

        type = action[2]

        if block_1[1]:
            if block_2[1]:
                blocker_cards = player_cards[block_1[0]]
                if did_block_1_lie(type, blocker_cards):
                    card_idx = [p for p in players if p.name == block_1[0]][0].dispose(self.game_state, self.history)
                    lose_block(block_1[0], player_cards, card_idx, player_deaths)
                    self.take_action(action)
                else:
                    card_idx = [p for p in players if p.name == block_2[0]][0].dispose(self.game_state, self.history)
                    lose_block(block_2[0], player_cards, card_idx, player_deaths)
            else:
                if block_1[2]:
                    sender_cards = player_cards[action[0]]
                    if did_action_lie(type, sender_cards):
                        card_idx = [p for p in players if p.name == action[0]][0].dispose(self.game_state, self.history)
                        lose_block(action[0], player_cards, card_idx, player_deaths)
                    else:
                        card_idx = [p for p in players if p.name == block_1[0]][0].dispose(self.game_state, self.history)
                        lose_block(block_1[0], player_cards, card_idx, player_deaths)
                        self.take_action(action)
        else:
            self.take_action(action)

        self.update_next_player()

    def take_action(self, action):
        players, deck, player_cards, player_deaths, player_coins, current_player = self.unpack_gamestate()
        player1_name, player2_name, type = action

        if type == 'Income':
            income(player1_name, player_coins)
        elif type == 'Foreign Aid':
            foreign_aid(player1_name, player_coins)
        elif type == 'Tax':
            tax(player1_name, player_coins)
        elif type == 'Steal':
            steal(player1_name, player2_name, player_coins)
        elif type == 'Coup':
            card_idx = [p for p in players if p.name == player2_name][0].dispose(self.game_state, self.history)
            coup(player1_name, player2_name, player_coins, player_cards, card_idx, player_deaths)
        elif type == 'Assassinate' and len(player_cards[player2_name]) > 0:
            card_idx = [p for p in players if p.name == player2_name][0].dispose(self.game_state, self.history)
            assassinate(player1_name, player2_name, player_coins, player_cards, card_idx, player_deaths)
        elif type == 'Exchange':
            cards = player_cards[current_player.name].copy() + [deck.pop(), deck.pop()]
            cards_idxs = current_player.keep(cards, self.game_state, self.history)
            assert len(cards_idxs) == len(player_cards[player1_name])
            exchange(player1_name, player_cards, cards, cards_idxs, deck)

    def update_next_player(self):
        self.game_state['players'] = self.game_state['players'][1:] + [self.game_state['players'][0]]
        self.game_state['players'] = [p for p in self.game_state['players'] if len(self.game_state['player_cards'][p.name]) > 0]
        self.game_state['current_player'] = self.game_state['players'][0]

    def display_game_state(self):
        players, deck, player_cards, player_deaths, player_coins, current_player = self.unpack_gamestate()

        print("----- COUP -----")
        if self.debug:
            for p in player_cards:
                print(p, player_cards[p])
        print("player_coins:", player_coins)
        print("player_deaths:", player_deaths)
        print("current_player:", current_player.name)

    def display_turn(self, turn):
        action, block_1, block_2 = turn
        print("{} has taken the action {} on {}".format(action[0], action[2], action[1]))
        if block_1[1]:
            if block_1[2]:
                print("{} blocks by calling bluff.".format(block_1[0]))
            else:
                print("{} blocks by counter action.".format(block_1[0]))
        if block_2[1]:
            print("{} blocks by calling bluff.".format(block_2[0]))

    def toggle_debug(self):
        self.debug = not self.debug