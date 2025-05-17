from flask import Flask, render_template, jsonify, request
from player import *
from game import *
from utils import did_action_lie, did_block_1_lie
import json

app = Flask(__name__, static_url_path='', static_folder='static', template_folder='templates')

# Game state
game = None
game_log = []
previous_state = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/new_game', methods=['POST'])
def new_game():
    global game, game_log, previous_state
    game_log = []
    previous_state = None
    
    # Setup players based on request data
    data = request.json
    players = []
    
    for player_data in data.get('players', []):
        player_type = player_data.get('type', 'RANDOM')
        name = player_data.get('name', f"Player {len(players)+1}")
        
        if player_type == 'RANDOM':
            players.append(Player(name, RANDOM_FUNCS))
        elif player_type == 'TRUTH':
            players.append(Player(name, TRUTH_FUNCS))
        elif player_type == 'GREEDY':
            players.append(Player(name, GREEDY_FUNCS))
        elif player_type == 'INCOME':
            players.append(Player(name, INCOME_FUNCS))
    
    # Create game
    game = Game(players, debug=True)
    
    # Save initial state
    previous_state = copy_game_state(game.game_state)
    
    return jsonify({
        'status': 'success',
        'message': 'New game created',
        'game_state': serialize_game_state(game.game_state)
    })

@app.route('/api/next_turn', methods=['GET'])
def next_turn():
    global game, game_log, previous_state
    
    if game is None:
        return jsonify({'status': 'error', 'message': 'No game in progress'})
    
    if len(game.game_state['players']) <= 1:
        return jsonify({
            'status': 'game_over',
            'winner': game.game_state['players'][0].name if game.game_state['players'] else None,
            'game_state': serialize_game_state(game.game_state)
        })
    
    # Store the pre-turn state before taking the turn
    pre_turn_state = copy_game_state(game.game_state)
    
    # Take next turn
    action = game.simulate_turn()
    
    # Get the latest turn history
    latest_state, latest_turn = game.history[-1]
    action, block_1, block_2 = latest_turn
    
    # Create a more detailed log entry with better action descriptions
    log_entry = {
        'turn': len(game_log) + 1,
        'blocks': [],
        'outcomes': [],
        'state_changes': []
    }
    
    # Format the action message based on the action type
    action_player = action[0]
    target_player = action[1]
    action_type = action[2]
    
    if action_player == target_player:
        # Action targeting self
        if action_type == 'Income':
            log_entry['action'] = f"{action_player} took Income (+1 coin)"
        elif action_type == 'Foreign Aid':
            log_entry['action'] = f"{action_player} took Foreign Aid (+2 coins)"
        elif action_type == 'Tax':
            log_entry['action'] = f"{action_player} collected Tax (+3 coins)"
        elif action_type == 'Exchange':
            log_entry['action'] = f"{action_player} exchanged cards with the deck"
        else:
            log_entry['action'] = f"{action_player} used {action_type}"
    else:
        # Action targeting another player
        if action_type == 'Steal':
            coins_stolen = min(pre_turn_state['player_coins'][target_player], 2)
            log_entry['action'] = f"{action_player} stole {coins_stolen} coins from {target_player}"
        elif action_type == 'Coup':
            log_entry['action'] = f"{action_player} launched a Coup against {target_player}"
        elif action_type == 'Assassinate':
            log_entry['action'] = f"{action_player} attempted to assassinate {target_player}"
        else:
            log_entry['action'] = f"{action_player} used {action_type} on {target_player}"
    
    # Process blocks and outcomes
    # Case 1: Bluff call on a block
    if block_2[1]:  # There was a second block (calling the first block's bluff)
        blocker = block_1[0]
        challenger = block_2[0]
        blocker_cards = pre_turn_state['player_cards'][blocker]
        
        # Check if blocker was lying
        if action_type in ACTION_BLOCKER and did_block_1_lie(action_type, blocker_cards):
            # Blocker was lying
            lost_card = find_lost_card(pre_turn_state, game.game_state, blocker)
            log_entry['blocks'].append(f"{blocker} blocked by counter action.")
            log_entry['blocks'].append(f"{challenger} called bluff.")
            log_entry['outcomes'].append(f"Challenge succeeded: {blocker} was lying and lost {lost_card}.")
        else:
            # Blocker was telling truth
            lost_card = find_lost_card(pre_turn_state, game.game_state, challenger)
            log_entry['blocks'].append(f"{blocker} blocked by counter action.")
            log_entry['blocks'].append(f"{challenger} called bluff.")
            log_entry['outcomes'].append(f"Challenge failed: {blocker} had the required card. {challenger} lost {lost_card}.")
    
    # Case 2: Direct bluff call on the action
    elif block_1[1] and block_1[2]:  # There was a block calling the action a lie
        challenger = block_1[0]
        actor = action_player
        actor_cards = pre_turn_state['player_cards'][actor]
        
        # Check if actor was lying
        if action_type in ACTION_SENDER and did_action_lie(action_type, actor_cards):
            # Actor was lying
            lost_card = find_lost_card(pre_turn_state, game.game_state, actor)
            log_entry['blocks'].append(f"{challenger} called bluff on {actor}'s {action_type}.")
            log_entry['outcomes'].append(f"Challenge succeeded: {actor} was lying and lost {lost_card}.")
        else:
            # Actor was telling truth
            lost_card = find_lost_card(pre_turn_state, game.game_state, challenger)
            log_entry['blocks'].append(f"{challenger} called bluff on {actor}'s {action_type}.")
            log_entry['outcomes'].append(f"Challenge failed: {actor} had the {ACTION_SENDER.get(action_type, 'required')} card. {challenger} lost {lost_card}.")
    
    # Case 3: Role counter (no bluff call)
    elif block_1[1] and not block_1[2]:
        blocker = block_1[0]
        required_role = ", ".join(ACTION_BLOCKER.get(action_type, []))
        log_entry['blocks'].append(f"{blocker} blocked with {required_role}.")
    
    # Track significant state changes
    track_state_changes(pre_turn_state, game.game_state, log_entry['state_changes'])
    
    # Save this turn's state for comparing with the next turn
    previous_state = copy_game_state(game.game_state)
    
    game_log.append(log_entry)
    
    return jsonify({
        'status': 'success', 
        'game_state': serialize_game_state(game.game_state),
        'log': game_log
    })

@app.route('/api/game_state', methods=['GET'])
def get_game_state():
    global game, game_log
    
    if game is None:
        return jsonify({'status': 'error', 'message': 'No game in progress'})
    
    return jsonify({
        'status': 'success',
        'game_state': serialize_game_state(game.game_state),
        'log': game_log,
        'game_over': len(game.game_state['players']) <= 1,
        'winner': game.game_state['players'][0].name if len(game.game_state['players']) == 1 else None
    })

def serialize_game_state(game_state):
    return {
        'players': [p.name for p in game_state['players']],
        'player_cards': {k: v for k, v in game_state['player_cards'].items()},
        'player_deaths': {k: v for k, v in game_state['player_deaths'].items()},
        'player_coins': {k: v for k, v in game_state['player_coins'].items()},
        'current_player': game_state['current_player'].name,
        'deck_size': len(game_state['deck'])
    }

def copy_game_state(game_state):
    """Create a copy of relevant parts of the game state for comparison"""
    return {
        'player_cards': {k: v.copy() for k, v in game_state['player_cards'].items()},
        'player_deaths': {k: v.copy() for k, v in game_state['player_deaths'].items()},
        'player_coins': {k: v for k, v in game_state['player_coins'].items()}
    }

def find_lost_card(old_state, new_state, player_name):
    """Find which card a player lost by comparing old and new states"""
    old_cards = set(old_state['player_cards'].get(player_name, []))
    new_cards = set(new_state['player_cards'].get(player_name, []))
    lost_cards = old_cards - new_cards
    
    if lost_cards:
        return next(iter(lost_cards))
    
    # If we can't determine from cards, check player_deaths
    old_deaths = set(old_state['player_deaths'].get(player_name, []))
    new_deaths = set(new_state['player_deaths'].get(player_name, []))
    new_deaths_list = list(new_deaths - old_deaths)
    
    if new_deaths_list:
        return new_deaths_list[0]
    
    return "unknown card"

def track_state_changes(old_state, new_state, changes_list):
    """Track significant state changes between turns"""
    # Track coin changes
    for player, new_coins in new_state['player_coins'].items():
        old_coins = old_state['player_coins'].get(player, 0)
        diff = new_coins - old_coins
        if diff != 0:
            direction = "gained" if diff > 0 else "lost"
            changes_list.append(f"{player} {direction} {abs(diff)} coins")
    
    # Track card losses
    for player, new_deaths in new_state['player_deaths'].items():
        old_deaths = old_state['player_deaths'].get(player, [])
        if len(new_deaths) > len(old_deaths):
            new_lost_cards = [card for card in new_deaths if card not in old_deaths]
            for card in new_lost_cards:
                changes_list.append(f"{player} lost {card}")

if __name__ == '__main__':
    app.run(debug=True) 