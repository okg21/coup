from flask import Flask, render_template, jsonify, request, Response
from player import *
from game import *
from utils import did_action_lie, did_block_1_lie
import json
import time
import threading
from collections import Counter
from io import StringIO
import csv

app = Flask(__name__, static_url_path='', static_folder='static', template_folder='templates')

# Game state
game = None
game_log = []
previous_state = None

# Simulation state
simulation_active = False
simulation_progress = 0
simulation_results = {}
simulation_logs = []
simulation_winner_counts = Counter()
simulation_stop_flag = False
simulation_thread = None

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

@app.route('/api/start_simulation', methods=['POST'])
def start_simulation():
    global simulation_active, simulation_progress, simulation_results, simulation_logs, simulation_winner_counts, simulation_stop_flag, simulation_thread
    
    if simulation_active:
        return jsonify({'status': 'error', 'message': 'Simulation already in progress'})
    
    data = request.json
    num_games = data.get('num_games', 100)
    players_config = data.get('players', [])
    
    # Reset simulation state
    simulation_active = True
    simulation_progress = 0
    simulation_results = {}
    simulation_logs = []
    simulation_winner_counts = Counter()
    simulation_stop_flag = False
    
    # Start simulation in a background thread
    simulation_thread = threading.Thread(target=run_simulation, args=(num_games, players_config))
    simulation_thread.daemon = True
    simulation_thread.start()
    
    return jsonify({
        'status': 'success',
        'message': 'Simulation started'
    })

@app.route('/api/simulation_progress', methods=['GET'])
def get_simulation_progress():
    global simulation_active, simulation_progress, simulation_results, simulation_winner_counts
    
    if not simulation_active and simulation_progress == 0:
        return jsonify({
            'status': 'not_started',
            'progress': 0
        })
    
    if not simulation_active and simulation_progress == 100:
        # Convert counter to dict for JSON serialization
        winners_dict = dict(simulation_winner_counts)
        total_games = sum(winners_dict.values())
        
        # Calculate percentages
        win_percentages = {
            player: (count / total_games) * 100 
            for player, count in winners_dict.items()
        }
        
        return jsonify({
            'status': 'completed',
            'progress': 100,
            'results': simulation_results,
            'winners': winners_dict,
            'win_percentages': win_percentages,
            'total_games': total_games
        })
    
    return jsonify({
        'status': 'in_progress',
        'progress': simulation_progress
    })

@app.route('/api/stop_simulation', methods=['POST'])
def stop_simulation():
    global simulation_stop_flag
    
    simulation_stop_flag = True
    
    return jsonify({
        'status': 'success',
        'message': 'Simulation stop requested'
    })

@app.route('/api/download_simulation_logs', methods=['GET'])
def download_simulation_logs():
    global simulation_logs
    
    if not simulation_logs:
        return jsonify({'status': 'error', 'message': 'No simulation logs available'})
    
    # Create CSV in memory
    output = StringIO()
    writer = csv.writer(output)
    
    # Write headers
    writer.writerow(['Game', 'Turn', 'Action', 'Blocks', 'Outcomes', 'State Changes', 'Winner'])
    
    # Write data
    for game_idx, game_data in enumerate(simulation_logs):
        winner = game_data['winner']
        for turn_idx, log_entry in enumerate(game_data['log']):
            blocks = '; '.join(log_entry.get('blocks', []))
            outcomes = '; '.join(log_entry.get('outcomes', []))
            state_changes = '; '.join(log_entry.get('state_changes', []))
            
            writer.writerow([
                game_idx + 1,
                turn_idx + 1,
                log_entry.get('action', ''),
                blocks,
                outcomes,
                state_changes,
                winner if turn_idx == len(game_data['log']) - 1 else ''
            ])
    
    # Create response
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={
            'Content-Disposition': 'attachment; filename=coup_simulation_logs.csv'
        }
    )

def run_simulation(num_games, players_config):
    global simulation_active, simulation_progress, simulation_results, simulation_logs, simulation_winner_counts, simulation_stop_flag
    
    try:
        for game_idx in range(num_games):
            if simulation_stop_flag:
                break
                
            # Create players
            players = []
            for player_data in players_config:
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
            
            # Create and run game
            sim_game = Game(players, debug=False)
            game_logs = []
            
            prev_state = copy_game_state(sim_game.game_state)
            
            # Run game until completion
            while len(sim_game.game_state['players']) > 1:
                if simulation_stop_flag:
                    break
                    
                # Take a turn
                pre_turn_state = copy_game_state(sim_game.game_state)
                sim_game.simulate_turn()
                
                # Get the latest turn history
                latest_state, latest_turn = sim_game.history[-1]
                action, block_1, block_2 = latest_turn
                
                # Create log entry for this turn
                log_entry = create_log_entry(action, block_1, block_2, pre_turn_state, sim_game.game_state, len(game_logs) + 1)
                game_logs.append(log_entry)
                
                # Update previous state
                prev_state = copy_game_state(sim_game.game_state)
            
            # Record winner
            winner = sim_game.game_state['players'][0].name if sim_game.game_state['players'] else None
            if winner:
                simulation_winner_counts[winner] += 1
            
            # Store log for this game
            if not simulation_stop_flag:
                simulation_logs.append({
                    'game': game_idx + 1,
                    'winner': winner,
                    'log': game_logs
                })
            
            # Update progress
            simulation_progress = int((game_idx + 1) / num_games * 100)
            
            # Small delay to prevent UI blocking
            time.sleep(0.01)
        
        # Finalize simulation if not stopped
        if not simulation_stop_flag:
            simulation_progress = 100
        
        # Calculate results
        simulation_results = {
            'total_games': len(simulation_logs),
            'winners': dict(simulation_winner_counts),
        }
    
    except Exception as e:
        print(f"Simulation error: {e}")
    finally:
        simulation_active = False

def create_log_entry(action, block_1, block_2, pre_turn_state, current_state, turn_number):
    log_entry = {
        'turn': turn_number,
        'blocks': [],
        'outcomes': [],
        'state_changes': []
    }
    
    # Format the action
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
    # Similar to next_turn logic for blocks and outcomes
    # Case 1: Bluff call on a block
    if block_2[1]:
        blocker = block_1[0]
        challenger = block_2[0]
        blocker_cards = pre_turn_state['player_cards'][blocker]
        
        # Check if blocker was lying
        if action_type in ACTION_BLOCKER and did_block_1_lie(action_type, blocker_cards):
            lost_card = find_lost_card(pre_turn_state, current_state, blocker)
            log_entry['blocks'].append(f"{blocker} blocked by counter action.")
            log_entry['blocks'].append(f"{challenger} called bluff.")
            log_entry['outcomes'].append(f"Challenge succeeded: {blocker} was lying and lost {lost_card}.")
        else:
            lost_card = find_lost_card(pre_turn_state, current_state, challenger)
            log_entry['blocks'].append(f"{blocker} blocked by counter action.")
            log_entry['blocks'].append(f"{challenger} called bluff.")
            log_entry['outcomes'].append(f"Challenge failed: {blocker} had the required card. {challenger} lost {lost_card}.")
    
    # Case 2: Direct bluff call on the action
    elif block_1[1] and block_1[2]:
        challenger = block_1[0]
        actor = action_player
        actor_cards = pre_turn_state['player_cards'][actor]
        
        if action_type in ACTION_SENDER and did_action_lie(action_type, actor_cards):
            lost_card = find_lost_card(pre_turn_state, current_state, actor)
            log_entry['blocks'].append(f"{challenger} called bluff on {actor}'s {action_type}.")
            log_entry['outcomes'].append(f"Challenge succeeded: {actor} was lying and lost {lost_card}.")
        else:
            lost_card = find_lost_card(pre_turn_state, current_state, challenger)
            log_entry['blocks'].append(f"{challenger} called bluff on {actor}'s {action_type}.")
            log_entry['outcomes'].append(f"Challenge failed: {actor} had the {ACTION_SENDER.get(action_type, 'required')} card. {challenger} lost {lost_card}.")
    
    # Case 3: Role counter (no bluff call)
    elif block_1[1] and not block_1[2]:
        blocker = block_1[0]
        required_role = ", ".join(ACTION_BLOCKER.get(action_type, []))
        log_entry['blocks'].append(f"{blocker} blocked with {required_role}.")
    
    # Track state changes
    track_state_changes(pre_turn_state, current_state, log_entry['state_changes'])
    
    return log_entry

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