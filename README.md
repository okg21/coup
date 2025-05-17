![coup](https://github.com/riensou/coup/assets/90002238/8c00b7d3-032c-40c7-9ae4-e643c5575e9a)


Python implentation of the popular bluffing card game, _Coup_.

For a description of the rules, click [here](https://www.ultraboardgames.com/coup/game-rules.php).


**Player Descriptions:**
1. Random
- all logic functions make completely random choices
- uses random_dispose, random_keep

2. Truth Teller
- only chooses moves that never bluff based on its own cards
- uses random_dispose, random_keep

3. User
- chooses moves based on user input to the terminal

4. Income
- always takes income or assinates/coups whenever possible
- uses random_dispose, random_keep

5. Neural Network
- utilizes Q-learning with a neural network to learn the best decision_fn
- uses income_block, random_dispose, random_keep


**Running the program:**
1. Create player objects using the various FUNCS from player.py
2. Put all players into the players list and create the game object.
3. Run python3 game.py

**Running the UI Overlay:**
1. Install the required dependencies: `pip install -r requirements.txt`
2. Start the Flask server: `python server.py`
3. Open your browser and navigate to `http://localhost:5000`
4. Use the UI to set up players and watch the game in a nice viewing experience


**Model Files Format:**
"{number of players}-{number of episodes trained on}-{types of opponents}"


**Missing Features / Bugs:**
* Blocking the 'Steal' action is ambiguous, and doesn't force the player to specifically claim 'Captain' or 'Ambassador'
* Winning a challenge does not cause the player to reshuffle their card back into the deck
* ~~There is no implementation for a GUI yet~~ (Added a web-based UI overlay!)
* RL agents are only trained to be the first player, need to fix cases when they aren't first
