document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const newGameBtn = document.getElementById('new-game-btn');
    const simulateGamesBtn = document.getElementById('simulate-games-btn');
    const nextTurnBtn = document.getElementById('next-turn-btn');
    const playersContainer = document.getElementById('players-container');
    const gameLogContainer = document.getElementById('game-log');
    const deckCount = document.getElementById('deck-count');
    const currentPlayerName = document.getElementById('current-player-name');
    const setupModal = document.getElementById('setup-modal');
    const gameOverModal = document.getElementById('game-over-modal');
    const simulationModal = document.getElementById('simulation-modal');
    const winnerName = document.getElementById('winner-name');
    const addPlayerBtn = document.getElementById('add-player-btn');
    const removePlayerBtn = document.getElementById('remove-player-btn');
    const playerForms = document.getElementById('player-forms');
    const startGameBtn = document.getElementById('start-game-btn');
    const newGameAfterWinBtn = document.getElementById('new-game-after-win-btn');
    const toggleGuideBtn = document.getElementById('toggle-guide-btn');
    const actionGuide = document.getElementById('action-guide');
    const modalCloseBtn = document.querySelector('.modal-close');
    const gameOverCloseBtn = document.querySelector('.game-over-close');
    const simulationCloseBtn = document.querySelector('.simulation-close');
    const simPlayerForms = document.getElementById('simulation-players');
    const simAddPlayerBtn = document.getElementById('sim-add-player-btn');
    const simRemovePlayerBtn = document.getElementById('sim-remove-player-btn');
    const startSimulationBtn = document.getElementById('start-simulation-btn');
    const stopSimulationBtn = document.getElementById('stop-simulation-btn');
    const downloadLogsBtn = document.getElementById('download-logs-btn');
    const simulationProgressBar = document.getElementById('simulation-progress-bar');
    const simulationProgressText = document.getElementById('simulation-progress-text');
    const simulationStats = document.getElementById('simulation-stats');
    const simulationProgressContainer = document.querySelector('.simulation-progress-container');
    const simulationForm = document.querySelector('.simulation-form');
    const simulationCount = document.getElementById('simulation-count');

    // Game state
    let gameState = null;
    let gameTurnLog = [];
    let playerColors = {};
    let playerOrder = []; // Store the initial order of players
    const playerTypeLabels = {
        'RANDOM': 'Random Player',
        'TRUTH': 'Truth Teller',
        'GREEDY': 'Greedy Player',
        'INCOME': 'Income Focused'
    };

    // Simulation state
    let simulationActive = false;
    let simulationProgressInterval = null;

    // Initialize game
    showSetupModal();

    // Event Listeners
    newGameBtn.addEventListener('click', showSetupModal);
    simulateGamesBtn.addEventListener('click', showSimulationModal);
    nextTurnBtn.addEventListener('click', handleNextTurn);
    addPlayerBtn.addEventListener('click', addPlayerForm);
    removePlayerBtn.addEventListener('click', removePlayerForm);
    startGameBtn.addEventListener('click', startNewGame);
    newGameAfterWinBtn.addEventListener('click', showSetupModal);
    toggleGuideBtn.addEventListener('click', toggleActionGuide);
    modalCloseBtn.addEventListener('click', closeSetupModal);
    gameOverCloseBtn.addEventListener('click', closeGameOverModal);
    simulationCloseBtn.addEventListener('click', closeSimulationModal);
    simAddPlayerBtn.addEventListener('click', addSimPlayerForm);
    simRemovePlayerBtn.addEventListener('click', removeSimPlayerForm);
    startSimulationBtn.addEventListener('click', startSimulation);
    stopSimulationBtn.addEventListener('click', stopSimulation);
    downloadLogsBtn.addEventListener('click', downloadSimulationLogs);

    function toggleActionGuide() {
        actionGuide.classList.toggle('show');
    }

    function showSetupModal() {
        gameOverModal.classList.remove('show');
        simulationModal.classList.remove('show');
        setupModal.classList.add('show');
        // Reset player forms to default state with 2 players
        playerForms.innerHTML = '';
        addDefaultPlayerForms();
    }

    function closeSetupModal() {
        setupModal.classList.remove('show');
    }

    function closeGameOverModal() {
        gameOverModal.classList.remove('show');
    }

    function showSimulationModal() {
        setupModal.classList.remove('show');
        gameOverModal.classList.remove('show');
        simulationModal.classList.add('show');
        
        // Reset simulation form
        resetSimulationForm();
    }

    function closeSimulationModal() {
        simulationModal.classList.remove('show');
        
        // If a simulation is active, stop it
        if (simulationActive) {
            stopSimulation();
        }
    }

    function resetSimulationForm() {
        // Reset simulation state
        simulationCount.value = 100;
        simPlayerForms.innerHTML = '';
        simulationStats.innerHTML = '';
        simulationProgressBar.style.width = '0%';
        simulationProgressText.textContent = '0%';
        
        // Hide progress container and buttons
        simulationProgressContainer.style.display = 'none';
        simulationForm.style.display = 'block';
        startSimulationBtn.style.display = 'block';
        stopSimulationBtn.style.display = 'none';
        downloadLogsBtn.style.display = 'none';
        
        // Add default player forms
        addDefaultSimPlayerForms();
    }

    function addDefaultPlayerForms() {
        for (let i = 0; i < 2; i++) {
            const playerIndex = i + 1;
            const playerForm = document.createElement('div');
            playerForm.className = 'player-form';
            const playerId = `player${playerIndex}`;
            
            playerForm.innerHTML = `
                <label for="${playerId}-name">Player ${playerIndex}:</label>
                <input type="text" id="${playerId}-name" class="player-name" value="Player ${playerIndex}">
                <select id="${playerId}-type" class="player-type" aria-label="Select player ${playerIndex} type">
                    <option value="RANDOM">Random</option>
                    <option value="TRUTH">Truth Teller</option>
                    <option value="GREEDY">Greedy</option>
                    <option value="INCOME">Income</option>
                </select>
            `;
            playerForms.appendChild(playerForm);
        }
    }

    function addDefaultSimPlayerForms() {
        for (let i = 0; i < 2; i++) {
            const playerIndex = i + 1;
            const playerForm = document.createElement('div');
            playerForm.className = 'player-form';
            const playerId = `sim-player${playerIndex}`;
            
            playerForm.innerHTML = `
                <label for="${playerId}-name">Player ${playerIndex}:</label>
                <input type="text" id="${playerId}-name" class="player-name" value="Player ${playerIndex}">
                <select id="${playerId}-type" class="player-type" aria-label="Select player ${playerIndex} type">
                    <option value="RANDOM">Random</option>
                    <option value="TRUTH">Truth Teller</option>
                    <option value="GREEDY">Greedy</option>
                    <option value="INCOME">Income</option>
                </select>
            `;
            simPlayerForms.appendChild(playerForm);
        }
    }

    function addPlayerForm() {
        const playerCount = playerForms.children.length;
        if (playerCount >= 6) {
            alert('Maximum 6 players allowed');
            return;
        }

        const playerIndex = playerCount + 1;
        const playerId = `player${playerIndex}`;
        const playerForm = document.createElement('div');
        playerForm.className = 'player-form';
        
        playerForm.innerHTML = `
            <label for="${playerId}-name">Player ${playerIndex}:</label>
            <input type="text" id="${playerId}-name" class="player-name" value="Player ${playerIndex}">
            <select id="${playerId}-type" class="player-type" aria-label="Select player ${playerIndex} type">
                <option value="RANDOM">Random</option>
                <option value="TRUTH">Truth Teller</option>
                <option value="GREEDY">Greedy</option>
                <option value="INCOME">Income</option>
            </select>
        `;
        playerForms.appendChild(playerForm);
    }

    function addSimPlayerForm() {
        const playerCount = simPlayerForms.children.length;
        if (playerCount >= 6) {
            alert('Maximum 6 players allowed');
            return;
        }

        const playerIndex = playerCount + 1;
        const playerId = `sim-player${playerIndex}`;
        const playerForm = document.createElement('div');
        playerForm.className = 'player-form';
        
        playerForm.innerHTML = `
            <label for="${playerId}-name">Player ${playerIndex}:</label>
            <input type="text" id="${playerId}-name" class="player-name" value="Player ${playerIndex}">
            <select id="${playerId}-type" class="player-type" aria-label="Select player ${playerIndex} type">
                <option value="RANDOM">Random</option>
                <option value="TRUTH">Truth Teller</option>
                <option value="GREEDY">Greedy</option>
                <option value="INCOME">Income</option>
            </select>
        `;
        simPlayerForms.appendChild(playerForm);
    }

    function removePlayerForm() {
        if (playerForms.children.length <= 2) {
            alert('Minimum 2 players required');
            return;
        }
        playerForms.removeChild(playerForms.lastChild);
    }

    function removeSimPlayerForm() {
        if (simPlayerForms.children.length <= 2) {
            alert('Minimum 2 players required');
            return;
        }
        simPlayerForms.removeChild(simPlayerForms.lastChild);
    }

    function startNewGame() {
        const players = [];
        const forms = playerForms.children;
        
        // Collect player data from forms
        for (let i = 0; i < forms.length; i++) {
            const nameInput = forms[i].querySelector('.player-name');
            const typeSelect = forms[i].querySelector('.player-type');
            
            players.push({
                name: nameInput.value.trim() || `Player ${i+1}`,
                type: typeSelect.value
            });
        }

        // Use fixed gold and scarlet for first two players
        playerColors = {};
        const cssRoot = document.documentElement;
        const goldColor = getComputedStyle(cssRoot).getPropertyValue('--gold').trim();
        const scarletColor = getComputedStyle(cssRoot).getPropertyValue('--scarlet').trim();
        
        // Assign colors to players - fixed colors for P1 and P2, random for others
        players.forEach((player, index) => {
            if (index === 0) {
                playerColors[player.name] = goldColor;
            } else if (index === 1) {
                playerColors[player.name] = scarletColor;
            } else {
                playerColors[player.name] = getRandomColor();
            }
        });

        // Start the game via API
        fetch('/api/new_game', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ players })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                gameState = data.game_state;
                gameTurnLog = [];
                
                // Save the initial player order for consistent display
                playerOrder = Object.keys(gameState.player_cards);
                
                setupModal.classList.remove('show');
                updateUI();
                nextTurnBtn.disabled = false;
            } else {
                alert('Failed to start game: ' + data.message);
            }
        })
        .catch(error => {
            console.error('Error starting game:', error);
            alert('Error starting game. Check console for details.');
        });
    }

    function startSimulation() {
        const numGames = parseInt(simulationCount.value);
        if (isNaN(numGames) || numGames < 1) {
            alert('Please enter a valid number of games');
            return;
        }
        
        const players = [];
        const forms = simPlayerForms.children;
        
        // Collect player data from forms
        for (let i = 0; i < forms.length; i++) {
            const nameInput = forms[i].querySelector('.player-name');
            const typeSelect = forms[i].querySelector('.player-type');
            
            players.push({
                name: nameInput.value.trim() || `Player ${i+1}`,
                type: typeSelect.value
            });
        }
        
        // Hide form and show progress
        simulationForm.style.display = 'none';
        simulationProgressContainer.style.display = 'block';
        startSimulationBtn.style.display = 'none';
        stopSimulationBtn.style.display = 'block';
        downloadLogsBtn.style.display = 'none';
        
        // Reset progress UI
        simulationProgressBar.style.width = '0%';
        simulationProgressText.textContent = '0%';
        simulationStats.innerHTML = '';
        
        // Start simulation
        fetch('/api/start_simulation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                num_games: numGames,
                players: players
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                simulationActive = true;
                
                // Start polling for progress
                simulationProgressInterval = setInterval(checkSimulationProgress, 500);
            } else {
                alert('Failed to start simulation: ' + data.message);
                resetSimulationForm();
            }
        })
        .catch(error => {
            console.error('Error starting simulation:', error);
            alert('Error starting simulation. Check console for details.');
            resetSimulationForm();
        });
    }

    function stopSimulation() {
        fetch('/api/stop_simulation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                simulationActive = false;
            }
        })
        .catch(error => {
            console.error('Error stopping simulation:', error);
        });
    }

    function checkSimulationProgress() {
        fetch('/api/simulation_progress')
            .then(response => response.json())
            .then(data => {
                const progress = data.progress || 0;
                
                // Update UI
                simulationProgressBar.style.width = `${progress}%`;
                simulationProgressText.textContent = `${progress}%`;
                
                if (data.status === 'completed') {
                    clearInterval(simulationProgressInterval);
                    simulationActive = false;
                    
                    // Update stats
                    updateSimulationStats(data);
                    
                    // Show download button
                    stopSimulationBtn.style.display = 'none';
                    downloadLogsBtn.style.display = 'block';
                }
                else if (data.status === 'not_started') {
                    clearInterval(simulationProgressInterval);
                    simulationActive = false;
                    resetSimulationForm();
                }
            })
            .catch(error => {
                console.error('Error checking simulation progress:', error);
            });
    }

    function updateSimulationStats(data) {
        const totalGames = data.total_games;
        const winners = data.winners;
        const winPercentages = data.win_percentages;
        
        // Create stats table
        let statsHtml = `
            <h3>Simulation Results (${totalGames} games)</h3>
            <table class="stats-table">
                <thead>
                    <tr>
                        <th>Player</th>
                        <th>Wins</th>
                        <th>Win %</th>
                        <th></th>
                    </tr>
                </thead>
                <tbody>
        `;
        
        // Sort players by win count (descending)
        const sortedPlayers = Object.keys(winners).sort((a, b) => winners[b] - winners[a]);
        
        // Add rows for each player
        sortedPlayers.forEach(player => {
            const wins = winners[player];
            const percentage = winPercentages[player].toFixed(1);
            const barWidth = percentage + '%';
            
            statsHtml += `
                <tr>
                    <td>${player}</td>
                    <td>${wins}</td>
                    <td>${percentage}%</td>
                    <td>
                        <div class="win-bar" style="width: ${barWidth}"></div>
                    </td>
                </tr>
            `;
        });
        
        statsHtml += `
                </tbody>
            </table>
        `;
        
        simulationStats.innerHTML = statsHtml;
    }

    function downloadSimulationLogs() {
        window.location.href = '/api/download_simulation_logs';
    }

    function handleNextTurn() {
        nextTurnBtn.disabled = true;
        
        fetch('/api/next_turn')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    gameState = data.game_state;
                    gameTurnLog = data.log;
                    updateUI();
                    nextTurnBtn.disabled = false;
                } else if (data.status === 'game_over') {
                    gameState = data.game_state;
                    updateUI();
                    showGameOverModal(data.winner);
                } else {
                    alert('Error: ' + data.message);
                    nextTurnBtn.disabled = false;
                }
            })
            .catch(error => {
                console.error('Error during turn:', error);
                alert('Error during turn. Check console for details.');
                nextTurnBtn.disabled = false;
            });
    }

    function updateUI() {
        if (!gameState) return;
        
        // Update deck info
        deckCount.textContent = gameState.deck_size;
        
        // Update current player
        currentPlayerName.textContent = gameState.current_player;
        
        // Update players
        updatePlayersUI();
        
        // Update game log
        updateGameLog();
    }

    function updatePlayersUI() {
        playersContainer.innerHTML = '';
        
        // Use the saved player order instead of the current game state order
        playerOrder.forEach((playerName, index) => {
            const playerCard = document.createElement('div');
            playerCard.className = 'player-card';
            
            // Add player index class to apply player-specific styling like colored names
            playerCard.classList.add(`player--p${index + 1}`);
            
            // Check if player is active (current player)
            if (playerName === gameState.current_player) {
                playerCard.classList.add('active');
            }
            
            // Check if player is dead (no cards left)
            const isDead = gameState.player_cards[playerName].length === 0;
            if (isDead) {
                playerCard.style.opacity = '0.7';
            }
            
            // Find player type from initial player setup
            const playerType = Object.keys(playerTypeLabels).find(type => 
                playerName.startsWith(type) || 
                Array.from(document.querySelectorAll('.player-form')).some(form => {
                    const nameInput = form.querySelector('.player-name');
                    const typeSelect = form.querySelector('.player-type');
                    return nameInput.value === playerName && typeSelect.value === type;
                })
            ) || 'RANDOM';
            
            playerCard.innerHTML = `
                <h3>${playerName}</h3>
                <div class="player-strategy">${playerTypeLabels[playerType] || 'AI Player'}</div>
                <div class="player-coins">
                    <div class="coin-icon" role="img" aria-label="Coins"></div>
                    <span>${gameState.player_coins[playerName] || 0} coins</span>
                </div>
                <div class="player-cards">
                    ${renderPlayerCards(playerName)}
                </div>
                <div class="player-deaths">
                    ${renderDeadCards(playerName)}
                </div>
            `;
            
            playersContainer.appendChild(playerCard);
        });
    }

    function renderPlayerCards(playerName) {
        const cards = gameState.player_cards[playerName] || [];
        if (cards.length === 0) return '<div class="no-cards">Eliminated</div>';
        
        return cards.map(card => {
            return `<div class="card ${card}" role="img" aria-label="${card} card"><span>${card}</span></div>`;
        }).join('');
    }

    function renderDeadCards(playerName) {
        const deadCards = gameState.player_deaths[playerName] || [];
        if (deadCards.length === 0) return '';
        
        return `
            <div class="dead-cards">
                Lost: ${deadCards.map(card => `<span class="dead-card">${card}</span>`).join(', ')}
            </div>
        `;
    }

    function updateGameLog() {
        const logContainer = gameLogContainer;
        logContainer.innerHTML = '';
        
        if (gameTurnLog.length === 0) {
            logContainer.innerHTML = '<div class="game-log-entry">Game started. Click "Next Turn" to begin.</div>';
            return;
        }
        
        gameTurnLog.forEach((entry, index) => {
            const logEntry = document.createElement('div');
            logEntry.className = 'game-log-entry';
            
            // Create the log entry content with the action
            let logContent = `<div class="action-main">${entry.action}</div>`;
            
            // Add blocks if any
            if (entry.blocks && entry.blocks.length > 0) {
                entry.blocks.forEach(block => {
                    logContent += `<div class="block-action">→ ${block}</div>`;
                });
            }
            
            // Add challenge outcomes if any
            if (entry.outcomes && entry.outcomes.length > 0) {
                entry.outcomes.forEach(outcome => {
                    const outcomeClass = outcome.includes('succeeded') ? 'success' : 
                                       outcome.includes('failed') ? 'failure' : '';
                    logContent += `<div class="challenge-outcome ${outcomeClass}">⚡ ${outcome}</div>`;
                });
            }
            
            // Add state changes if any
            if (entry.state_changes && entry.state_changes.length > 0) {
                logContent += `<div class="state-changes">`;
                entry.state_changes.forEach(change => {
                    logContent += `<div class="state-change">• ${change}</div>`;
                });
                logContent += `</div>`;
            }
            
            // Add turn number as a small label
            logContent = `<div class="turn-number">Turn ${entry.turn}</div>` + logContent;
            
            logEntry.innerHTML = logContent;
            logContainer.appendChild(logEntry);
        });
        
        // Scroll to bottom
        logContainer.scrollTop = logContainer.scrollHeight;
    }

    function showGameOverModal(winner) {
        nextTurnBtn.disabled = true;
        winnerName.textContent = winner;
        gameOverModal.classList.add('show');
    }

    function getRandomColor() {
        // Generate a random muted color that fits the theme
        const hue = Math.floor(Math.random() * 360);
        const saturation = Math.floor(Math.random() * 30) + 20; // 20-50%
        const lightness = Math.floor(Math.random() * 15) + 40; // 40-55%
        return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
    }
}); 