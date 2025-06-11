/*  snake-game.js  */
/* Snake Game Component with JSON Support – robust fetch & parse */
(function () {
    /**
     * CONFIGURATION
     * ------------------------------------------------------------------
     * 1. Place your game-log JSON under /data or supply a full URL.
     * 2. <div id="snake-game-container" data-json="/static/games/game-42.json"></div>
     */
    const DEFAULT_JSON = './resources/game/o3-mini.json';

    /* Adjustable frames-per-second variable */
    let FPS = 3; // Change this value to adjust how many rounds per second are rendered

    /* ──────────────────────────────────────────────────────────────── */
    let gameJsonData = null;
    let jsonList = ['./resources/game/o3-mini.json',
        './resources/game/Claude-3.7-Sonnet.json',
        './resources/game/Gemini-2.5-Pro.json',
    ];
    let currentJsonIndex = 0;
    let currentAbsoluteURL = null; // Track the current absolute URL

    // Game state variables
    let maxRounds = 0,
        currentRound = 0,
        playing = false,
        W = 10,
        H = 10;

    let canvas = null;
    let ctx = null;

    /**
     * Helper function to update the file status message.
     * @param {string} message - The message to display.
     * @param {boolean} [isError=false] - True if the message is an error.
     */
    function updateFileStatus(message, isError = false) {
        const fileStatusEl = document.getElementById('fileStatus');
        if (fileStatusEl) {
            fileStatusEl.textContent = message;
            fileStatusEl.className = `font-mono text-lg ${isError ? 'text-red-600' : 'text-black font-bold'}`;
        }
    }

    /**
     * Set up the Match buttons handler (only once)
     */
    function setupMatchButtons() {
        const matchButtons = [
            { id: 'matchBtn1', path: './resources/game/o3-mini.json' },
            { id: 'matchBtn2', path: './resources/game/Claude-3.7-Sonnet.json' },
            { id: 'matchBtn3', path: './resources/game/Gemini-2.5-Pro.json' }
        ];

        matchButtons.forEach(({ id, path }) => {
            const btn = document.getElementById(id);
            if (!btn) return;

            // Remove any existing event listeners to prevent duplicates
            btn.onclick = null;
            
            btn.onclick = () => {
                console.log(`${id} clicked. Loading:`, path);
                
                if (path && typeof path === 'string') {
                    boot(path);
                } else {
                    console.error('Invalid path for match button:', path);
                    updateFileStatus('Error: Could not load the selected match.', true);
                }
            };
        });
    }

    /**
     * Update the Match buttons state
     */
    function updateMatchButtons() {
        const matchButtons = ['matchBtn1', 'matchBtn2', 'matchBtn3'];
        const matchPaths = [
            './resources/game/o3-mini.json',
            './resources/game/Claude-3.7-Sonnet.json', 
            './resources/game/Gemini-2.5-Pro.json'
        ];

        matchButtons.forEach((btnId, index) => {
            const btn = document.getElementById(btnId);
            if (!btn) return;

            const absolutePath = new URL(matchPaths[index], window.location.href).toString();
            const isCurrentMatch = currentAbsoluteURL === absolutePath;
            
            // Highlight the current match button
            if (isCurrentMatch) {
                btn.style.transform = 'scale(1.05)';
                btn.style.boxShadow = '0 0 10px rgba(59, 130, 246, 0.5)';
                btn.style.fontWeight = 'bold';
            } else {
                btn.style.transform = 'scale(1)';
                btn.style.boxShadow = 'none';
                btn.style.fontWeight = 'normal';
            }
        });
    }

    /* ------------------------------------------------------------------
     * INITIALIZATION ENTRY POINT
     * ------------------------------------------------------------------
     * The snake-game section is injected into the page asynchronously via
     *   loadSection('snake-demo', …). When this script runs at page load
     *   time, the element #snake-game-container may not yet be in the DOM.
     *   The following helper waits for the element to appear (using
     *   MutationObserver as a fallback) and then performs a one-time
     *   initialization.                                                    */

    function initializeSnakeGame() {
        const container = document.getElementById('snake-game-container');
        if (!container || container.getAttribute('data-snake-init') === 'done') return;

        // Mark as initialised to avoid duplicate boots
        container.setAttribute('data-snake-init', 'done');

        // Inject the game UI template
        container.innerHTML = markup();

        // Cache canvas & context
        canvas = document.getElementById('gameCanvas');
        ctx = canvas ? canvas.getContext('2d') : null;

        // Prepare aux helpers
        normalizeJsonList();
        setupMatchButtons();

        // Kick off the game logic
        const path = container.getAttribute('data-json') || DEFAULT_JSON;
        boot(path);
    }

    (function waitForContainer() {
        // If the document is still loading, delay until it is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', waitForContainer);
            return;
        }

        // If the container is already present → initialise immediately
        if (document.getElementById('snake-game-container')) {
            initializeSnakeGame();
            return;
        }

        // Otherwise observe the DOM for its insertion
        const observer = new MutationObserver((mutations, obs) => {
            if (document.getElementById('snake-game-container')) {
                obs.disconnect();
                initializeSnakeGame();
            }
        });
        observer.observe(document.body, { childList: true, subtree: true });
    })();

    /* ========================= TEMPLATE ============================ */
    function markup() {
        return `
          <div class="h-full flex flex-col">
            <div class="bg-gray-100 flex-1 overflow-hidden">
              <div class="bg-white shadow rounded-lg p-4 mb-4 text-center">
                <span id="fileStatus" class="font-mono text-lg text-black font-bold">Loading…</span>
              </div>

              <div class="grid grid-cols-1 lg:grid-cols-[1fr_min-content_1fr] gap-4 mb-6 h-full">
                <div class="thoughts-panel player1-border flex flex-col border rounded-lg overflow-hidden">
                  <div class="p-3 border-b border-gray-100 sticky top-0 z-10 text-center player1-bg">
                    <h2 id="player1Name" class="font-mono text-sm text-black font-bold">ViGaL (Ours)</h2>
                  </div>
                  <div class="flex-1 p-4 overflow-auto">
                    <div id="player1Thoughts" class="font-mono text-xs"></div>
                  </div>
                  <div class="p-3 border-t border-gray-100 flex justify-between">
                    <span id="player1Status" class="text-green-500 font-bold text-sm">WAITING</span>
                  </div>
                </div>

                <div class="flex items-center justify-center">
                  <canvas id="gameCanvas" width="400" height="400" class="game-canvas rounded-lg shadow"></canvas>
                </div>

                <div class="thoughts-panel player2-border flex flex-col border rounded-lg overflow-hidden">
                  <div class="p-3 border-b border-gray-100 sticky top-0 z-10 text-center player2-bg">
                    <h2 id="player2Name" class="font-mono text-sm text-black font-bold">Player 2</h2>
                  </div>
                  <div class="flex-1 p-4 overflow-auto">
                    <div id="player2Thoughts" class="font-mono text-xs"></div>
                  </div>
                  <div class="p-3 border-t border-gray-100 flex justify-between">
                    <span id="player2Status" class="text-red-500 font-bold text-sm">WAITING</span>
                  </div>
                </div>
              </div>

              <div class="bg-white shadow rounded-lg p-4 mb-4">
                <div class="flex justify-center items-center gap-4">
                    <button id="playBtn"  class="bg-blue-500 text-white px-4 py-2 rounded font-mono text-lg" disabled>⏸️ Pause</button>
                    <button id="prevBtn"  class="bg-gray-500 text-white px-4 py-2 rounded font-mono text-lg" disabled>⏪️ Prev</button>
                    <span id="roundInfo" class="font-mono text-lg text-gray-600">Round 0/0</span>
                    <button id="nextBtn"  class="bg-gray-500 text-white px-4 py-2 rounded font-mono text-lg" disabled>⏩ Next</button>
                    <button id="endBtn"   class="bg-gray-500 text-white px-4 py-2 rounded font-mono text-lg" disabled>⏭️ End</button>
                </div>

                <div class="mt-2"> <input id="progressBar" type="range" min="0" max="100" value="0"
                         class="block w-full max-w-md mx-auto" disabled> </div>

                <div class="mt-4 flex justify-center gap-2"> 
                    <button id="matchBtn1" class="bg-purple-200 hover:bg-purple-300 text-gray-800 px-3 py-2 rounded font-mono text-sm transition-colors duration-200 flex-1 max-w-[180px]">vs. o3-mini</button>
                    <button id="matchBtn2" class="bg-orange-200 hover:bg-orange-300 text-gray-800 px-3 py-2 rounded font-mono text-sm transition-colors duration-200 flex-1 max-w-[180px]">vs. Claude-3.7-Sonnet</button>
                    <button id="matchBtn3" class="bg-blue-200 hover:bg-blue-300 text-gray-800 px-3 py-2 rounded font-mono text-sm transition-colors duration-200 flex-1 max-w-[180px]">vs. Gemini-2.5-Pro</button>
                </div>

                <div class="mt-2 text-center">
                  <span id="gameInfo" class="font-mono text-xs text-gray-500"></span>
                </div>
              </div>
            </div>
          </div>`;
    }

    /* ============================ BOOT ============================= */
    function boot(jsonPath) {
        if (!jsonPath || typeof jsonPath !== 'string') {
            console.error('Boot aborted: Invalid JSON path provided.', jsonPath);
            updateFileStatus('Error: Invalid JSON path.', true);
            updateMatchButtons();
            return;
        }

        const absoluteURL = new URL(jsonPath, window.location.href).toString();
        currentAbsoluteURL = absoluteURL; // Store current absolute URL

        // Initial status update
        updateFileStatus(`Loading: ${jsonPath.split('/').pop()}...`);

        /* Fetch and parse JSON */
        fetch(absoluteURL, { cache: 'no-cache' })
            .then(async (res) => {
                if (!res.ok) throw new Error(`HTTP ${res.status} while fetching ${absoluteURL}`);
                const raw = await res.text();
                const clean = raw.replace(/^\uFEFF/, '').trim(); // strip BOM if present
                return JSON.parse(clean);
            })
            .then((json) => {
                gameJsonData = json;

                // Build or update JSON list from metadata if provided
                if (Array.isArray(gameJsonData.metadata?.file_list)) {
                    const metadataFiles = gameJsonData.metadata.file_list
                        .map(p => {
                            if (p && typeof p === 'string') {
                                try {
                                    const resolvedUrl = new URL(p, window.location.href).toString();
                                    // Prevent URLs that literally end with "/null"
                                    if (resolvedUrl.endsWith('/null')) {
                                        console.warn(`Path in metadata.file_list resolved to an URL ending with '/null': '${p}' -> '${resolvedUrl}'. Skipping.`);
                                        return null;
                                    }
                                    return resolvedUrl;
                                } catch (e) {
                                    console.warn(`Invalid path in metadata.file_list: '${p}'. Error: ${e.message}`);
                                    return null;
                                }
                            }
                            return null; // Non-string or empty entries are invalid
                        })
                        .filter(p => p !== null); // Remove invalid paths

                    if (metadataFiles.length > 0) {
                        jsonList = metadataFiles; // Replace global jsonList (already normalized)
                        console.log('Updated jsonList from metadata:', jsonList);
                    }
                }

                // If jsonList ended up empty, use the current file as fallback
                if (jsonList.length === 0 && absoluteURL && !absoluteURL.endsWith('/null')) {
                    jsonList = [absoluteURL];
                }

                // Update currentJsonIndex based on the current file
                currentJsonIndex = jsonList.indexOf(absoluteURL);
                console.log('Current file index:', currentJsonIndex, 'in list:', jsonList);

                const p1 = gameJsonData.metadata?.models?.['1'] || 'ViGaL (Ours)';
                const p2 = gameJsonData.metadata?.models?.['2'] || 'Opponent';
                updateFileStatus(`${p1} vs ${p2}`);

                // Update Match buttons state
                updateMatchButtons();
                
                initGame();
            })
            .catch((err) => {
                console.error(err);
                updateFileStatus(`Error: ${err.message}`, true);
                // Optionally disable controls if a game can't load
                ['playBtn','prevBtn','nextBtn','endBtn', 'progressBar'].forEach(id => {
                    const el = document.getElementById(id);
                    if (el) el.disabled = true;
                });
                updateMatchButtons();
            });
    }

    /* ======================= GAME INIT ========================= */
    function initGame() {
        if (!gameJsonData || !Array.isArray(gameJsonData.rounds)) {
            updateFileStatus('Error: Game data is invalid or missing rounds.', true);
            throw new Error('Game data is invalid or missing rounds array in JSON');
        }

        maxRounds = gameJsonData.metadata?.actual_rounds || gameJsonData.rounds.length;
        // Add one extra round for the final display state
        maxRounds = maxRounds + 1;
        ({ width: W = 10, height: H = 10 } = gameJsonData.rounds[0] || {});

        document.getElementById('player1Name').textContent =
            gameJsonData.metadata?.models?.['1'] || 'ViGaL (Ours)';
        document.getElementById('player2Name').textContent =
            gameJsonData.metadata?.models?.['2'] || 'Opponent';

        const pb = document.getElementById('progressBar');
        pb.max = Math.max(0, maxRounds - 1); // Ensure max is not negative
        pb.value = 0; // Reset progress bar
        currentRound = 0; // Reset current round
        playing = false; // Reset playing state

        pb.disabled = (maxRounds === 0);
        ['playBtn', 'prevBtn', 'nextBtn', 'endBtn']
            .forEach(id => document.getElementById(id).disabled = (maxRounds === 0));

        setupGameControls(); // Set up game control event listeners
        render(); // initial render
    }

    /* ======================= GAME CONTROLS ========================== */
    function setupGameControls() {
        const on = (id, event, fn) => {
            const el = document.getElementById(id);
            if (el) {
                el.removeEventListener(event, el._snakeGameHandler);
                el._snakeGameHandler = fn;
                el.addEventListener(event, fn);
            }
        };

        on('playBtn', 'click', () => {
            if (!gameJsonData || maxRounds === 0) return;
            if (!playing && currentRound === maxRounds - 1) { // Replay
                currentRound = 0;
                playing = true;
            } else {
                playing = !playing;
            }
            render();
        });
        on('prevBtn', 'click', () => {
            if (!gameJsonData || maxRounds === 0) return;
            currentRound = Math.max(currentRound - 1, 0);
            playing = false; // Stop playing on manual navigation
            render();
        });
        on('nextBtn', 'click', () => {
            if (!gameJsonData || maxRounds === 0) return;
            currentRound = Math.min(currentRound + 1, maxRounds - 1);
            playing = false; // Stop playing on manual navigation
            render();
        });
        on('endBtn', 'click', () => {
            if (!gameJsonData || maxRounds === 0) return;
            currentRound = maxRounds - 1;
            playing = false;
            render();
        });

        on('progressBar', 'input', (e) => {
            if (!gameJsonData || maxRounds === 0) return;
            currentRound = +e.target.value;
            playing = false; // Stop playing on manual seek
            render();
        });
    }

    /* ======================= HELPER FUNCTIONS ====================== */
    
    /**
     * Format death reason for better display
     * @param {string} reason - Raw death reason from JSON
     * @returns {string} - Formatted death reason
     */
    function formatDeathReason(reason) {
        if (!reason) return 'unknown';
        
        switch (reason.toLowerCase()) {
            case 'wall':
                return 'collide into the wall';
            case 'body_collision':
                return 'collide with itself or another snake';
            default:
                return reason; // Return original if not recognized
        }
    }

    /**
     * Format apple count with emoji
     * @param {number} count - Number of apples
     * @returns {string} - Formatted string with apple emoji
     */
    function formatAppleCount(count) {
        return count === 1 ? '1 🍎' : `${count} 🍎`;
    }

    /* ======================= DRAW HELPERS ====================== */
    function drawBoard() {
        const s = Math.min(canvas.width / W, canvas.height / H);
        ctx.fillStyle = '#f9fafb'; // Light gray background for the board
        ctx.fillRect(0, 0, W * s, H * s);
        ctx.strokeStyle = '#e5e7eb'; // Grid line color

        for (let i = 0; i <= W; i++) {
            ctx.beginPath();
            ctx.moveTo(i * s, 0);
            ctx.lineTo(i * s, H * s);
            ctx.stroke();
        }
        for (let j = 0; j <= H; j++) {
            ctx.beginPath();
            ctx.moveTo(0, j * s);
            ctx.lineTo(W * s, j * s);
            ctx.stroke();
        }
    }

    function drawRound() {
        if (!gameJsonData || !gameJsonData.rounds) {
             console.warn("drawRound called with invalid game data.");
             return; // Prevent errors if data is not ready
        }
        
        // Check if we're in the final display state (beyond actual rounds)
        const actualRounds = gameJsonData.metadata?.actual_rounds || gameJsonData.rounds.length;
        const isFinalDisplay = currentRound >= actualRounds;
        
        // Use the last actual round's data for the final display
        const roundIndex = isFinalDisplay ? actualRounds - 1 : currentRound;
        const rd = gameJsonData.rounds[roundIndex];
        
        if (!rd) {
            console.warn(`No data for round ${roundIndex}.`);
            return;
        }
        
        const s = Math.min(canvas.width / W, canvas.height / H);

        drawBoard();

        /* Draw apples */
        ctx.fillStyle = '#ef4444'; // Red for apples
        rd.apples?.forEach(([x, y]) => {
            ctx.beginPath();
            ctx.arc(x * s + s / 2, (H - 1 - y) * s + s / 2, s / 3, 0, Math.PI * 2);
            ctx.fill();
        });

        /* Draw snakes */
        const colors = { '1': '#4F7022', '2': '#036C8E' }; // P1 Green, P2 Blue
        Object.entries(rd.snake_positions || {}).forEach(([pid, segs]) => {
            let alive = rd.alive?.[pid];
            
            // In final display state, override alive status based on game result
            if (isFinalDisplay && gameJsonData.metadata?.game_result) {
                alive = gameJsonData.metadata.game_result[pid] === 'won';
            }
            
            ctx.fillStyle = alive ? colors[pid] : '#9ca3af'; // Gray for eliminated

            segs.forEach(([x, y], i) => {
                ctx.globalAlpha = i ? Math.max(0.3, 0.8 - i * 0.1) : 1; // Fade tail
                ctx.fillRect(x * s + 2, (H - 1 - y) * s + 2, s - 4, s - 4);
            });
            ctx.globalAlpha = 1; // Reset alpha
        });
    }

    /* ======================= THOUGHT PARSER ==================== */
    function escapeHTML(str) {
        return str.replace(/[&<>"']/g, ch => (
            { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]
        ));
    }

    const thinkRegex = /<think>([\s\S]*?)<\/think>/i;
    const bestRegex = /<best_answer>([\s\S]*?)<\/best_answer>/i;
    const worstRegex = /<worst_answer>([\s\S]*?)<\/worst_answer>/i;

    function thoughtLines(rd, pid) {
        const snippets = [];
        if (!rd) return snippets; // Safety check

        if (!rd.alive?.[pid]) {
            snippets.push(`<p>Round ${rd.round_number ?? currentRound}: ELIMINATED</p>`);
            return snippets;
        }

        // Fix the move history lookup - get the most recent move for this round
        let mv = null;
        if (rd.move_history && Array.isArray(rd.move_history) && rd.move_history.length > 0) {
            // move_history contains all moves leading up to this round, get the last (most recent) one
            const moveObj = rd.move_history[rd.move_history.length - 1]; // Get the last move object
            if (moveObj && moveObj[pid]) {
                mv = moveObj[pid];
            }
        } else if (rd.move_history && rd.move_history[pid]) {
            // If move_history is an object with player IDs as keys
            mv = rd.move_history[pid];
        }
        
        // Also check if there's a direct moves or actions property
        if (!mv && rd.moves && rd.moves[pid]) {
            mv = rd.moves[pid];
        }
        if (!mv && rd.actions && rd.actions[pid]) {
            mv = rd.actions[pid];
        }

        if (mv && mv.rationale) {
            const txt = mv.rationale;
            const mThink = thinkRegex.exec(txt);
            const mBest = bestRegex.exec(txt);
            const mWorst = worstRegex.exec(txt);

            if (mThink && mThink[1]) {
                const content = mThink[1].trim().split('\n').map(line => escapeHTML(line)).join('<br>');
                snippets.push(`
                  <details class="toggle-thought mb-1"${pid === '1' ? ' open' : ''}>
                    <summary class="font-mono text-xs cursor-pointer flex items-center gap-1">
                      <svg class="toggle-arrow w-3 h-3 stroke-current text-gray-600 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 12 12">
                        <path d="M3 4.5l3 3 3-3" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                      </svg>
                      <span>Thought:</span>
                    </summary>
                    <div class="pl-4 text-xs text-gray-700 mt-1">${content}</div>
                  </details>
                `);
            }
            if (mBest && mBest[1]) {
                snippets.push(`<p class="font-mono text-xs">Best move: <b>${escapeHTML(mBest[1].trim())}</b></p>`);
            }
            if (mWorst && mWorst[1]) {
                snippets.push(`<p class="font-mono text-xs">Worst move: <b>${escapeHTML(mWorst[1].trim())}</b></p>`);
            }
        }
        
        // Add debug info if no move found
        if (!mv) {
            console.log(`No move found for player ${pid} in round ${currentRound}. Move history length: ${rd.move_history ? rd.move_history.length : 'N/A'}`);
            if (rd.move_history && rd.move_history.length > 0) {
                console.log('Move history structure:', rd.move_history);
            }
        }
        
        return snippets;
    }

    /* ======================= RENDER LOOP ======================= */
    function render() {
        if (!gameJsonData || !gameJsonData.rounds || currentRound < 0 || currentRound >= maxRounds ) {
             // If data is not ready or currentRound is out of bounds, don't render game state.
             // Initial blank draw or a loading state might be handled elsewhere or by default canvas state.
             if (ctx) { // Draw a blank board if context is available
                ctx.fillStyle = '#f3f4f6'; // Page background color
                ctx.fillRect(0, 0, canvas.width, canvas.height);
             }
            return;
        }

        // Check if we're in the final display state
        const actualRounds = gameJsonData.metadata?.actual_rounds || gameJsonData.rounds.length;
        const isFinalDisplay = currentRound >= actualRounds;
        const roundIndex = isFinalDisplay ? actualRounds - 1 : currentRound;
        const rd = gameJsonData.rounds[roundIndex];
        
        if (!rd) { // Safety check for the specific round data
            console.warn(`No data for round ${roundIndex}.`);
            return;
        }

        // Update round display
        if (isFinalDisplay) {
            document.getElementById('roundInfo').textContent = `FINAL RESULT`;
        } else {
            document.getElementById('roundInfo').textContent =
                `Round ${currentRound}/${Math.max(0, actualRounds - 1)}`;
        }
        
        // Update player names to show Winner/Loser at all rounds
        const player1Name = gameJsonData.metadata?.models?.['1'] || 'ViGaL (Ours)';
        const player2Name = gameJsonData.metadata?.models?.['2'] || 'Opponent';
        const gameResult1 = gameJsonData.metadata?.game_result?.['1'];
        const gameResult2 = gameJsonData.metadata?.game_result?.['2'];
        
        if (gameResult1 === 'won') {
            document.getElementById('player1Name').textContent = `Winner: ${player1Name}`;
            document.getElementById('player2Name').textContent = `Loser: ${player2Name}`;
        } else if (gameResult2 === 'won') {
            document.getElementById('player1Name').textContent = `Loser: ${player1Name}`;
            document.getElementById('player2Name').textContent = `Winner: ${player2Name}`;
        } else {
            // Fallback if no clear winner
            document.getElementById('player1Name').textContent = player1Name;
            document.getElementById('player2Name').textContent = player2Name;
        }
        
        document.getElementById('progressBar').value = currentRound;

        ['1', '2'].forEach(pid => {
            const thoughtEl = document.getElementById(`player${pid}Thoughts`);

            let alive = rd.alive?.[pid];
            const st = document.getElementById(`player${pid}Status`);
            
            // Calculate apple count/score
            let appleCount = 0;
            let scoreText = '';
            
            // Try different methods to get apple count
            if (rd.scores && typeof rd.scores[pid] === 'number') {
                appleCount = rd.scores[pid];
            } 
            else if (rd.apple_count && typeof rd.apple_count[pid] === 'number') {
                appleCount = rd.apple_count[pid];
            }
            else if (rd.snake_lengths && typeof rd.snake_lengths[pid] === 'number') {
                // Get initial snake length from first round
                let initialLength = 1; // default
                if (gameJsonData.rounds && gameJsonData.rounds[0] && 
                    gameJsonData.rounds[0].snake_lengths && 
                    typeof gameJsonData.rounds[0].snake_lengths[pid] === 'number') {
                    initialLength = gameJsonData.rounds[0].snake_lengths[pid];
                }
                appleCount = Math.max(0, rd.snake_lengths[pid] - initialLength);
            }
            else if (rd.snake_positions && rd.snake_positions[pid] && Array.isArray(rd.snake_positions[pid])) {
                // Get initial snake length from first round
                let initialLength = 1; // default
                if (gameJsonData.rounds && gameJsonData.rounds[0] && 
                    gameJsonData.rounds[0].snake_positions && 
                    gameJsonData.rounds[0].snake_positions[pid] && 
                    Array.isArray(gameJsonData.rounds[0].snake_positions[pid])) {
                    initialLength = gameJsonData.rounds[0].snake_positions[pid].length;
                }
                appleCount = Math.max(0, rd.snake_positions[pid].length - initialLength);
            }
            
            // Always show apple count
            scoreText = ` (${formatAppleCount(appleCount)})`;
            
            // In final display state, show winner/loser status
            if (isFinalDisplay && gameJsonData.metadata?.game_result) {
                const result = gameJsonData.metadata.game_result[pid];
                if (result === 'won') {
                    st.textContent = `WINNER${scoreText}`;
                    st.className = 'text-green-600 font-bold text-sm animate-pulse';
                } else if (result === 'lost') {
                    st.textContent = `LOSER${scoreText}`;
                    st.className = 'text-red-600 font-bold text-sm';
                } else {
                    st.textContent = `UNKNOWN${scoreText}`;
                    st.className = 'text-gray-500 font-bold text-sm';
                }
            } else {
                // Normal round display
                st.textContent = alive ? `ALIVE${scoreText}` : `ELIMINATED${scoreText}`;
                st.className = `${alive ? 'text-green-500' : 'text-red-500'} font-bold text-sm`;
            }

            // Show thoughts only for non-final display rounds
            if (!isFinalDisplay) {
                thoughtEl.innerHTML = thoughtLines(rd, pid).join('');

                const newDetails = thoughtEl.querySelector('details');
                if (newDetails) {
                    // Always open P1 thoughts, P2 closed by default, or maintain previous state if preferred.
                    newDetails.open = pid === '1' ? true : false; 
                }
            } else {
                // Show final game summary in thoughts panel
                const gameResult = gameJsonData.metadata?.game_result?.[pid];
                const deathInfo = gameJsonData.metadata?.death_info?.[pid];
                const finalScore = gameJsonData.metadata?.final_scores?.[pid] || appleCount;
                
                let summaryHtml = `<div class="text-center">`;
                if (gameResult === 'won') {
                    summaryHtml += `<p class="text-green-600 font-bold text-lg mb-2">🏆 VICTORY!</p>`;
                } else if (gameResult === 'lost') {
                    summaryHtml += `<p class="text-red-600 font-bold text-lg mb-2">💀 DEFEATED</p>`;
                }
                
                summaryHtml += `<p class="font-mono text-sm mb-2">Final Score: ${formatAppleCount(finalScore)}</p>`;
                
                if (deathInfo) {
                    summaryHtml += `<p class="font-mono text-xs text-gray-600">Death reason: ${formatDeathReason(deathInfo.reason)} (Round ${deathInfo.round})</p>`;
                }
                
                summaryHtml += `</div>`;
                thoughtEl.innerHTML = summaryHtml;
            }
        });

        const playBtn = document.getElementById('playBtn');
        if (!playing && currentRound === maxRounds - 1 && maxRounds > 0) {
            playBtn.textContent = '🔄 Replay';
        } else {
            playBtn.textContent = playing ? '⏸️ Pause' : '▶️ Play';
        }

        // Update game info section
        const gameInfoEl = document.getElementById('gameInfo');
        if (gameInfoEl) {
            if (isFinalDisplay && gameJsonData.metadata) {
                // Show final game statistics
                const meta = gameJsonData.metadata;
                let infoText = '';
                
                if (meta.time_taken) {
                    const minutes = Math.floor(meta.time_taken / 60);
                    const seconds = Math.round(meta.time_taken % 60);
                    infoText += `Game Duration: ${minutes}m ${seconds}s`;
                }
                
                if (meta.actual_rounds) {
                    if (infoText) infoText += ' • ';
                    infoText += `Total Rounds: ${meta.actual_rounds}`;
                }
                
                if (meta.final_scores) {
                    if (infoText) infoText += ' • ';
                    const scores = Object.entries(meta.final_scores).map(([pid, score]) => `P${pid}: ${formatAppleCount(score)}`).join(' vs ');
                    infoText += `Final Scores: ${scores}`;
                }
                
                gameInfoEl.textContent = infoText;
            } else {
                // Clear game info for normal rounds
                gameInfoEl.textContent = '';
            }
        }

        drawRound();
    }

    /* Advance the "playing" loop based on FPS */
    // Ensure interval is managed (cleared and set) if boot can be called multiple times
    // For now, assuming one main interval.
    if (!window.snakeGameInterval) { // Basic guard against multiple intervals
         window.snakeGameInterval = setInterval(() => {
            if (playing && gameJsonData && maxRounds > 0) {
                if (currentRound < maxRounds - 1) {
                    currentRound++;
                } else {
                    playing = false; // End of game
                }
                render();
            }
        }, 1000 / FPS);
    }

    /* Draw a blank background immediately, before JSON loads fully if canvas is ready */
    if (canvas && ctx) {
        ctx.fillStyle = '#f3f4f6'; // Page background color
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    /**
     * Convert jsonList entries to absolute URLs
     */
    function normalizeJsonList() {
        jsonList = jsonList.map(path => {
            if (path && typeof path === 'string') {
                try {
                    return new URL(path, window.location.href).toString();
                } catch (e) {
                    console.warn(`Invalid path in jsonList: '${path}'. Error: ${e.message}`);
                    return null;
                }
            }
            return null;
        }).filter(url => url !== null);
        
        console.log('Normalized jsonList:', jsonList);
    }
})();