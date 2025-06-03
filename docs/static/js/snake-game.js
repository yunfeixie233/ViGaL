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
     * Set up the Next Match button handler (only once)
     */
    function setupNextMatchButton() {
        const nextMatchBtn = document.getElementById('nextMatchBtn');
        if (!nextMatchBtn) return;

        // Remove any existing event listeners to prevent duplicates
        nextMatchBtn.onclick = null;
        
        nextMatchBtn.onclick = () => {
            console.log('Next Match clicked. Current index:', currentJsonIndex, 'JSON list length:', jsonList.length);
            console.log('Current absolute URL:', currentAbsoluteURL);
            console.log('JSON list:', jsonList);
            
            if (jsonList.length === 0) {
                console.error('JSON list is empty. Cannot switch to next match.');
                updateFileStatus('Error: No game files available for "Next Match".', true);
                return;
            }

            if (jsonList.length === 1) {
                console.log('Only one file in list, staying on current file.');
                updateFileStatus('Only one game file available.', false);
                return;
            }

            // Calculate next index
            let nextIdx;
            if (currentJsonIndex === -1) {
                // Current file not in list, start from beginning
                nextIdx = 0;
                console.log('Current file not found in list, starting from index 0');
            } else {
                // Move to next file in list
                nextIdx = (currentJsonIndex + 1) % jsonList.length;
                console.log('Moving from index', currentJsonIndex, 'to index', nextIdx);
            }

            const nextJsonPath = jsonList[nextIdx];
            console.log('Loading next file:', nextJsonPath, 'at index:', nextIdx);
            
            if (nextJsonPath && typeof nextJsonPath === 'string') {
                boot(nextJsonPath);
            } else {
                console.error('Next JSON path is invalid:', nextJsonPath);
                updateFileStatus('Error: Could not determine next valid match.', true);
            }
        };
    }

    /**
     * Update the Next Match button state
     */
    function updateNextMatchButton() {
        const nextMatchBtn = document.getElementById('nextMatchBtn');
        if (!nextMatchBtn) return;

        // Disable if we have 0 or 1 files
        nextMatchBtn.disabled = jsonList.length <= 1;
        
        if (jsonList.length <= 1) {
            nextMatchBtn.title = jsonList.length === 0 ? 'No game files available' : 'Only one game file available';
        } else {
            nextMatchBtn.title = `Switch to next game (${jsonList.length} files available)`;
        }
    }

    document.addEventListener('DOMContentLoaded', () => {
        const container = document.getElementById('snake-game-container');
        if (!container) return;
        container.innerHTML = markup();    // inject template
        
        // Get canvas and context
        canvas = document.getElementById('gameCanvas');
        ctx = canvas ? canvas.getContext('2d') : null;
        
        // Normalize jsonList to absolute URLs
        normalizeJsonList();
        
        // Set up Next Match button (only once)
        setupNextMatchButton();
        
        const path = container.getAttribute('data-json') || DEFAULT_JSON;
        boot(path);                      // start boot sequence
    });

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
                  <div class="p-3 border-b border-gray-100 sticky top-0 z-10 text-center player1-bg bg-opacity-50 bg-green-50">
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
                  <div class="p-3 border-b border-gray-100 sticky top-0 z-10 text-center player2-bg bg-opacity-50 bg-blue-50">
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
                  <button id="playBtn"  class="bg-blue-500 text-white px-4 py-2 rounded font-mono text-sm" disabled>⏸️ Pause</button>
                  <button id="prevBtn"  class="bg-gray-500 text-white px-4 py-2 rounded font-mono text-sm" disabled>⏮️ Prev</button>
                  <span id="roundInfo" class="font-mono text-sm text-gray-600">Round 0/0</span>
                  <button id="nextBtn"  class="bg-gray-500 text-white px-4 py-2 rounded font-mono text-sm" disabled>⏭️ Next</button>
                  <button id="endBtn"   class="bg-gray-500 text-white px-4 py-2 rounded font-mono text-sm" disabled>⏩ End</button>
                </div>

                <div class="mt-2"> <input id="progressBar" type="range" min="0" max="100" value="0"
                         class="block w-full max-w-md mx-auto" disabled> </div>

                <div class="mt-2 flex justify-center"> <button id="nextMatchBtn" class="bg-gray-500 text-white px-4 py-2 rounded font-mono text-sm">Next Match</button>
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
            updateNextMatchButton();
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

                // Update Next Match button state
                updateNextMatchButton();
                
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
                updateNextMatchButton();
            });
    }

    /* ======================= GAME INIT ========================= */
    function initGame() {
        if (!gameJsonData || !Array.isArray(gameJsonData.rounds)) {
            updateFileStatus('Error: Game data is invalid or missing rounds.', true);
            throw new Error('Game data is invalid or missing rounds array in JSON');
        }

        maxRounds = gameJsonData.metadata?.actual_rounds || gameJsonData.rounds.length;
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
        if (!gameJsonData || !gameJsonData.rounds || !gameJsonData.rounds[currentRound]) {
             console.warn("drawRound called with invalid game data or currentRound out of bounds.");
             return; // Prevent errors if data is not ready
        }
        const rd = gameJsonData.rounds[currentRound];
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
            const alive = rd.alive?.[pid];
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

        const mv = rd.move_history?.find(m => m[pid])?.[pid] || rd.move_history?.at(-1)?.[pid]; // Try to find latest move for pid
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

        const rd = gameJsonData.rounds[currentRound];
         if (!rd) { // Safety check for the specific round data
            console.warn(`No data for round ${currentRound}.`);
            return;
        }

        document.getElementById('roundInfo').textContent =
            `Round ${currentRound}/${Math.max(0, maxRounds - 1)}`;
        document.getElementById('progressBar').value = currentRound;

        ['1', '2'].forEach(pid => {
            const thoughtEl = document.getElementById(`player${pid}Thoughts`);

            const alive = rd.alive?.[pid];
            const st = document.getElementById(`player${pid}Status`);
            st.textContent = alive ? 'ALIVE' : 'ELIMINATED';
            st.className = `${alive ? 'text-green-500' : 'text-red-500'} font-bold text-sm`;

            thoughtEl.innerHTML = thoughtLines(rd, pid).join('');

            const newDetails = thoughtEl.querySelector('details');
            if (newDetails) {
                // Always open P1 thoughts, P2 closed by default, or maintain previous state if preferred.
                newDetails.open = pid === '1' ? true : false; 
            }
        });

        const playBtn = document.getElementById('playBtn');
        if (!playing && currentRound === maxRounds - 1 && maxRounds > 0) {
            playBtn.textContent = '🔄 Replay';
        } else {
            playBtn.textContent = playing ? '⏸️ Pause' : '▶️ Play';
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