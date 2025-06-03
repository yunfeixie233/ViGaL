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
    let currentJsonIndex = 0; // Will be updated in boot based on the loaded JSON's presence in jsonList

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

    document.addEventListener('DOMContentLoaded', () => {
        const container = document.getElementById('snake-game-container');
        if (!container) return;
        container.innerHTML = markup();    // inject template
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
            const nextMatchBtnEl = document.getElementById('nextMatchBtn');
            if (nextMatchBtnEl) nextMatchBtnEl.disabled = true;
            return;
        }

        const absoluteURL = new URL(jsonPath, window.location.href).toString();
        updateFileStatus(`Loading: ${jsonPath.split('/').pop()}...`);

        fetch(absoluteURL, { cache: 'no-cache' })
            .then(async (res) => {
                if (!res.ok) throw new Error(`HTTP ${res.status} while fetching ${absoluteURL}`);
                const raw = await res.text();
                const clean = raw.replace(/^\uFEFF/, '').trim();
                return JSON.parse(clean);
            })
            .then((json) => {
                gameJsonData = json;

                if (Array.isArray(gameJsonData.metadata?.file_list)) {
                    const metadataFiles = gameJsonData.metadata.file_list
                        .map(p => {
                            if (p && typeof p === 'string') {
                                try {
                                    const resolvedUrl = new URL(p, window.location.href).toString();
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
                            return null;
                        })
                        .filter(p => p !== null);

                    if (metadataFiles.length > 0) {
                        jsonList = metadataFiles;
                    }
                }

                if (jsonList.length === 0 && absoluteURL && !absoluteURL.endsWith('/null')) {
                    jsonList = [absoluteURL];
                }

                currentJsonIndex = jsonList.indexOf(absoluteURL);

                const p1 = gameJsonData.metadata?.models?.['1'] || 'ViGaL (Ours)';
                const p2 = gameJsonData.metadata?.models?.['2'] || 'Opponent';
                updateFileStatus(`${p1} vs ${p2}`);

                const nextMatchBtnEl = document.getElementById('nextMatchBtn');
                if (nextMatchBtnEl) {
                    if (jsonList.length === 0) {
                        nextMatchBtnEl.disabled = true;
                    } else {
                        const uniquePaths = new Set(jsonList);
                        nextMatchBtnEl.disabled = uniquePaths.size <= 1;
                    }
                }
                initGame(absoluteURL); // Pass absoluteURL to initGame for context
            })
            .catch((err) => {
                console.error(err);
                updateFileStatus(`Error: ${err.message}`, true);
                ['playBtn','prevBtn','nextBtn','endBtn', 'progressBar', 'nextMatchBtn'].forEach(id => {
                    const el = document.getElementById(id);
                    if (el) el.disabled = true;
                });
            });

        let maxRounds = 0,
            currentRound = 0,
            playing = false,
            W = 10, H = 10;

        const canvas = /** @type {HTMLCanvasElement} */ (document.getElementById('gameCanvas'));
        const ctx = canvas.getContext('2d');

        /* ======================= GAME INIT ========================= */
        // Pass currentLoadedURL (which is absoluteURL from boot) for the click handler's closure
        function initGame(currentLoadedURL) {
            if (!gameJsonData || !Array.isArray(gameJsonData.rounds)) {
                updateFileStatus('Error: Game data is invalid or missing rounds.', true);
                throw new Error('Game data is invalid or missing rounds array in JSON');
            }

            maxRounds = gameJsonData.metadata?.actual_rounds || gameJsonData.rounds.length;
            ({ width: W = 10, height: H = 10 } = gameJsonData.rounds[0] || {});

            document.getElementById('player1Name').textContent = gameJsonData.metadata?.models?.['1'] || 'ViGaL (Ours)';
            document.getElementById('player2Name').textContent = gameJsonData.metadata?.models?.['2'] || 'Opponent';

            const pb = document.getElementById('progressBar');
            pb.max = Math.max(0, maxRounds - 1);
            pb.value = 0;
            currentRound = 0;
            playing = false;

            pb.disabled = (maxRounds === 0);
            ['playBtn', 'prevBtn', 'nextBtn', 'endBtn']
                .forEach(id => document.getElementById(id).disabled = (maxRounds === 0));

            const nextMatchBtnEl = document.getElementById('nextMatchBtn');
            nextMatchBtnEl.onclick = () => {
                // This handler assumes the button is enabled, meaning jsonList has >1 unique paths.
                
                // Determine the starting index for our search.
                // If currentJsonIndex is -1 (current file not in list), start search from index 0.
                // Otherwise, start search from currentJsonIndex.
                const searchStartIndex = (currentJsonIndex === -1) ? 0 : currentJsonIndex;

                let nextPathToLoad = null;
                // Iterate through jsonList to find the next *different* URL.
                // We check up to jsonList.length items to ensure we cycle through the list.
                for (let i = 1; i <= jsonList.length; i++) {
                    const nextTryIndex = (searchStartIndex + i) % jsonList.length;
                    if (jsonList[nextTryIndex] !== currentLoadedURL) {
                        nextPathToLoad = jsonList[nextTryIndex];
                        break; // Found a different URL
                    }
                }

                if (nextPathToLoad) {
                    boot(nextPathToLoad);
                } else {
                    // This case implies all items in jsonList are identical to currentLoadedURL,
                    // or no distinct next item was found.
                    // The button should have been disabled by the uniquePaths.size <= 1 check in boot().
                    console.warn("Next Match clicked, but no different JSON file found to switch to. This might indicate an issue if the button was expected to be enabled.");
                }
            };
            render();
        }

        /* ======================= DRAW HELPERS ====================== */
        function drawBoard() {
            const s = Math.min(canvas.width / W, canvas.height / H);
            ctx.fillStyle = '#f9fafb';
            ctx.fillRect(0, 0, W * s, H * s);
            ctx.strokeStyle = '#e5e7eb';
            for (let i = 0; i <= W; i++) { ctx.beginPath(); ctx.moveTo(i * s, 0); ctx.lineTo(i * s, H * s); ctx.stroke(); }
            for (let j = 0; j <= H; j++) { ctx.beginPath(); ctx.moveTo(0, j * s); ctx.lineTo(W * s, j * s); ctx.stroke(); }
        }

        function drawRound() {
            if (!gameJsonData || !gameJsonData.rounds || !gameJsonData.rounds[currentRound]) {
                 console.warn("drawRound called with invalid game data or currentRound out of bounds."); return;
            }
            const rd = gameJsonData.rounds[currentRound];
            const s = Math.min(canvas.width / W, canvas.height / H);
            drawBoard();
            ctx.fillStyle = '#ef4444';
            rd.apples?.forEach(([x, y]) => { ctx.beginPath(); ctx.arc(x * s + s / 2, (H - 1 - y) * s + s / 2, s / 3, 0, Math.PI * 2); ctx.fill(); });
            const colors = { '1': '#4F7022', '2': '#036C8E' };
            Object.entries(rd.snake_positions || {}).forEach(([pid, segs]) => {
                const alive = rd.alive?.[pid];
                ctx.fillStyle = alive ? colors[pid] : '#9ca3af';
                segs.forEach(([x, y], i) => {
                    ctx.globalAlpha = i ? Math.max(0.3, 0.8 - i * 0.1) : 1;
                    ctx.fillRect(x * s + 2, (H - 1 - y) * s + 2, s - 4, s - 4);
                });
                ctx.globalAlpha = 1;
            });
        }

        /* ======================= THOUGHT PARSER ==================== */
        function escapeHTML(str) { return str.replace(/[&<>"']/g, ch => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' }[ch])); }
        const thinkRegex = /<think>([\s\S]*?)<\/think>/i;
        const bestRegex = /<best_answer>([\s\S]*?)<\/best_answer>/i;
        const worstRegex = /<worst_answer>([\s\S]*?)<\/worst_answer>/i;
        function thoughtLines(rd, pid) {
            const snippets = []; if (!rd) return snippets;
            if (!rd.alive?.[pid]) { snippets.push(`<p>Round ${rd.round_number ?? currentRound}: ELIMINATED</p>`); return snippets; }
            const mv = rd.move_history?.find(m => m[pid])?.[pid] || rd.move_history?.at(-1)?.[pid];
            if (mv && mv.rationale) {
                const txt = mv.rationale; const mThink = thinkRegex.exec(txt); const mBest = bestRegex.exec(txt); const mWorst = worstRegex.exec(txt);
                if (mThink && mThink[1]) { const content = mThink[1].trim().split('\n').map(line => escapeHTML(line)).join('<br>');
                    snippets.push(`<details class="toggle-thought mb-1"${pid === '1' ? ' open' : ''}><summary class="font-mono text-xs cursor-pointer flex items-center gap-1"><svg class="toggle-arrow w-3 h-3 stroke-current text-gray-600 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 12 12"><path d="M3 4.5l3 3 3-3" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg><span>Thought:</span></summary><div class="pl-4 text-xs text-gray-700 mt-1">${content}</div></details>`);
                }
                if (mBest && mBest[1]) { snippets.push(`<p class="font-mono text-xs">Best move: <b>${escapeHTML(mBest[1].trim())}</b></p>`); }
                if (mWorst && mWorst[1]) { snippets.push(`<p class="font-mono text-xs">Worst move: <b>${escapeHTML(mWorst[1].trim())}</b></p>`); }
            } return snippets;
        }

        /* ======================= RENDER LOOP ======================= */
        function render() {
            if (!gameJsonData || !gameJsonData.rounds || currentRound < 0 || currentRound >= maxRounds ) {
                 if (ctx) { ctx.fillStyle = '#f3f4f6'; ctx.fillRect(0, 0, canvas.width, canvas.height); } return;
            }
            const rd = gameJsonData.rounds[currentRound]; if (!rd) { console.warn(`No data for round ${currentRound}.`); return; }
            document.getElementById('roundInfo').textContent = `Round ${currentRound}/${Math.max(0, maxRounds - 1)}`;
            document.getElementById('progressBar').value = currentRound;
            ['1', '2'].forEach(pid => {
                const thoughtEl = document.getElementById(`player${pid}Thoughts`);
                const alive = rd.alive?.[pid]; const st = document.getElementById(`player${pid}Status`);
                st.textContent = alive ? 'ALIVE' : 'ELIMINATED'; st.className = `${alive ? 'text-green-500' : 'text-red-500'} font-bold text-sm`;
                thoughtEl.innerHTML = thoughtLines(rd, pid).join('');
                const newDetails = thoughtEl.querySelector('details'); if (newDetails) { newDetails.open = pid === '1' ? true : false; }
            });
            const playBtn = document.getElementById('playBtn');
            if (!playing && currentRound === maxRounds - 1 && maxRounds > 0) { playBtn.textContent = '🔄 Replay'; }
            else { playBtn.textContent = playing ? '⏸️ Pause' : '▶️ Play'; }
            drawRound();
        }

        /* ======================= CONTROLS ========================== */
        const on = (id, event, fn) => { const el = document.getElementById(id); if (el) el.addEventListener(event, fn); };
        on('playBtn', 'click', () => { if (!gameJsonData || maxRounds === 0) return; if (!playing && currentRound === maxRounds - 1) { currentRound = 0; playing = true; } else { playing = !playing; } render(); });
        on('prevBtn', 'click', () => { if (!gameJsonData || maxRounds === 0) return; currentRound = Math.max(currentRound - 1, 0); playing = false; render(); });
        on('nextBtn', 'click', () => { if (!gameJsonData || maxRounds === 0) return; currentRound = Math.min(currentRound + 1, maxRounds - 1); playing = false; render(); });
        on('endBtn', 'click', () => { if (!gameJsonData || maxRounds === 0) return; currentRound = maxRounds - 1; playing = false; render(); });
        on('progressBar', 'input', (e) => { if (!gameJsonData || maxRounds === 0) return; currentRound = +e.target.value; playing = false; render(); });

        if (!window.snakeGameInterval) {
             window.snakeGameInterval = setInterval(() => {
                if (playing && gameJsonData && maxRounds > 0) {
                    if (currentRound < maxRounds - 1) { currentRound++; } else { playing = false; } render();
                }
            }, 1000 / FPS);
        }
        if (ctx) { ctx.fillStyle = '#f3f4f6'; ctx.fillRect(0, 0, canvas.width, canvas.height); }
    } // End of boot function
})();