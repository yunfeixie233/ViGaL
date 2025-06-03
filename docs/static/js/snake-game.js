/*  snake-game.js  */
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

    document.addEventListener('DOMContentLoaded', () => {
        const container = document.getElementById('snake-game-container');
        if (!container) return;
        container.innerHTML = markup();    // inject template
        const path = container.getAttribute('data-json') || DEFAULT_JSON;
        boot(path);                        // start boot sequence
    });

    /* ========================= TEMPLATE ============================ */
    function markup() {
        return `
          <div class="h-full flex flex-col">
            <!-- Title & Controls Wrapper -->
            <div class="bg-gray-100 flex-1 overflow-hidden">
              <!-- Status line -->
              <div class="bg-white shadow rounded-lg p-4 mb-4 text-center">
                <span id="fileStatus" class="font-mono text-lg text-black font-bold">Loading…</span>
              </div>

              <!-- Main grid: Thoughts panels and Canvas -->
              <div class="grid grid-cols-1 lg:grid-cols-[1fr_min-content_1fr] gap-4 mb-6 h-full">
                <!-- ViGaL (Ours) thoughts / status -->
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

                <!-- Canvas -->
                <div class="flex items-center justify-center">
                  <canvas id="gameCanvas" width="400" height="400" class="game-canvas rounded-lg shadow"></canvas>
                </div>

                <!-- Player 2 thoughts / status -->
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

              <!-- Controls -->
              <div class="bg-white shadow rounded-lg p-4 mb-4 text-center">
                <div class="flex justify-center items-center gap-4">
                  <button id="playBtn"  class="bg-blue-500 text-white px-4 py-2 rounded font-mono text-sm" disabled>⏸️ Pause</button>
                  <button id="prevBtn"  class="bg-gray-500 text-white px-4 py-2 rounded font-mono text-sm" disabled>⏮️ Prev</button>
                  <span id="roundInfo" class="font-mono text-sm text-gray-600">Round 0/0</span>
                  <button id="nextBtn"  class="bg-gray-500 text-white px-4 py-2 rounded font-mono text-sm" disabled>⏭️ Next</button>
                  <button id="endBtn"   class="bg-gray-500 text-white px-4 py-2 rounded font-mono text-sm" disabled>⏩ End</button>
                </div>
                <div class="flex justify-center items-center mt-2">
                  <input id="progressBar" type="range" min="0" max="100" value="0"
                         class="flex-1 max-w-md" disabled>
                </div>
                <div class="flex justify-center items-center mt-2">
                  <button id="nextMatchBtn" class="bg-gray-500 text-white px-4 py-2 rounded font-mono text-sm">Next Match</button>
                </div>
                <!-- Removed Game ID / Time display -->
                <div class="mt-2">
                  <span id="gameInfo" class="font-mono text-xs text-gray-500"></span>
                </div>
              </div>
            </div>
          </div>`;
    }

    /* ============================ BOOT ============================= */
    function boot(jsonPath) {
        const fileStatus  = document.getElementById('fileStatus');
        const absoluteURL = new URL(jsonPath, window.location.href).toString();

        /* Fetch and parse JSON */
        fetch(absoluteURL, { cache: 'no-cache' })
          .then(async (res) => {
            if (!res.ok) throw new Error(`HTTP ${res.status} while fetching ${absoluteURL}`);
            const raw   = await res.text();
            const clean = raw.replace(/^\uFEFF/, '').trim();   // strip BOM if present
            return JSON.parse(clean);
          })
          .then((json) => {
            gameJsonData = json;

            // Build or update JSON list from metadata if provided
            if (Array.isArray(gameJsonData.metadata?.file_list)) {
              jsonList = gameJsonData.metadata.file_list.map(p => new URL(p, window.location.href).toString());
            } else if (jsonList.length === 0) {
              jsonList = [absoluteURL];
            }
            currentJsonIndex = jsonList.indexOf(absoluteURL);

            // Display "ViGaL (Ours) vs <OpponentName>" instead of Loaded URL
            const p1 = gameJsonData.metadata?.models?.['1'] || 'ViGaL (Ours)';
            const p2 = gameJsonData.metadata?.models?.['2'] || 'Opponent';
            fileStatus.textContent = `${p1} vs ${p2}`;
            fileStatus.className   = 'font-mono text-lg text-black font-bold';

            initGame();
          })
          .catch((err) => {
            console.error(err);
            fileStatus.textContent = `Error: ${err.message}`;
            fileStatus.className   = 'font-mono text-sm text-red-600';
          });

        /* Local state */
        let maxRounds    = 0,
            currentRound = 0,
            playing      = false,
            W = 10,
            H = 10;

        const canvas = /** @type {HTMLCanvasElement} */ (document.getElementById('gameCanvas'));
        const ctx    = canvas.getContext('2d');

        /* ======================= GAME INIT ========================= */
        function initGame() {
          if (!Array.isArray(gameJsonData.rounds))
            throw new Error('rounds array missing in JSON');

          maxRounds = gameJsonData.metadata?.actual_rounds || gameJsonData.rounds.length;
          ({ width: W = 10, height: H = 10 } = gameJsonData.rounds[0] || {});

          // Set player names (dark black bold)
          document.getElementById('player1Name').textContent =
            gameJsonData.metadata?.models?.['1'] || 'ViGaL (Ours)';
          document.getElementById('player2Name').textContent =
            gameJsonData.metadata?.models?.['2'] || 'Opponent';

          // Removed Game ID and Time display assignment
          // document.getElementById('gameInfo').textContent =
          //   `Game ID: ${gameJsonData.metadata?.game_id ?? 'N/A'} | Time: ${(gameJsonData.metadata?.time_taken ?? 0).toFixed(1)}s`;

          const pb = document.getElementById('progressBar');
          pb.max      = maxRounds - 1;
          pb.disabled = false;

          ['playBtn','prevBtn','nextBtn','endBtn']
            .forEach(id => document.getElementById(id).disabled = false);

          // Attach Next Match handler
          document.getElementById('nextMatchBtn').onclick = () => {
            const nextIndex = (currentJsonIndex + 1) % jsonList.length;
            boot(jsonList[nextIndex]);
          };

          render();    // initial render
        }

        /* ======================= DRAW HELPERS ====================== */
        function drawBoard() {
          const s = Math.min(canvas.width / W, canvas.height / H);
          ctx.fillStyle = '#f9fafb';
          ctx.fillRect(0, 0, W * s, H * s);
          ctx.strokeStyle = '#e5e7eb';

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
          const rd = gameJsonData.rounds[currentRound];
          const s  = Math.min(canvas.width / W, canvas.height / H);

          drawBoard();

          /* Draw apples */
          ctx.fillStyle = '#ef4444';
          rd.apples?.forEach(([x, y]) => {
            ctx.beginPath();
            ctx.arc(x * s + s / 2, (H - 1 - y) * s + s / 2, s / 3, 0, Math.PI * 2);
            ctx.fill();
          });

          /* Draw snakes */
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
        /** Escape HTML special characters */
        function escapeHTML(str) {
          return str.replace(/[&<>"']/g, ch => (
            { '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":"&#39;" }[ch]
          ));
        }

        /* Precompile regexes for performance */
        const thinkRegex = /<think>([\s\S]*?)<\/think>/i;
        const bestRegex  = /<best_answer>([\s\S]*?)<\/best_answer>/i;
        const worstRegex = /<worst_answer>([\s\S]*?)<\/worst_answer>/i;

        /**
         * Build an array of HTML snippets for this player's rationales,
         * with a <details> toggle for “Thought” content.
         */
        function thoughtLines(rd, pid) {
          const snippets = [];

          /* If eliminated, show elimination info */
          if (!rd.alive?.[pid]) {
            snippets.push(`<p>Round ${rd.round_number}: ELIMINATED</p>`);
            return snippets;
          }

          const mv = rd.move_history?.at(-1)?.[pid];
          if (mv && mv.rationale) {
            const txt = mv.rationale;

            const mThink = thinkRegex.exec(txt);
            const mBest  = bestRegex.exec(txt);
            const mWorst = worstRegex.exec(txt);

            if (mThink && mThink[1]) {
              const content = mThink[1]
                .trim()
                .split('\n')
                .map(line => escapeHTML(line))
                .join('<br>');

              /* Insert a proper <details> / <summary> block */
              snippets.push(`
                <details class="toggle-thought mb-1"${pid === '1' ? ' open' : ''}>
                  <summary class="font-mono text-xs cursor-pointer flex items-center gap-1">
                    <svg class="toggle-arrow w-3 h-3 stroke-current text-gray-600 transition-transform" 
                         fill="none" stroke="currentColor" viewBox="0 0 12 12">
                      <path d="M3 4.5l3 3 3-3" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                    <span>Thought:</span>
                  </summary>
                  <div class="pl-4 text-xs text-gray-700 mt-1">${content}</div>
                </details>
              `);
            }

            if (mBest && mBest[1]) {
              const content = escapeHTML(mBest[1].trim());
              snippets.push(`<p class="font-mono text-xs">Best move: <b>${content}</b></p>`);
            }
            if (mWorst && mWorst[1]) {
              const content = escapeHTML(mWorst[1].trim());
              snippets.push(`<p class="font-mono text-xs">Worst move: <b>${content}</b></p>`);
            }
          }

          return snippets;
        }

        /* ======================= RENDER LOOP ======================= */
        function render() {
          if (!gameJsonData) return;

          const rd = gameJsonData.rounds[currentRound];

          document.getElementById('roundInfo').textContent =
            `Round ${currentRound}/${maxRounds - 1}`;
          document.getElementById('progressBar').value = currentRound;

          ['1', '2'].forEach(pid => {
            // Save current expanded state before update
            const thoughtEl = document.getElementById(`player${pid}Thoughts`);
            const wasOpen = thoughtEl.querySelector('details')?.open ?? false;

            /* Update alive / eliminated status */
            const alive = rd.alive?.[pid];
            const st    = document.getElementById(`player${pid}Status`);
            st.textContent = alive ? 'ALIVE' : 'ELIMINATED';
            st.className   = `${alive ? 'text-green-500' : 'text-red-500'} font-bold text-sm`;

            /* Populate thoughts panel via innerHTML */
            thoughtEl.innerHTML = thoughtLines(rd, pid).join('');

            // Restore expand/collapse state after update
            const newDetails = thoughtEl.querySelector('details');
            if (newDetails) {
              newDetails.open = pid === '1' ? true : false;
            }
          });

          /* Toggle play/pause or replay label */
          const playBtn = document.getElementById('playBtn');
          if (!playing && currentRound === maxRounds - 1) {
            playBtn.textContent = '🔄 Replay';
          } else {
            playBtn.textContent = playing ? '⏸️ Pause' : '▶️ Play';
          }

          drawRound();
        }

        /* ======================= CONTROLS ========================== */
        const on = (id, fn) => document.getElementById(id).addEventListener('click', fn);

        on('playBtn', () => {
          if (!gameJsonData) return;
          if (!playing && currentRound === maxRounds - 1) {
            currentRound = 0;
            playing = true;
            render();
          } else {
            playing = !playing;
            render();
          }
        });
        on('prevBtn', () => {
          if (!gameJsonData) return;
          currentRound = Math.max(currentRound - 1, 0);
          render();
        });
        on('nextBtn', () => {
          if (!gameJsonData) return;
          currentRound = Math.min(currentRound + 1, maxRounds - 1);
          render();
        });
        on('endBtn', () => {
          if (!gameJsonData) return;
          currentRound = maxRounds - 1;
          playing = false;
          render();
        });

        document.getElementById('progressBar').addEventListener('input', (e) => {
          if (!gameJsonData) return;
          currentRound = +e.target.value;
          render();
        });

        /* Advance the “playing” loop based on FPS */
        setInterval(() => {
          if (playing && gameJsonData) {
            if (currentRound < maxRounds - 1) {
              currentRound++;
            } else {
              playing = false;
            }
            render();
          }
        }, 1000 / FPS);

        /* Draw a blank background immediately, before JSON loads */
        ctx.fillStyle = '#f3f4f6';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
})();
