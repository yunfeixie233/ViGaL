/*  snake-game.js  */
/* Snake Game Component with JSON Support – robust fetch & parse */
(function () {
    /**
     * CONFIGURATION
     * ------------------------------------------------------------------
     * 1. Place your game‑log JSON under /data or supply a full URL.
     * 2. <div id="snake-game-container" data-json="/static/games/game‑42.json"></div>
     */
    const DEFAULT_JSON = './resources/game/o3-mini.json';
  
    /* ──────────────────────────────────────────────────────────────── */
    let gameJsonData = null;
  
    document.addEventListener('DOMContentLoaded', () => {
      const container = document.getElementById('snake-game-container');
      if (!container) return;
      container.innerHTML = markup();                      // inject template
      const path = container.getAttribute('data-json') || DEFAULT_JSON;
      boot(path);                                          // start!
    });
  
    /* ========================= TEMPLATE ============================ */
    function markup() {
      return `
        <div class="bg-gray-100 py-6 rounded-lg">
          <!-- Status line -->
          <div class="bg-white shadow rounded-lg p-4 mb-4 text-center">
            <span id="fileStatus" class="font-mono text-sm text-gray-600">Loading…</span>
          </div>
  
          <!-- Grid -->
          <div class="grid grid-cols-1 lg:grid-cols-[1fr_min-content_1fr] gap-4 mb-6">
            <!-- Player 1 thoughts / status -->
            <div class="thoughts-panel player1-border flex flex-col">
              <div class="p-3 border-b border-gray-100 sticky top-0 z-10 text-center player1-bg bg-opacity-50">
                <h2 id="player1Name" class="font-mono text-sm">Player 1</h2>
              </div>
              <div class="flex-1 p-4 overflow-auto">
                <div id="player1Thoughts" class="font-mono text-xs space-y-1"></div>
              </div>
              <div class="p-3 border-t border-gray-100 flex justify-between">
                <span id="player1Status" class="text-green-500 font-bold text-sm">WAITING</span>
                <span class="flex gap-1 font-mono text-sm">
                  <span id="player1Score" class="text-gray-400">0</span>
                  <span id="player1Apples"></span>
                </span>
              </div>
            </div>
  
            <!-- Canvas -->
            <div class="flex justify-center">
              <canvas id="gameCanvas" width="400" height="400" class="game-canvas"></canvas>
            </div>
  
            <!-- Player 2 thoughts / status -->
            <div class="thoughts-panel player2-border flex flex-col">
              <div class="p-3 border-b border-gray-100 sticky top-0 z-10 text-center player2-bg bg-opacity-50">
                <h2 id="player2Name" class="font-mono text-sm">Player 2</h2>
              </div>
              <div class="flex-1 p-4 overflow-auto">
                <div id="player2Thoughts" class="font-mono text-xs space-y-1"></div>
              </div>
              <div class="p-3 border-t border-gray-100 flex justify-between">
                <span id="player2Status" class="text-red-500 font-bold text-sm">WAITING</span>
                <span class="flex gap-1 font-mono text-sm">
                  <span id="player2Apples"></span>
                  <span id="player2Score" class="text-gray-400">0</span>
                </span>
              </div>
            </div>
          </div>
  
          <!-- Controls -->
          <div class="bg-white shadow rounded-lg p-4 mb-4 text-center">
            <div class="flex justify-center gap-4">
              <button id="playBtn" class="bg-blue-500 text-white px-4 py-2 rounded font-mono text-sm" disabled>⏸️ Pause</button>
              <button id="prevBtn" class="bg-gray-500 text-white px-4 py-2 rounded font-mono text-sm" disabled>⏮️ Prev</button>
              <span   id="roundInfo" class="font-mono text-sm text-gray-600">Round 0/0</span>
              <button id="nextBtn" class="bg-gray-500 text-white px-4 py-2 rounded font-mono text-sm" disabled>⏭️ Next</button>
              <button id="endBtn"  class="bg-gray-500 text-white px-4 py-2 rounded font-mono text-sm" disabled>⏩ End</button>
            </div>
            <input id="progressBar" type="range" min="0" max="100" value="0"
                   class="w-full max-w-md mt-2" disabled>
            <div class="mt-2">
              <span id="gameInfo" class="font-mono text-xs text-gray-500">Waiting…</span>
            </div>
          </div>
        </div>`;
    }
  
    /* ============================ BOOT ============================= */
    function boot(jsonPath) {
      const fileStatus  = document.getElementById('fileStatus');
      const absoluteURL = new URL(jsonPath, window.location.href).toString();
  
      fetch(absoluteURL, { cache: 'no-cache' })
        .then(async (res) => {
          if (!res.ok) throw new Error(`HTTP ${res.status} while fetching ${absoluteURL}`);
          const raw = await res.text();
          const clean = raw.replace(/^\uFEFF/, '').trim();   // strip BOM
          return JSON.parse(clean);
        })
        .then((json) => {
          gameJsonData = json;
          fileStatus.textContent = `Loaded ${absoluteURL}`;
          fileStatus.className   = 'font-mono text-sm text-green-600';
          initGame();
        })
        .catch((err) => {
          console.error(err);
          fileStatus.textContent = `Error: ${err.message}`;
          fileStatus.className   = 'font-mono text-sm text-red-600';
        });
  
      /* ---------- local state ---------- */
      let maxRounds = 0,
          currentRound = 0,
          playing = false,
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
  
        document.getElementById('player1Name').textContent =
          gameJsonData.metadata?.models?.['1'] || 'Player 1';
        document.getElementById('player2Name').textContent =
          gameJsonData.metadata?.models?.['2'] || 'Player 2';
  
        document.getElementById('gameInfo').textContent =
          `Game ID: ${gameJsonData.metadata?.game_id ?? 'N/A'} ` +
          `| Time: ${(gameJsonData.metadata?.time_taken ?? 0).toFixed(1)} s`;
  
        const pb = document.getElementById('progressBar');
        pb.max      = maxRounds - 1;
        pb.disabled = false;
  
        ['playBtn', 'prevBtn', 'nextBtn', 'endBtn']
          .forEach(id => document.getElementById(id).disabled = false);
  
        render();    // first render
      }
  
      /* ======================= DRAW HELPERS ====================== */
      function drawBoard() {
        const s = Math.min(canvas.width / W, canvas.height / H);
        ctx.fillStyle = '#f9fafb';
        ctx.fillRect(0, 0, W * s, H * s);
        ctx.strokeStyle = '#e5e7eb';
        for (let i = 0; i <= W; i++) {
          ctx.beginPath(); ctx.moveTo(i * s, 0);     ctx.lineTo(i * s, H * s); ctx.stroke();
        }
        for (let j = 0; j <= H; j++) {
          ctx.beginPath(); ctx.moveTo(0, j * s);     ctx.lineTo(W * s, j * s); ctx.stroke();
        }
      }
  
      function drawRound() {
        const rd = gameJsonData.rounds[currentRound];
        const s  = Math.min(canvas.width / W, canvas.height / H);
  
        drawBoard();
  
        /* Apples */
        ctx.fillStyle = '#ef4444';
        rd.apples?.forEach(([x, y]) => {
          ctx.beginPath();
          ctx.arc(x * s + s / 2, (H - 1 - y) * s + s / 2, s / 3, 0, 2 * Math.PI);
          ctx.fill();
        });
  
        /* Snakes */
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
      const esc = (str) =>
        str.replace(/[&<>"']/g, ch => (
          { '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' }[ch]
        ));
  
      /** Extract <think>, <best_answer>, <worst_answer> from rationale */
      function thoughtLines(rd, pid) {
        const lines = [];
  
        /* If player is dead this round, show elimination message only */
        if (!rd.alive?.[pid]) {
          lines.push(`Round ${rd.round_number}: ELIMINATED`);
          lines.push(`Score: ${rd.scores[pid]}`);
          return lines;
        }
  
        const mv = (rd.move_history || []).at(-1)?.[pid];   // last decision
        if (mv && mv.rationale) {
          const txt = mv.rationale;
          const extract = (tag) => {
            const m = txt.match(new RegExp(`<${tag}>([\\s\\S]*?)<\\/${tag}>`, 'i'));
            return m ? m[1].trim() : '';
          };
  
          const tk   = extract('think');
          const best = extract('best_answer');
          const worst= extract('worst_answer');
  
          if (tk)   lines.push(`Thought:<br>${esc(tk).replace(/\n/g,'<br>')}`);
          if (best) lines.push(`Best move: <b>${esc(best)}</b>`);
          if (worst)lines.push(`Worst move: <b>${esc(worst)}</b>`);
        }
  
        lines.push(`Score: ${rd.scores[pid]}`);
        lines.push(`Snake length: ${rd.snake_positions?.[pid]?.length ?? 0}`);
        return lines;
      }
  
      /* ======================= RENDER LOOP ======================= */
      function render() {
        if (!gameJsonData) return;
  
        const rd = gameJsonData.rounds[currentRound];
  
        document.getElementById('roundInfo').textContent =
          `Round ${currentRound}/${maxRounds - 1}`;
        document.getElementById('progressBar').value = currentRound;
  
        ['1', '2'].forEach(pid => {
          /* Score & apples */
          const score = rd.scores?.[pid] ?? 0;
          document.getElementById(`player${pid}Score`).textContent = score;
          document.getElementById(`player${pid}Apples`).textContent = '🍎'.repeat(score);
  
          /* Alive / dead lamp */
          const alive = rd.alive?.[pid];
          const st    = document.getElementById(`player${pid}Status`);
          st.textContent = alive ? 'ALIVE' : 'ELIMINATED';
          st.className   = `${alive ? 'text-green-500' : 'text-red-500'} font-bold text-sm`;
  
          /* Thoughts panel */
          document.getElementById(`player${pid}Thoughts`).innerHTML =
            thoughtLines(rd, pid).map(t => `<p>${t}</p>`).join('');
        });
  
        /* toggle play/pause label */
        document.getElementById('playBtn').textContent = playing ? '⏸️ Pause' : '▶️ Play';
  
        drawRound();
      }
  
      /* ======================= CONTROLS ========================== */
      const on = (id, fn) => document.getElementById(id).addEventListener('click', fn);
  
      on('playBtn', () => { if (!gameJsonData) return; playing = !playing; render(); });
      on('prevBtn', () => { if (!gameJsonData) return;
        currentRound = Math.max(currentRound - 1, 0); render(); });
      on('nextBtn', () => { if (!gameJsonData) return;
        currentRound = Math.min(currentRound + 1, maxRounds - 1); render(); });
      on('endBtn',  () => { if (!gameJsonData) return;
        currentRound = maxRounds - 1; playing = false; render(); });
  
      document.getElementById('progressBar').addEventListener('input', (e) => {
        if (!gameJsonData) return;
        currentRound = +e.target.value; render();
      });
  
      /* Main loop when “playing” */
      setInterval(() => {
        if (playing && gameJsonData) {
          if (currentRound < maxRounds - 1) currentRound++;
          else playing = false;
          render();
        }
      }, 1000);
  
      /* Draw blank board immediately (before JSON loads) */
      ctx.fillStyle = '#f3f4f6';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
  })();
  