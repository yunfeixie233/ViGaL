// snake-game.js – Snake & Rotation Game visualiser with size‑configurable carousels
(function () {
    /* ===================== CONFIG ===================== */
    const DEFAULT_JSON = './resources/game/o3-mini.json';
    // Carousel dimensions in pixels – tweak to adjust both carousels at once
    const CAROUSEL_W = 640;
    const CAROUSEL_H = 320;
  
    /* ===================== STATE ===================== */
    let gameJsonData = null;
  
    /* ===================== BOOTSTRAP ===================== */
    document.addEventListener('DOMContentLoaded', () => {
      const container = document.getElementById('snake-game-container');
      if (!container) return;
  
      container.innerHTML = markup();
  
      // Apply carousel dimensions
      document.querySelectorAll('.game-carousel').forEach((el) => {
        el.style.width = `${CAROUSEL_W}px`;
        el.style.height = `${CAROUSEL_H}px`;
      });
  
      const path = container.getAttribute('data-json') || DEFAULT_JSON;
      boot(path);
    });
  
    /* ===================== MARKUP ===================== */
    function markup() {
      return `
        <div class="bg-gray-100 py-6">
          <!-- Status line -->
          <div class="bg-white shadow rounded-lg p-4 mb-4 text-center">
            <span id="fileStatus" class="font-mono text-sm text-gray-600">Loading…</span>
          </div>
  
          <!-- Grid -->
          <div class="grid grid-cols-1 lg:grid-cols-[1fr_min-content_1fr] gap-4 mb-6">
            ${playerPanel('1', 'green')}
            <div class="flex justify-center"><canvas id="gameCanvas" width="400" height="400" class="game-canvas"></canvas></div>
            ${playerPanel('2', 'red')}
          </div>
  
          <!-- Controls -->
          <div class="bg-white shadow rounded-lg p-4 mb-4 text-center">
            <div class="flex justify-center gap-4">
              <button id="playBtn"  class="bg-blue-500 text-white px-4 py-2 rounded font-mono text-sm" disabled>⏸️ Pause</button>
              <button id="prevBtn"  class="bg-gray-500 text-white px-4 py-2 rounded font-mono text-sm" disabled>⏮️ Prev</button>
              <span   id="roundInfo" class="font-mono text-sm text-gray-600">Round 0/0</span>
              <button id="nextBtn"  class="bg-gray-500 text-white px-4 py-2 rounded font-mono text-sm" disabled>⏭️ Next</button>
              <button id="endBtn"   class="bg-gray-500 text-white px-4 py-2 rounded font-mono text-sm" disabled>⏩ End</button>
            </div>
            <input id="progressBar" type="range" min="0" max="100" value="0" class="w-full max-w-md mt-2" disabled>
            <div class="mt-2"><span id="gameInfo" class="font-mono text-xs text-gray-500">Waiting…</span></div>
          </div>
  
          <!-- Snake carousel -->
          <div class="mx-auto my-6"><div id="snakeCarousel" class="game-carousel overflow-hidden rounded-lg shadow bg-white flex items-center justify-center font-mono text-sm text-gray-500">Snake screenshots</div></div>
          <!-- Rotation carousel (clone) -->
          <div class="mx-auto my-6"><div id="rotationCarousel" class="game-carousel overflow-hidden rounded-lg shadow bg-white flex items-center justify-center font-mono text-sm text-gray-500">Rotation screenshots</div></div>
        </div>`;
    }
  
    function playerPanel(n, color) {
      return `
        <div class="thoughts-panel player${n}-border flex flex-col">
          <div class="p-3 border-b border-gray-100 sticky top-0 z-10 text-center bg-opacity-50 player${n}-bg"><h2 id="player${n}Name" class="font-mono text-sm">Player ${n}</h2></div>
          <div class="flex-1 p-4 overflow-auto"><div id="player${n}Thoughts" class="font-mono text-xs space-y-1"></div></div>
          <div class="p-3 border-t border-gray-100 flex justify-between text-sm font-mono">
            <span id="player${n}Status" class="text-${color}-500 font-bold">WAITING</span>
            <span class="flex gap-1"><span id="player${n}Apples"></span><span id="player${n}Score" class="text-gray-400">0</span></span>
          </div>
        </div>`;
    }
  
    /* ===================== BOOT (fetch & build) ===================== */
    function boot(jsonPath) {
      const fileStatus = document.getElementById('fileStatus');
      const absolutePath = new URL(jsonPath, window.location.href).toString();
  
      fetch(absolutePath, { cache: 'no-cache' })
        .then((r) => (r.ok ? r.text() : Promise.reject(`HTTP ${r.status}`)))
        .then((txt) => JSON.parse(txt.replace(/^\uFEFF/, '')))
        .then((json) => {
          gameJsonData = json;
          fileStatus.textContent = `Loaded ${absolutePath}`;
          fileStatus.className = 'font-mono text-sm text-green-600';
          initGame();
        })
        .catch((err) => {
          console.error(err);
          fileStatus.textContent = `Error: ${err}`;
          fileStatus.className = 'font-mono text-sm text-red-600';
        });
  
      /* —— local vars —— */
      let maxRounds = 0,
        currentRound = 0,
        playing = false,
        W = 10,
        H = 10;
  
      const ctx = /** @type {HTMLCanvasElement} */ (document.getElementById('gameCanvas')).getContext('2d');
  
      function initGame() {
        if (!Array.isArray(gameJsonData.rounds)) throw new Error('rounds array missing');
        maxRounds = gameJsonData.metadata?.actual_rounds || gameJsonData.rounds.length;
        ({ width: W = 10, height: H = 10 } = gameJsonData.rounds[0] || {});
  
        document.getElementById('player1Name').textContent = gameJsonData.metadata?.models?.['1'] || 'Player 1';
        document.getElementById('player2Name').textContent = gameJsonData.metadata?.models?.['2'] || 'Player 2';
        document.getElementById('gameInfo').textContent = `Game ID: ${gameJsonData.metadata?.game_id ?? 'N/A'} | Time: ${(gameJsonData.metadata?.time_taken ?? 0).toFixed(1)}s`;
  
        const pb = document.getElementById('progressBar');
        pb.max = maxRounds - 1;
        pb.disabled = false;
        ['playBtn', 'prevBtn', 'nextBtn', 'endBtn'].forEach((id) => (document.getElementById(id).disabled = false));
  
        render();
      }
  
      /* —— drawing —— */
      function board() {
        const cell = Math.min(400 / W, 400 / H);
        ctx.fillStyle = '#f9fafb';
        ctx.fillRect(0, 0, W * cell, H * cell);
        ctx.strokeStyle = '#e5e7eb';
        for (let i = 0; i <= W; i++) {
          ctx.beginPath();
          ctx.moveTo(i * cell, 0);
          ctx.lineTo(i * cell, H * cell);
          ctx.stroke();
        }
        for (let j = 0; j <= H; j++) {
          ctx.beginPath();
          ctx.moveTo(0, j * cell);
          ctx.lineTo(W * cell, j * cell);
          ctx.stroke();
        }
      }
      function draw() {
        board();
        const rd = gameJsonData.rounds[currentRound];
        const cell = Math.min(400 / W, 400 / H);
        ctx.fillStyle = '#ef4444';
        (rd.apples || []).forEach(([x, y]) => {
          ctx.beginPath();
          ctx.arc(x * cell + cell / 2, (H - 1 - y) * cell + cell / 2, cell / 3, 0, 2 * Math.PI);
          ctx.fill();
        });
        const colors = { '1': '#4F7022', '2': '#036C8E' };
        Object.entries(rd.snake_positions || {}).forEach(([pid, segs]) => {
          ctx.fillStyle = rd.alive?.[pid] ? colors[pid] : '#9ca3af';
          segs.forEach(([x, y], i) => {
            ctx.globalAlpha = i ? Math.max(0.3, 0.8 - i * 0.1) : 1;
            ctx.fillRect(x * cell + 2, (H - 1 - y) * cell + 2, cell - 4, cell - 4);
          });
          ctx.globalAlpha = 1;
        });
      }
  
      /* —— thoughts —— */
      const parseTag = (txt, tag) => {
        const m = txt.match(new RegExp(`<${tag}>([\s\S]*?)<\/${tag}>`));
        return m ? m[1].trim() : null;
      };
      const makeThoughts = (rd, pid) => {
        const out = [];
        if (!rd.alive?.[pid]) {
          out.push(`Round ${rd.round_number}: ELIMINATED`, `Score: ${rd.scores[pid]}`);
          return out;
        }
        const mv = rd.move_history?.at(-1)?.[pid];
        if (mv) {
          out.push(`Round ${rd.round_number}: Move ${mv.move}`);
          const r = mv.rationale || '';
          parseTag(r, 'think') && out.push(`Thought: ${parseTag(r, 'think')}`);
          parseTag(r, 'best_answer') && out.push(`Best move: ${parseTag(r, 'best_answer')}`);
          parseTag(r, 'worst_answer') && out.push(`Worst move: ${parseTag(r, 'worst_answer')}`);
        }
        out.push(`Score: ${rd.scores[pid]}`, `Length: ${(rd.snake_positions[pid] || []).length}`);
        return out;
      };
  
      /* —— render —— */
      function render() {
        if (!gameJsonData) return;
        const rd = gameJsonData.rounds[currentRound];
        document.getElementById('roundInfo').textContent = `Round ${currentRound}/${maxRounds - 1}`;
        document.getElementById('progressBar').value = currentRound;
        ['1', '2'].forEach((pid) => {
          document.getElementById(`player${pid}Score`).textContent = rd.scores[pid] || 0;
          document.getElementById(`player${pid}Apples`).textContent = '🍎'.repeat(rd.scores[pid] || 0);
          const stat = document.getElementById(`player${pid}Status`);
          const alive = rd.alive[pid];
          stat.textContent = alive ? 'ALIVE' : 'ELIMINATED';
          stat.className = `${alive ? 'text-green-500' : 'text-red-500'} font-bold`;
          document.getElementById(`player${pid}Thoughts`).innerHTML = makeThoughts(rd, pid)
            .map((l) => `<p>${l}</p>`) .join('');
        });
        document.getElementById('playBtn').textContent = playing ? '⏸️ Pause' : '▶️ Play';
        draw();
      }
  
      /* —— listeners —— */
      const on = (id, fn) => document.getElementById(id).addEventListener('click', fn);
      on('playBtn', () => ((playing = !playing), render()));
      on('prevBtn', () => ((currentRound = Math.max(currentRound - 1, 0)), render()));
      on('nextBtn', () => ((currentRound = Math.min(currentRound + 1, maxRounds - 1)), render()));
      on('endBtn', () => ((currentRound = maxRounds - 1), (playing = false), render()));
      document.getElementById('progressBar').addEventListener('input', (e) => {
        currentRound = +e.target.value;
        render();
      });
  
      setInterval(() => {
        if (!playing) return;
        currentRound < maxRounds - 1 ? currentRound++ : (playing = false);
        render();
      }, 1000);
  
      // blank board initially
      ctx.fillStyle = '#f3f4f6';
      ctx.fillRect(0, 0, 400, 400);
    }
  })();
  