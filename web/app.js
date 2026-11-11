"use strict";

// ------------------------------------------------------------------ helpers
const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x) => (x == null ? "–" : `${Math.round(x * 100)}%`);

async function api(path, body) {
  const opt = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch(path, opt);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || r.statusText);
  return data;
}

const GLYPH = { k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟" };
const FILES = "abcdefgh";

function parseFen(fen) {
  const rows = fen.split(" ")[0].split("/");
  const pos = {};
  rows.forEach((row, ri) => {
    let f = 0;
    for (const ch of row) {
      if (/\d/.test(ch)) { f += +ch; continue; }
      pos[FILES[f] + (8 - ri)] = ch;
      f++;
    }
  });
  return pos;
}

// -------------------------------------------------------------------- board
class Board {
  constructor(el, onMove) {
    this.el = el;
    this.onMove = onMove;
    this.state = { fen: "8/8/8/8/8/8/8/8 w - - 0 1", orientation: "white", legal: [], arrows: [], heat: null, last: null, interactive: true };
    this.sel = null;
    el.addEventListener("click", (e) => {
      const sq = e.target.closest(".sq");
      if (sq) this.click(sq.dataset.sq);
    });
  }
  set(patch) { Object.assign(this.state, patch); if ("fen" in patch) this.sel = null; this.render(); }
  click(sq) {
    const { legal, interactive } = this.state;
    if (!interactive) return;
    if (this.sel) {
      const mv = legal.find((u) => u.slice(0, 2) === this.sel && u.slice(2, 4) === sq && (u.length === 4 || u[4] === "q"));
      if (mv) { this.sel = null; this.onMove(mv); return; }
    }
    this.sel = legal.some((u) => u.slice(0, 2) === sq) ? sq : null;
    this.render();
  }
  squares() {
    const out = [];
    const ranks = this.state.orientation === "white" ? [8, 7, 6, 5, 4, 3, 2, 1] : [1, 2, 3, 4, 5, 6, 7, 8];
    const files = this.state.orientation === "white" ? [...FILES] : [...FILES].reverse();
    for (const r of ranks) for (const f of files) out.push(f + r);
    return out;
  }
  center(sq) {
    const list = this.squares();
    const i = list.indexOf(sq);
    return [((i % 8) + 0.5) * 12.5, (Math.floor(i / 8) + 0.5) * 12.5];
  }
  render() {
    const { fen, legal, heat, last, arrows } = this.state;
    const pos = parseFen(fen);
    const targets = this.sel ? new Set(legal.filter((u) => u.startsWith(this.sel)).map((u) => u.slice(2, 4))) : new Set();
    const lastSq = last ? [last.slice(0, 2), last.slice(2, 4)] : [];
    const list = this.squares();
    let html = "";
    list.forEach((sq, i) => {
      const fi = FILES.indexOf(sq[0]);
      const dark = (fi + +sq[1]) % 2 === 0;
      const cls = ["sq", dark ? "d" : "l"];
      if (lastSq.includes(sq)) cls.push("last");
      if (this.sel === sq) cls.push("sel");
      if (targets.has(sq)) cls.push("target");
      const p = pos[sq];
      const h = heat && heat[sq] ? `<span class="heat" style="background: rgba(var(--heat), ${(0.15 + 0.55 * heat[sq]).toFixed(2)})"></span>` : "";
      const piece = p ? `<span class="pc ${p === p.toUpperCase() ? "w" : "b"}">${GLYPH[p.toLowerCase()]}</span>` : "";
      const coordF = i >= 56 ? `<span class="coord f">${sq[0]}</span>` : "";
      const coordR = i % 8 === 0 ? `<span class="coord r">${sq[1]}</span>` : "";
      html += `<div class="${cls.join(" ")}" data-sq="${sq}">${h}${piece}${coordF}${coordR}</div>`;
    });
    let svg = "";
    if (arrows && arrows.length) {
      svg = `<svg class="arrows" viewBox="0 0 100 100"><defs>`;
      arrows.forEach((a, i) => { svg += `<marker id="ah${i}" markerWidth="4" markerHeight="4" refX="2" refY="2" orient="auto"><path d="M0,0 L4,2 L0,4 z" fill="${a.color}"/></marker>`; });
      svg += `</defs>`;
      arrows.forEach((a, i) => {
        const [x1, y1] = this.center(a.uci.slice(0, 2));
        const [x2, y2] = this.center(a.uci.slice(2, 4));
        const dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy) || 1;
        const ex = x2 - (dx / L) * 3.5, ey = y2 - (dy / L) * 3.5;
        svg += `<line x1="${x1}" y1="${y1}" x2="${ex}" y2="${ey}" stroke="${a.color}" stroke-width="${a.width || 2}" stroke-linecap="round" opacity="${a.opacity || 0.85}" marker-end="url(#ah${i})"/>`;
      });
      svg += `</svg>`;
    }
    this.el.innerHTML = html + svg;
  }
}

const COLORS = { actual: "#1f8a4c", model: "#2f6fde", guess: "#e07b18", threat: "#c2412d" };

// -------------------------------------------------------------------- views
let players = [];
let currentPlayer = null;

function showView(name) {
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${name}`));
  document.querySelectorAll("nav button").forEach((b) => b.classList.toggle("active", b.dataset.view === name));
  if (name === "analyse") analyse.refresh();
}
document.querySelectorAll("nav button").forEach((b) => b.addEventListener("click", () => showView(b.dataset.view)));

async function loadPlayers(selectId) {
  players = await api("/api/players");
  const list = $("#player-list");
  list.innerHTML = players.length
    ? players.map((p) => `<li data-id="${esc(p.id)}"><div class="name">${esc(p.name)}</div><small>${p.games} games · model hits ${pct(p.top1)} of held-out moves</small></li>`).join("")
    : `<li class="muted">No players yet - add one below.</li>`;
  list.querySelectorAll("li[data-id]").forEach((li) => li.addEventListener("click", () => selectPlayer(li.dataset.id)));
  const opts = players.map((p) => `<option value="${esc(p.id)}">${esc(p.name)}</option>`).join("");
  $("#play-player").innerHTML = opts;
  $("#an-player").innerHTML = opts;
  const id = selectId || currentPlayer || (players[0] && players[0].id);
  if (id) selectPlayer(id);
}

function barRow(label, value, max, text, cls = "") {
  const w = Math.max(0, Math.min(100, (value / max) * 100));
  return `<div class="bar-row"><span>${label}</span><div class="bar"><i class="${cls}" style="width:${w}%"></i></div><span class="num">${text}</span></div>`;
}
function divRow(label, delta, text) {
  // diverging bar around the centre, delta in log-odds
  const w = Math.min(50, Math.abs(delta) * 25);
  const style = delta >= 0 ? `left:50%;width:${w}%` : `left:${50 - w}%;width:${w}%`;
  return `<div class="bar-row"><span>${label}</span><div class="bar div"><i class="${delta < 0 ? "neg" : ""}" style="${style}"></i></div><span class="num">${text}</span></div>`;
}

async function selectPlayer(id) {
  currentPlayer = id;
  document.querySelectorAll("#player-list li").forEach((li) => li.classList.toggle("sel", li.dataset.id === id));
  $("#play-player").value = id;
  $("#an-player").value = id;
  loadGames();
  const p = await api(`/api/players/${id}`);
  const ev = p.evaluation || {};
  const acc = [
    ["This player's model", ev.personal_with_book],
    ["…without opening book", ev.personal],
    ["Average-player model", ev.population],
    ["Stockfish top move", ev.engine_top_move],
    ["Random legal move", ev.random],
  ].filter(([, v]) => v);
  const bs = p.blind_spots || {};
  const rows = (bs.rows || []).filter((r) => r.index >= 1.15).slice(0, 8);
  const seen = new Set();
  const strengths = (bs.rows || []).filter((r) => r.index <= 0.8).reverse()
    .filter((r) => !seen.has(r.motif) && seen.add(r.motif)).slice(0, 4);
  const style = (p.style || []).filter((s) => Math.abs(s.delta) >= 0.15).slice(0, 10);
  const st = p.stats || {};
  $("#profile").innerHTML = `
    <div class="card">
      <h2>${esc(p.name)}</h2>
      <p class="muted">${st.games} games (${st.white_games} as White, ${st.black_games} as Black) · ${st.decisions} decisions analysed${st.avg_loss_pawns != null ? ` · engine match ${pct(st.engine_match)} · blunder rate ${pct(st.blunder_rate)}` : ""}</p>
      <button class="primary" id="play-this">Predict ${esc(p.name)}'s moves →</button>
    </div>
    <div class="card">
      <h3>How well can we predict this player?</h3>
      <p class="muted">Top-1 accuracy on ${p.heldout_games} held-out games the model never saw.</p>
      ${acc.map(([n, v]) => barRow(n, v.top1, 1, `${pct(v.top1)}${v.top3 != null ? ` <small>top3 ${pct(v.top3)}</small>` : ""}`)).join("")}
    </div>
    <div class="card">
      <h3>Blind spots</h3>
      <p class="muted">Index = how much more often they miss this kind of move than their own average (1.0 = average for them). Threats: they don't react. Walk-ins: they allow it. Opportunities: they don't play it.</p>
      ${rows.length ? rows.map((r) => barRow(esc(r.label), r.index, 3, `×${r.index.toFixed(2)} <small>${r.missed}/${r.events}</small>`, "bad") + (r.examples || []).slice(0, 1).map((e) => `<div class="example">e.g. move ${e.move_number}: played ${esc(e.played)}, ${esc(e.note)}</div>`).join("")).join("") : `<p class="muted">No clear blind spot yet (needs more games).</p>`}
      ${strengths.length ? `<h4>Sharp at</h4>${strengths.map((r) => `<span class="tag">${esc(r.label.replace(/^(Overlooks threats from|Misses winning|Walks into) /, ""))} ×${r.index.toFixed(2)}</span>`).join("")}` : ""}
      ${(p.refutations && p.refutations.rows && p.refutations.rows.length) ? `<h4>What punishes their blunders (engine)</h4>${p.refutations.rows.slice(0, 4).map((r) => barRow(esc(r.label), r.lift, 3, `×${r.lift.toFixed(2)}`, r.lift > 1.2 ? "bad" : "")).join("")}` : ""}
    </div>
    <div class="card">
      <h3>Style fingerprint</h3>
      <p class="muted">How much more (→) or less (←) likely than an average player they are to pick a move with this property.</p>
      ${style.map((s) => divRow(esc(s.label), s.delta, `×${s.odds.toFixed(2)}`)).join("") || `<p class="muted">Plays like the average player.</p>`}
    </div>`;
  $("#play-this").addEventListener("click", () => { showView("play"); });
}

$("#play-player").addEventListener("change", (e) => selectPlayer(e.target.value));
$("#an-player").addEventListener("change", (e) => { currentPlayer = e.target.value; analyse.refresh(); });

async function loadGames() {
  if (!currentPlayer) return;
  const games = await api(`/api/players/${currentPlayer}/games`);
  $("#play-game").innerHTML = `<option value="">Random held-out game</option>` + games.map((g) => `<option value="${g.index}">${esc(g.white)} – ${esc(g.black)} (${esc(g.result)}, ${Math.ceil(g.plies / 2)} moves)</option>`).join("");
  const p = players.find((x) => x.id === currentPlayer);
  $("#intro-player").textContent = p ? p.name : "…";
}

// --------------------------------------------------------------- building
$("#pgn-file").addEventListener("change", async (e) => {
  const f = e.target.files[0];
  if (f) $("#pgn-text").value = await f.text();
});
$("#build-btn").addEventListener("click", async () => {
  const body = { lichess: $("#lichess-user").value.trim() || null, chesscom: $("#chesscom-user").value.trim() || null, pgn: $("#pgn-text").value.trim() || null, player: $("#pgn-player").value.trim() || null };
  const status = $("#build-status");
  try {
    let job = await api("/api/players", body);
    $("#build-btn").disabled = true;
    while (job.status !== "done" && job.status !== "error") {
      status.textContent = job.status;
      await new Promise((r) => setTimeout(r, 1000));
      job = await api(`/api/jobs/${job.id}`);
    }
    if (job.status === "error") throw new Error(job.error);
    status.textContent = "Profile ready.";
    await loadPlayers(job.player_id);
  } catch (err) {
    status.textContent = `Error: ${err.message}`;
  } finally {
    $("#build-btn").disabled = false;
  }
});

// -------------------------------------------------------------------- play
const play = {
  board: null, session: null, guess: null, revealed: false,
  init() {
    this.board = new Board($("#play-board"), (uci) => this.pick(uci));
    this.board.set({ fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", interactive: false });
    $("#start-btn").addEventListener("click", () => this.start());
    $("#new-game-btn").addEventListener("click", () => { $("#play-intro").classList.remove("hidden"); });
    $("#submit-guess").addEventListener("click", () => this.submit(this.guess));
    $("#skip-btn").addEventListener("click", () => this.submit(null));
    $("#hint-btn").addEventListener("click", () => this.hint());
  },
  async start() {
    const gi = $("#play-game").value;
    try {
      const s = await api("/api/game", { player_id: $("#play-player").value, game_index: gi === "" ? null : +gi });
      $("#play-intro").classList.add("hidden");
      $("#reveal").classList.add("hidden");
      $("#hint").classList.add("hidden");
      this.apply(s);
    } catch (e) { alert(e.message); }
  },
  apply(s) {
    this.session = s;
    this.guess = null;
    $("#score-user").textContent = s.user_score;
    $("#score-model").textContent = s.model_score;
    $("#hits-user").textContent = s.rounds_played ? `${s.user_hits}/${s.rounds_played}` : "";
    $("#hits-model").textContent = s.rounds_played ? `${s.model_hits}/${s.rounds_played}` : "";
    $("#game-title").textContent = `${s.white} – ${s.black}`;
    this.board.set({ fen: s.fen, orientation: s.player_color, legal: s.legal, last: s.last_move, interactive: !s.finished, heat: null, arrows: this.prevArrows || [] });
    $("#history").innerHTML = s.history.map((m, i) => (i % 2 === 0 ? `<b>${i / 2 + 1}.</b> ` : "") + esc(m)).join(" ");
    $("#submit-guess").disabled = true;
    $("#hint-btn").disabled = s.finished;
    $("#skip-btn").disabled = s.finished;
    $("#guess-label").textContent = s.finished ? `Game over (${s.result}). You ${s.user_score >= s.model_score ? "beat" : "lost to"} the model ${s.user_score}–${s.model_score}.` : `${s.player} to move (${s.player_color}). What did they play?`;
  },
  pick(uci) {
    this.guess = uci;
    this.board.set({ arrows: [{ uci, color: COLORS.guess, width: 2.4 }] });
    $("#submit-guess").disabled = false;
    $("#guess-label").textContent = `Your guess: ${uci}. Lock it in, or pick another move.`;
  },
  async submit(uci) {
    if (!this.session) return;
    $("#submit-guess").disabled = true;
    const before = this.session;
    const r = await api(`/api/game/${this.session.id}/guess`, { uci });
    $("#hint").classList.add("hidden");
    this.renderReveal(r, before);
    this.prevArrows = [
      { uci: r.round.actual, color: COLORS.actual, width: 2.6 },
      ...(r.round.model_top !== r.round.actual ? [{ uci: r.round.model_top, color: COLORS.model, width: 1.8, opacity: 0.7 }] : []),
      ...(r.round.guess && r.round.guess !== r.round.actual ? [{ uci: r.round.guess, color: COLORS.guess, width: 1.6, opacity: 0.7 }] : []),
    ];
    this.apply(r.state);
    this.prevArrows = [];
  },
  async hint() {
    if (!this.session) return;
    const r = await api(`/api/game/${this.session.id}/hint`, {});
    $("#score-user").textContent = r.state.user_score;
    const h = r.hint;
    this.board.set({ heat: h.attention });
    $("#hint").classList.remove("hidden");
    $("#hint").innerHTML = `<h3>Through their eyes</h3><ul class="notes">${h.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>
      <p class="muted">Blue squares: where their attention most likely is.</p>`;
  },
  renderReveal(r, before) {
    const rd = r.round;
    const pred = r.prediction;
    const per = pred.perspective;
    const youOk = rd.guess === rd.actual;
    const modelOk = rd.model_top === rd.actual;
    const el = $("#reveal");
    el.classList.remove("hidden");
    el.innerHTML = `
      <div class="verdict">${esc(before.player)} played <b>${esc(rd.actual_san)}</b></div>
      <div>You: ${rd.guess_san ? `${esc(rd.guess_san)} <span class="${youOk ? "ok" : "no"}">${youOk ? "✓ +10" : "✗"}</span> <small>(model gave it ${pct(rd.p_guess)})</small>` : `<span class="muted">skipped</span>`}</div>
      <div>Model: ${esc(rd.model_top_san)} <span class="${modelOk ? "ok" : "no"}">${modelOk ? "✓ +10" : "✗"}</span> <small>(gave the real move ${pct(rd.p_actual)})</small></div>
      <div class="legend" style="margin-top:6px"><span><i style="background:${COLORS.actual}"></i>played</span><span><i style="background:${COLORS.model}"></i>model</span><span><i style="background:${COLORS.guess}"></i>you</span></div>
      <h4>Model's prediction</h4>
      ${movesList(pred, rd.actual)}
      ${perspectiveHtml(per, pred, r.reality)}`;
  },
};

function movesList(pred, actual) {
  return `<ul class="moves">${pred.moves.map((m) => `<li><span class="san">${esc(m.san)}${m.uci === actual ? " ✓" : ""}</span> ${barInline(m.p)} <span class="num">${pct(m.p)}</span>
      ${m.book ? `<span class="tag">played ${m.book}× here before</span>` : ""}
      <div class="why">${m.reasons.map((x) => `${x.effect > 0 ? "+" : "−"} ${esc(x.label)}${Math.abs(x.personal) >= 0.2 ? ` <b>(${x.personal > 0 ? "especially" : "unusually rarely"} for this player)</b>` : ""}`).join(" · ")}</div></li>`).join("")}</ul>
    ${pred.engine_best ? `<p class="muted">Engine's best: <b>${esc(pred.engine_best.san)}</b> - this player finds it ${pct(pred.engine_best.p)} (rank ${pred.engine_best.rank} in their eyes).</p>` : ""}`;
}
function barInline(p) { return `<span class="bar" style="display:inline-block;width:90px;vertical-align:middle"><i style="width:${Math.round(p * 100)}%"></i></span>`; }

function perspectiveHtml(per, pred, reality) {
  if (!per) return "";
  const tag = (t) => `<span class="tag ${t.blind_index >= 1.15 ? "hot" : ""}">${t.blind_label && t.blind_index >= 1.15 ? `blind spot: ${esc(t.blind_label)}` : esc(t.motifs.slice(0, 3).join(", "))}</span>`;
  const real = (key, mv) => {
    if (!reality) return "";
    const x = reality.find((r) => r[key] === mv);
    if (!x) return "";
    const v = key === "threat" ? x.player_reacted : x.player_found;
    return `<span class="${v ? "ok" : "no"}">${key === "threat" ? (v ? "reacted" : "ignored it") : (v ? "found it" : "missed it")}</span>`;
  };
  return `
    <h4>Through the player's eyes</h4>
    <ul class="notes">${per.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>
    ${per.threats.length ? `<h4>Threats against them</h4>${per.threats.map((t) => `<div>${esc(t.move)} → ${esc(t.piece)} ${esc(t.target)} · notices ${pct(t.p_notice)} ${tag(t)} ${real("threat", t.move)}</div>`).join("")}` : ""}
    ${per.chances.length ? `<h4>Chances they have</h4>${per.chances.map((c) => `<div>${esc(c.move)} (${c.kind === "mate" ? "mate" : `+${c.gain}`}) · finds ${pct(c.p_find)} ${tag(c)} ${real("chance", c.move)}</div>`).join("")}` : ""}
    ${per.risky_moves.length ? `<h4>Likely moves that backfire</h4>${per.risky_moves.map((m) => `<div>${esc(m.move)} (${pct(m.p)}) loses ${m.loses} to ${esc(m.punished_by)} ${tag(m)}</div>`).join("")}` : ""}
    <h4>Intentions</h4>${per.plans.map((p) => barRow(esc(p.label), p.p, 1, pct(p.p))).join("")}
    ${per.evaluation ? `<p class="muted">Objective eval ${(per.evaluation.objective_cp / 100).toFixed(1)}; expected after their move ${(per.evaluation.expected_after_their_move_cp / 100).toFixed(1)}.</p>` : ""}`;
}

// ----------------------------------------------------------------- analyse
const analyse = {
  board: null, moves: [], startFen: null, orientation: "white", last: null,
  init() {
    this.board = new Board($("#an-board"), (u) => { this.moves.push(u); this.refresh(); });
    $("#an-undo").addEventListener("click", () => { this.moves.pop(); this.refresh(); });
    $("#an-reset").addEventListener("click", () => { this.moves = []; this.startFen = null; this.refresh(); });
    $("#an-flip").addEventListener("click", () => { this.orientation = this.orientation === "white" ? "black" : "white"; this.board.set({ orientation: this.orientation }); });
    $("#an-load").addEventListener("click", () => { this.startFen = $("#an-fen").value.trim() || null; this.moves = []; this.refresh(); });
    $("#an-heat").addEventListener("change", () => this.refresh());
  },
  async refresh() {
    let pos;
    try { pos = await api("/api/position", { fen: this.startFen, moves: this.moves }); }
    catch (e) { $("#an-out").innerHTML = `<p class="no">${esc(e.message)}</p>`; this.moves.pop(); return; }
    $("#an-fen").value = pos.fen;
    $("#an-history").innerHTML = pos.history.map((m, i) => (i % 2 === 0 ? `<b>${i / 2 + 1}.</b> ` : "") + esc(m)).join(" ");
    this.board.set({ fen: pos.fen, legal: pos.legal, last: pos.last_move, orientation: this.orientation, arrows: [], heat: null });
    const pid = $("#an-player").value;
    if (!pid || pos.game_over) { $("#an-out").innerHTML = pos.game_over ? "<p>Game over.</p>" : ""; return; }
    $("#an-out").innerHTML = `<p class="muted">Thinking like ${esc(pid)}…</p>`;
    const pred = await api("/api/predict", { player_id: pid, fen: this.startFen, moves: this.moves });
    const top = pred.moves.slice(0, 3);
    this.board.set({
      heat: $("#an-heat").checked ? pred.perspective.attention : null,
      arrows: top.map((m, i) => ({ uci: m.uci, color: COLORS.model, width: 2.6 - i * 0.7, opacity: 0.9 - i * 0.2 })),
    });
    const name = (players.find((p) => p.id === pid) || {}).name || pid;
    $("#an-out").innerHTML = `<h3>${esc(name)} as ${pred.turn}${pred.in_book ? ` <span class="tag">in their opening book</span>` : ""}</h3>${movesList(pred, null)}${perspectiveHtml(pred.perspective, pred, null)}`;
  },
};

// -------------------------------------------------------------------- boot
(async function boot() {
  play.init();
  analyse.init();
  try {
    const h = await api("/api/health");
    $("#engine-badge").textContent = h.engine ? "Stockfish: on" : "Stockfish: off (heuristics only)";
  } catch (e) { /* ignore */ }
  await loadPlayers();
})();
