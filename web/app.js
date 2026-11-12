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
      const h = heat && heat[sq] ? `<span class="heat" style="background: rgba(0, 0, 255, ${(0.08 + 0.42 * heat[sq]).toFixed(2)})"></span>` : "";
      const piece = p ? `<span class="pc ${p === p.toUpperCase() ? "w" : "b"}">${GLYPH[p.toLowerCase()]}</span>` : "";
      const coordF = i >= 56 ? `<span class="coord f">${sq[0]}</span>` : "";
      const coordR = i % 8 === 0 ? `<span class="coord r">${sq[1]}</span>` : "";
      html += `<div class="${cls.join(" ")}" data-sq="${sq}">${h}${piece}${coordF}${coordR}</div>`;
    });
    let svg = "";
    if (arrows && arrows.length) {
      svg = `<svg class="arrows" viewBox="0 0 100 100">`;
      arrows.forEach((a) => { svg += arrowSvg(this.center(a.uci.slice(0, 2)), this.center(a.uci.slice(2, 4)), a.kind || "model", a.width || 2, a.opacity || 0.9); });
      svg += `</svg>`;
    }
    this.el.innerHTML = html + svg;
  }
}

// Two colours only, so the three arrows differ by line style:
//   played = solid blue, model = dashed blue, you = white with a blue outline.
const BLUE = "#0000ff", WHITE = "#ffffff";
function arrowSvg([x1, y1], [x2, y2], kind, w, op) {
  const dx = x2 - x1, dy = y2 - y1, L = Math.hypot(dx, dy) || 1, ux = dx / L, uy = dy / L;
  const head = 2.6 + w, ex = x2 - ux * 2.2, ey = y2 - uy * 2.2, bx = ex - ux * head, by = ey - uy * head;
  const px = -uy * head * 0.55, py = ux * head * 0.55;
  const tri = `${ex},${ey} ${bx + px},${by + py} ${bx - px},${by - py}`;
  const line = (stroke, width, extra = "") => `<line x1="${x1}" y1="${y1}" x2="${bx}" y2="${by}" stroke="${stroke}" stroke-width="${width}" stroke-linecap="butt" ${extra}/>`;
  if (kind === "guess") return `<g opacity="${op}">${line(BLUE, w + 1.2)}${line(WHITE, w - 0.2)}<polygon points="${tri}" fill="${WHITE}" stroke="${BLUE}" stroke-width="0.6"/></g>`;
  if (kind === "model") return `<g opacity="${op}">${line(BLUE, w, `stroke-dasharray="${w * 1.1} ${w * 0.8}"`)}<polygon points="${tri}" fill="${BLUE}"/></g>`;
  return `<g opacity="${op}">${line(BLUE, w)}<polygon points="${tri}" fill="${BLUE}"/></g>`;
}
function legendSvg(kind) {
  return `<svg width="34" height="12" viewBox="0 0 34 12" aria-hidden="true">${arrowSvg([1, 6], [33, 6], kind, 2.6, 1)}</svg>`;
}

// -------------------------------------------------------------------- views
let players = [];
let currentPlayer = null;

function showView(name) {
  $("#hero").classList.toggle("hidden", name !== "players");
  document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${name}`));
  document.querySelectorAll("nav button").forEach((b) => b.classList.toggle("active", b.dataset.view === name));
  if (name === "analyse") analyse.refresh();
  if (name === "live") live.loadSimGames();
  if (name === "map") map.load();
}
document.querySelectorAll("nav button").forEach((b) => b.addEventListener("click", () => showView(b.dataset.view)));

const KIND = { real: "real · Lichess blitz", synthetic: "synthetic test bot", uploaded: "uploaded", added: "added by you" };

async function loadPlayers(selectId) {
  players = await api("/api/players");
  const list = $("#player-list");
  list.innerHTML = players.length
    ? players.map((p) => `<li data-id="${esc(p.id)}"><div class="name">${esc(p.name)}</div><small>${KIND[p.kind] || ""}${p.rating ? ` · ${p.rating}` : ""} · ${p.games} games · hits ${pct(p.top1)}</small></li>`).join("")
    : `<li class="muted">No players yet - add one below.</li>`;
  list.querySelectorAll("li[data-id]").forEach((li) => li.addEventListener("click", () => selectPlayer(li.dataset.id)));
  const opts = players.map((p) => `<option value="${esc(p.id)}">${esc(p.name)}</option>`).join("");
  $("#play-player").innerHTML = opts;
  $("#an-player").innerHTML = opts;
  $("#lv-player").innerHTML = `<option value="">match by player name</option>` + opts;
  const id = selectId || currentPlayer || (players[0] && players[0].id);
  if (id) selectPlayer(id);
}

function barRow(label, value, max, text, cls = "") {
  const w = Math.max(0, Math.min(100, (value / max) * 100));
  return `<div class="bar-row"><span>${label}</span><div class="bar"><i class="${cls}" style="width:${w}%"></i></div><span class="num">${text}</span></div>`;
}
function divRow(label, delta, text, cls = "") {
  // diverging bar around the centre, delta in log-odds
  const w = Math.min(50, Math.abs(delta) * 25);
  const style = delta >= 0 ? `left:50%;width:${w}%` : `left:${50 - w}%;width:${w}%`;
  return `<div class="bar-row ${cls}"><span>${label}</span><div class="bar div"><i class="${delta < 0 ? "neg" : ""}" style="${style}"></i></div><span class="num">${text}</span></div>`;
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
    ["This player's model", ev.personal_with_book || ev.personal],
    ...(ev.personal_with_book ? [["…without opening book", ev.personal]] : []),
    ["Average-player model", ev.population],
    ["Stockfish top move", ev.engine_top_move],
    ["Random legal move", ev.random],
  ].filter(([, v]) => v);
  const bs = p.blind_spots || {};
  // Counted blind spots: need real evidence, and rank by index × log(evidence) so a 6/8 fluke
  // does not outrank a 40/120 pattern.
  const rows = (bs.rows || []).filter((r) => r.index >= 1.15 && r.events >= 10)
    .sort((a, b) => (b.index - 1) * Math.log(b.events) - (a.index - 1) * Math.log(a.events)).slice(0, 8);
  const seen = new Set();
  const strengths = (bs.rows || []).filter((r) => r.index <= 0.8).reverse()
    .filter((r) => !seen.has(r.motif) && seen.add(r.motif)).slice(0, 4);
  const reliable = (p.style || []).filter((s) => s.reliable && Math.abs(s.delta) >= 0.1);
  const style = reliable.length ? reliable.slice(0, 10) : (p.style || []).filter((s) => Math.abs(s.delta) >= 0.15).slice(0, 8);
  const learned = (bs.learned || []).filter((r) => r.reliable);
  const st = p.stats || {};
  const tp = st.time_pressure;
  const cap = st.capacity || null;
  const capRows = [];
  if (cap) {
    const cx = cap.complexity || {};
    if (cx.complex_over_simple != null) capRows.push(["Walks into complex tactics (4+ pieces) vs simple ones", cx.complex_over_simple, `×${cx.complex_over_simple.toFixed(2)} <small>${cx.complex.walked_in}/${cx.complex.available} vs ${cx.simple.walked_in}/${cx.simple.available}</small>`]);
    const ov = cap.overload || {};
    if (ov.crowded_over_quiet != null) {
      const q = ov.buckets[0], c = ov.buckets[2];
      capRows.push(["Blunders in crowded positions vs quiet ones", ov.crowded_over_quiet, `×${ov.crowded_over_quiet.toFixed(2)} <small>${pct(c.blunder_rate)} vs ${pct(q.blunder_rate)}</small>`]);
    }
    if (cap.recent && cap.recent.index != null) capRows.push(["Moves one of the 3 pieces moved most recently", cap.recent.index, `×${cap.recent.index.toFixed(2)} <small>${pct(cap.recent.chosen_share)} of moves, ${pct(cap.recent.candidate_share)} of options</small>`]);
    if (cap.stale && cap.stale.index != null) capRows.push(["Returns to a piece untouched for 8+ moves", cap.stale.index, `×${cap.stale.index.toFixed(2)} <small>${pct(cap.stale.chosen_share)} of moves, ${pct(cap.stale.candidate_share)} of options</small>`]);
  }
  const memStyle = (p.style || []).filter((s) => ["allows_simple", "allows_complex", "load_eng_loss", "load_hangs", "focus_recent", "focus_stale"].includes(s.feature) && s.reliable);
  const kindNote = {
    real: "A real Lichess player (public games). Their 75 most recent rated blitz games were analysed.",
    synthetic: "A synthetic test bot with deliberately built-in habits and blind spots, used to check that the profiler finds them.",
    added: "A player you added.",
    uploaded: "Built from an uploaded PGN.",
  }[p.kind] || "";
  $("#profile").innerHTML = `<div class="grid g6">
    <div class="s6 cell">
      <p class="kicker">${esc(KIND[p.kind] || "profile")}${st.rating ? ` · rating ${st.rating}` : ""}</p>
      <h2>${esc(p.name)}</h2>
      <p class="stats-line">${st.games} games (${st.white_games} as White, ${st.black_games} as Black) · ${st.decisions} decisions analysed${st.avg_loss_pawns != null ? ` · engine match ${pct(st.engine_match)} · blunder rate ${pct(st.blunder_rate)}` : ""}</p>
      <p class="muted">${kindNote}</p>
      ${tp ? `<p class="stats-line">Short of time (${pct(tp.share)} of moves): blunder rate ${pct(tp.blunder_rate)} vs ${pct(tp.blunder_rate_normal)} normally · average loss ${tp.avg_loss.toFixed(2)} vs ${tp.avg_loss_normal.toFixed(2)} pawns.</p>` : ""}
      <button class="primary" id="play-this">Guess ${esc(p.name)}'s moves →</button>
    </div>
    <div class="s3 cell">
      <h3>How predictable?</h3>
      <p class="muted">Share of moves where the model's first guess was exactly the move played, on ${p.heldout_games} games it never saw.</p>
      ${acc.map(([n, v]) => barRow(n, v.top1, 1, `${pct(v.top1)}${v.top3 != null ? ` <small>top3 ${pct(v.top3)}</small>` : ""}`)).join("")}
    </div>
    <div class="s3 cell">
      <h3>Style fingerprint</h3>
      <p class="muted">More (→, solid) or less (←, hatched) likely than an average player to pick a move with this property. ${reliable.length ? "Only traits that are clear beyond noise (±2 standard errors) are shown." : "No trait is clear beyond noise yet; the strongest hints are shown faded."}</p>
      ${style.map((s) => divRow(esc(s.label), s.delta, `×${s.odds.toFixed(2)}${s.se ? ` <small>±${(s.se).toFixed(2)}</small>` : ""}`, s.reliable ? "" : "faint")).join("") || `<p class="muted">Plays like the average player.</p>`}
    </div>
    <div class="s6 cell">
      <h3>Blind spots</h3>
      ${learned.length ? `<p class="kicker">Learned by the model</p>
      <p class="muted">How much more readily than an average player they allow a tactic of this kind, once the material it costs is accounted for (×1 = average). These weights are used in every prediction.</p>
      ${learned.map((r) => divRow(esc(r.label.replace(/^allows /, "")), r.delta, `×${r.odds.toFixed(2)} <small>±${(r.se || 0).toFixed(2)}</small>`)).join("")}
      <p class="kicker">Counted in their games</p>` : `<p class="muted">The model found no blind spot clear beyond noise for this player. Below: what the counts suggest.</p>`}
      <p class="muted">Index = how much more often they miss this kind of move than their own average (1.0 = their average). Threats: they don't react. Walk-ins: they allow it. Opportunities: they don't play it. Small counts are shrunk towards 1.</p>
      ${rows.length ? rows.map((r) => barRow(esc(r.label), r.index, 3, `×${r.index.toFixed(2)} <small>${r.missed}/${r.events}</small>`) + (r.examples || []).slice(0, 1).map((e) => `<div class="example">e.g. move ${e.move_number}: played ${esc(e.played)}, ${esc(e.note)}</div>`).join("")).join("") : `<p class="muted">No clear blind spot yet (needs more games).</p>`}
      ${strengths.length ? `<h4>Sharp at</h4>${strengths.map((r) => `<span class="tag">${esc(r.label.replace(/^(Overlooks threats from|Misses winning|Walks into) /, ""))} ×${r.index.toFixed(2)}</span>`).join("")}` : ""}
    </div>
    ${capRows.length ? `<div class="s6 cell">
      <h3>Working memory</h3>
      <p class="muted">How many pieces this player seems to hold in mind at once. ×1 = no effect; above 1 means the pattern is stronger than chance or than their own baseline. Counted in their games; the model weights below are what the predictor actually uses.</p>
      ${capRows.map(([l, v, t]) => barRow(l, v, 3, t)).join("")}
      ${memStyle.length ? `<p class="kicker">Learned by the model</p>${memStyle.map((s) => divRow(esc(s.label), s.delta, `×${s.odds.toFixed(2)} <small>±${(s.se || 0).toFixed(2)}</small>`)).join("")}` : ""}
    </div>` : ""}
    <div class="s6 cell">
      ${(p.refutations && p.refutations.rows && p.refutations.rows.length) ? `<h4>What punishes their blunders (engine)</h4>${p.refutations.rows.slice(0, 4).map((r) => barRow(esc(r.label), r.lift, 3, `×${r.lift.toFixed(2)}`)).join("")}` : ""}
    </div>
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
  const body = {
    lichess: $("#lichess-user").value.trim() || null, chesscom: $("#chesscom-user").value.trim() || null,
    pgn: $("#pgn-text").value.trim() || null, player: $("#pgn-player").value.trim() || null,
    rating: +$("#add-rating").value || null, include_in_map: $("#add-map").checked,
  };
  const status = $("#build-status");
  try {
    let job = await api("/api/players", body);
    $("#build-btn").disabled = true;
    while (job.status !== "done" && job.status !== "error") {
      status.textContent = `${job.status}${job.progress ? ` (${Math.round(job.progress * 100)}%)` : ""}`;
      await new Promise((r) => setTimeout(r, 1000));
      job = await api(`/api/jobs/${job.id}`);
    }
    if (job.status === "error") throw new Error(job.error);
    status.textContent = job.map ? "Profile ready and added to the cognition map (see the Map tab)." : "Profile ready.";
    map.data = null;  // reload the map with the new player
    if (job.map) map.sel = job.player_id;
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
    this.board.set({ arrows: [{ uci, kind: "guess", width: 2.4 }] });
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
      { uci: r.round.actual, kind: "actual", width: 2.6 },
      ...(r.round.model_top !== r.round.actual ? [{ uci: r.round.model_top, kind: "model", width: 1.8, opacity: 0.7 }] : []),
      ...(r.round.guess && r.round.guess !== r.round.actual ? [{ uci: r.round.guess, kind: "guess", width: 1.6, opacity: 0.7 }] : []),
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
      <div>You: ${rd.guess_san ? `${esc(rd.guess_san)} <span class="verdict-mark ${youOk ? "hit" : ""}">${youOk ? "✓ +10" : "✗ miss"}</span> <small>(model gave it ${pct(rd.p_guess)})</small>` : `<span class="muted">skipped</span>`}</div>
      <div>Model: ${esc(rd.model_top_san)} <span class="verdict-mark ${modelOk ? "hit" : ""}">${modelOk ? "✓ +10" : "✗ miss"}</span> <small>(gave the real move ${pct(rd.p_actual)})</small></div>
      <div class="legend"><span>${legendSvg("actual")}played</span><span>${legendSvg("model")}model</span><span>${legendSvg("guess")}you</span></div>
      <h4>Model's prediction</h4>
      ${movesList(pred, rd.actual)}
      ${perspectiveHtml(per, pred, r.reality)}`;
  },
};

function movesList(pred, actual) {
  return `<ul class="moves">${pred.moves.map((m) => `<li><span class="san">${esc(m.san)}${m.uci === actual ? " ✓" : ""}</span> ${barInline(m.p)} <span class="num">${pct(m.p)}</span>
      ${m.book ? `<span class="tag">played ${m.book}× here before</span>` : ""}
      <div class="why">${m.reasons.map((x) => `${x.effect > 0 ? "+" : "−"} ${x.has === false ? "not: " : ""}${esc(x.label)}${Math.abs(x.personal) >= 0.2 ? ` <b>(${x.personal > 0 ? "especially" : "unusually rarely"} for this player)</b>` : ""}`).join(" · ")}</div></li>`).join("")}</ul>
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
    return `<span class="verdict-mark ${v ? "hit" : ""}">${key === "threat" ? (v ? "reacted" : "ignored it") : (v ? "found it" : "missed it")}</span>`;
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
      arrows: top.map((m, i) => ({ uci: m.uci, kind: "model", width: 2.6 - i * 0.7, opacity: 0.9 - i * 0.2 })),
    });
    const name = (players.find((p) => p.id === pid) || {}).name || pid;
    $("#an-out").innerHTML = `<h3>${esc(name)} as ${pred.turn}${pred.in_book ? ` <span class="tag">in their opening book</span>` : ""}</h3>${movesList(pred, null)}${perspectiveHtml(pred.perspective, pred, null)}`;
  },
};

// -------------------------------------------------------------------- live
const live = {
  board: null, es: null, snap: null, guess: null,
  init() {
    this.board = new Board($("#lv-board"), (u) => this.pick(u));
    this.board.set({ fen: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", interactive: false });
    $("#lv-start").addEventListener("click", () => this.start({ source: $("#lv-source").value.trim(), player_id: $("#lv-player").value || null, color: $("#lv-color").value || null }));
    $("#lv-sim").addEventListener("click", () => {
      const pid = $("#lv-player").value;
      if (!pid) { alert("Pick a profile to simulate."); return; }
      this.start({ player_id: pid, simulate_game: +$("#lv-sim-game").value, delay: +$("#lv-delay").value || 4 });
    });
    $("#lv-player").addEventListener("change", () => this.loadSimGames());
    $("#lv-lock").addEventListener("click", () => this.lock());
    $("#lv-show").addEventListener("change", () => this.render());
  },
  async loadSimGames() {
    const pid = $("#lv-player").value;
    if (!pid) { $("#lv-sim-game").innerHTML = `<option value="">pick a profile above first</option>`; return; }
    const games = await api(`/api/players/${pid}/games`);
    $("#lv-sim-game").innerHTML = games.map((g) => `<option value="${g.index}">${esc(g.white)} – ${esc(g.black)} (${esc(g.result)})</option>`).join("");
  },
  async start(body) {
    try {
      if (this.es) this.es.close();
      const r = await api("/api/live", body);
      this.guess = null;
      this.es = new EventSource(`/api/live/${r.id}/events`);
      this.es.onmessage = (e) => { this.snap = JSON.parse(e.data); this.render(); if (["finished", "error"].includes(this.snap.status)) this.es.close(); };
      this.es.onerror = () => { $("#lv-status").textContent = "Connection lost."; };
      $("#lv-status").textContent = "Connecting…";
    } catch (e) { alert(e.message); }
  },
  pick(uci) {
    this.guess = uci;
    $("#lv-lock").disabled = false;
    this.board.set({ arrows: [{ uci, kind: "guess", width: 2.4 }] });
  },
  async lock() {
    if (!this.snap || !this.guess) return;
    try { await api(`/api/live/${this.snap.id}/guess`, { uci: this.guess }); }
    catch (e) { $("#lv-status").textContent = e.message; }
    $("#lv-lock").disabled = true;
  },
  render() {
    const s = this.snap;
    if (!s) return;
    const pend = s.pending;
    if (!pend || pend.guess) this.guess = pend && pend.guess;
    const show = $("#lv-show").checked || s.status === "finished";
    const arrows = [];
    if (pend && pend.guess) arrows.push({ uci: pend.guess, kind: "guess", width: 2.4 });
    else if (this.guess && pend) arrows.push({ uci: this.guess, kind: "guess", width: 2.4, opacity: 0.6 });
    if (pend && show) pend.prediction.moves.slice(0, 3).forEach((m, i) => arrows.push({ uci: m.uci, kind: "model", width: 2.4 - i * 0.6, opacity: 0.85 - i * 0.2 }));
    if (!pend && s.last && s.last.actual) arrows.push({ uci: s.last.actual, kind: "actual", width: 2.4 });
    this.board.set({ fen: s.fen, orientation: s.player_color || "white", legal: pend && !pend.guess ? s.legal : [], last: s.last_move, interactive: !!pend && !pend.guess, arrows, heat: pend && show ? pend.prediction.perspective.attention : null });
    $("#lv-title").textContent = `${s.white} – ${s.black}`;
    const sc = s.score;
    $("#lv-user").textContent = sc.user_hits; $("#lv-user-n").textContent = `/${sc.user_guesses}`;
    $("#lv-model").textContent = sc.model_hits; $("#lv-model-n").textContent = `/${sc.predicted} · top3 ${sc.model_top3}`;
    $("#lv-history").innerHTML = s.history.map((m, i) => (i % 2 === 0 ? `<b>${i / 2 + 1}.</b> ` : "") + esc(m)).join(" ");
    const fmt = (ms) => ms == null ? "–" : `${Math.floor(ms / 60000)}:${String(Math.floor(ms / 1000) % 60).padStart(2, "0")}`;
    $("#lv-clock").textContent = s.clock ? `⏱ White ${fmt(s.clock.white)} · Black ${fmt(s.clock.black)}` : "";
    let status;
    if (s.status === "error") status = `Error: ${s.error}`;
    else if (s.status === "finished") status = `Game over ${s.result || ""}. Model predicted ${sc.model_hits}/${sc.predicted} moves exactly (${sc.model_top3} in its top 3); you got ${sc.user_hits}/${sc.user_guesses}.`;
    else if (pend) status = (pend.guess ? `Guess locked: ${pend.guess}. Waiting for ${s.player}…` : `${s.player} is thinking. Click your guess on the board.`) + (pend.updates ? ` (prediction updated after ${pend.thinking_for}s of thinking)` : "");
    else status = s.status === "live" ? "Opponent to move…" : s.status;
    $("#lv-status").textContent = status;
    $("#lv-setup").classList.toggle("hidden", s.status === "live");
    const predEl = $("#lv-pred");
    if (pend && show) {
      predEl.classList.remove("hidden");
      predEl.innerHTML = `<h3>Prediction for ${esc(s.player)} <small>registered ${new Date(pend.made_at * 1000).toLocaleTimeString()}</small></h3>${movesList(pend.prediction, null)}${perspectiveHtml(pend.prediction.perspective, pend.prediction, null)}`;
    } else if (pend) {
      predEl.classList.remove("hidden");
      predEl.innerHTML = `<p class="muted">The model has made its prediction (hidden). Tick “show model's prediction” to see it, or guess first.</p>`;
    } else predEl.classList.add("hidden");
    const lastEl = $("#lv-last");
    if (s.last && s.last.actual) {
      const l = s.last;
      lastEl.classList.remove("hidden");
      lastEl.innerHTML = `<div class="verdict">${esc(s.player)} played <b>${esc(l.actual_san)}</b></div>
        <div>Model: ${esc(l.model_top)} <span class="verdict-mark ${l.model_hit ? "hit" : ""}">${l.model_hit ? "✓ hit" : "✗ miss"}</span> <small>(gave the real move ${pct(l.p_actual)})</small></div>
        <div>You: ${l.guess ? `${esc(l.guess)} <span class="verdict-mark ${l.user_hit ? "hit" : ""}">${l.user_hit ? "✓ hit" : "✗ miss"}</span>` : `<span class="muted">no guess</span>`}</div>
        ${!pend && l.perspective ? `<ul class="notes">${l.perspective.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul>` : ""}`;
    } else lastEl.classList.add("hidden");
  },
};

// --------------------------------------------------------------------- map
// Player types are told apart by shape, not colour (two-colour design).
function typeMark(type, x, y, r, extra = "") {
  const k = type % 6, sw = 1.6, f = (filled) => (filled ? `fill="${BLUE}" stroke="${BLUE}"` : `fill="${WHITE}" stroke="${BLUE}"`);
  const d = r * 1.25;
  if (k === 0) return `<circle cx="${x}" cy="${y}" r="${r}" ${f(true)} stroke-width="${sw}" ${extra}/>`;
  if (k === 1) return `<circle cx="${x}" cy="${y}" r="${r}" ${f(false)} stroke-width="${sw}" ${extra}/>`;
  if (k === 2) return `<rect x="${x - r}" y="${y - r}" width="${2 * r}" height="${2 * r}" ${f(true)} stroke-width="${sw}" ${extra}/>`;
  if (k === 3) return `<rect x="${x - r}" y="${y - r}" width="${2 * r}" height="${2 * r}" ${f(false)} stroke-width="${sw}" ${extra}/>`;
  if (k === 4) return `<polygon points="${x},${y - d} ${x + d},${y} ${x},${y + d} ${x - d},${y}" ${f(true)} stroke-width="${sw}" ${extra}/>`;
  return `<polygon points="${x},${y - d} ${x + d},${y} ${x},${y + d} ${x - d},${y}" ${f(false)} stroke-width="${sw}" ${extra}/>`;
}
const map = {
  data: null, sel: null,
  async load() {
    if (this.data) return this.render();
    try { this.data = await api("/api/map"); }
    catch (e) { $("#map-plot").innerHTML = `<p class="muted">${esc(e.message)}</p>`; return; }
    const opts = this.data.components.map((c) => `<option value="${c.index - 1}">PC${c.index} · r ${c.split_half_r.toFixed(2)}</option>`).join("");
    $("#map-x").innerHTML = opts; $("#map-y").innerHTML = opts; $("#map-y").value = "1";
    $("#map-x").onchange = $("#map-y").onchange = () => this.render();
    this.render();
    this.side();
    const pre = this.sel && this.data.players.find((p) => p.id === this.sel);
    if (pre) this.pick(pre);
  },
  axisName(c) {
    // Name each end of the axis by its own two strongest loadings. A negative loading on a
    // "bad" trait reads as its opposite at the positive end, so show it as "less …".
    const pos = [...c.loadings.filter((x) => x.loading > 0).map((x) => x.label), ...c.loadings.filter((x) => x.loading < 0).map((x) => `less ${x.label}`)];
    const neg = [...c.loadings.filter((x) => x.loading < 0).map((x) => x.label), ...c.loadings.filter((x) => x.loading > 0).map((x) => `less ${x.label}`)];
    return { pos: pos.slice(0, 2).join(", "), neg: neg.slice(0, 2).join(", ") };
  },
  render() {
    const d = this.data; if (!d) return;
    const xi = +$("#map-x").value, yi = +$("#map-y").value;
    const W = 640, H = 560, pad = 46, padB = 76;
    const xs = d.players.map((p) => p.pc[xi]), ys = d.players.map((p) => p.pc[yi]);
    const ext = (a) => { const m = Math.max(...a.map(Math.abs)) * 1.1 || 1; return [-m, m]; };
    const [x0, x1] = ext(xs), [y0, y1] = ext(ys);
    const sx = (v) => pad + ((v - x0) / (x1 - x0)) * (W - 2 * pad), sy = (v) => H - padB - ((v - y0) / (y1 - y0)) * (H - pad - padB);
    const cx = d.components[xi], cy = d.components[yi];
    const ax = this.axisName(cx), ay = this.axisName(cy);
    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Player map">
      <line class="axis" x1="${pad}" y1="${sy(0)}" x2="${W - pad}" y2="${sy(0)}"/><line class="axis" x1="${sx(0)}" y1="${pad}" x2="${sx(0)}" y2="${H - padB}"/>
      <text x="${pad}" y="16">↑ PC${yi + 1}: ${esc(ay.pos)}</text>
      <text x="${pad}" y="${H - 44}">↓ ${esc(ay.neg)}</text>
      <text x="${pad}" y="${H - 26}">← PC${xi + 1}: ${esc(ax.neg)}</text>
      <text x="${W - pad}" y="${H - 8}" text-anchor="end">${esc(ax.pos)} →</text>`;
    d.players.forEach((p, i) => {
      const r = 4 + ((p.rating - 800) / 2200) * 5;
      if (p.added) svg += `<circle cx="${sx(p.pc[xi]).toFixed(1)}" cy="${sy(p.pc[yi]).toFixed(1)}" r="${(Math.max(3, r) + 4).toFixed(1)}" fill="none" stroke="${BLUE}" stroke-width="1.5" stroke-dasharray="2 2"/><text x="${(sx(p.pc[xi]) + Math.max(3, r) + 6).toFixed(1)}" y="${(sy(p.pc[yi]) + 4).toFixed(1)}" style="font-weight:800">${esc(p.name)}</text>`;
      svg += `<g class="dot ${this.sel === p.id ? "sel" : ""}" data-i="${i}"><title>${esc(p.name)} (${p.rating})</title>${typeMark(p.type, +sx(p.pc[xi]).toFixed(1), +sy(p.pc[yi]).toFixed(1), Math.max(3, r))}</g>`;
    });
    svg += `</svg><p class="muted" style="margin:4px 8px">Each mark is a real Lichess blitz player; size = rating, shape = player type; dashed ring and name = players you added. Rating has been regressed out of the axes. Click a mark. Add players under <b>01 Profiles → + Add a player</b>.</p>`;
    $("#map-plot").innerHTML = svg;
    $("#map-plot").querySelectorAll(".dot").forEach((el) => el.addEventListener("click", () => this.pick(d.players[+el.dataset.i])));
    const weak = d.silhouette < 0.25 ? `<p class="no" style="font-weight:400">Weak separation (silhouette ${d.silhouette}): players form a continuum, not distinct types. Read the axes; treat the shapes as rough regions only.</p>` : "";
    $("#map-types").innerHTML = `<div class="cell"><h3>Player types (k-means, k=${d.k}, silhouette ${d.silhouette})</h3>${weak}${d.types.map((t) => `<div style="margin:10px 0"><svg class="swatch" width="16" height="16" viewBox="0 0 16 16">${typeMark(t.type, 8, 8, 5)}</svg><b>Type ${t.type + 1}</b> · ${t.size} players · mean rating ${t.mean_rating}<div class="why">${t.traits.slice(0, 4).map((x) => `${x.z > 0 ? "more" : "less"} ${esc(x.label)} (${x.z > 0 ? "+" : ""}${x.z})`).join(" · ")}</div></div>`).join("")}</div>`;
  },
  side() {
    const d = this.data;
    const pr = d.prediction;
    $("#map-side").innerHTML = `<div class="cell"><h3>Is this real?</h3>
      <p class="muted">Each player was also profiled twice, from two disjoint halves of their games. If an axis is a stable trait, both halves land in the same place (split-half r). Shuffled players give |r| ≤ ${d.null_split_half_r_95} (95%).</p>
      ${d.components.map((c) => barRow(`PC${c.index}`, Math.max(0, c.split_half_r), 1, `r ${c.split_half_r.toFixed(2)}`, c.split_half_r < 0.4 ? "neg" : "")).join("")}
      <h4>Prediction on held-out games (${pr.n} players)</h4>
      ${barRow("Personal model", pr.mean_top1.personal, 1, pct(pr.mean_top1.personal))}
      ${barRow("Average-player model", pr.mean_top1.population, 1, pct(pr.mean_top1.population))}
      ${barRow("Stockfish top move", pr.mean_top1.engine_top_move, 1, pct(pr.mean_top1.engine_top_move))}
      <p class="muted">Personal model better log-loss for ${pr.players_with_better_logloss}/${pr.n} players.</p></div>
      <div class="cell" id="map-axes"><h3>What the axes mean</h3>${d.components.map((c) => `<h4>PC${c.index} <small>${pct(c.explained)} of variance</small></h4>${c.loadings.slice(0, 5).map((l) => divRow(esc(l.label), l.loading * 4, l.loading.toFixed(2))).join("")}`).join("")}</div>
      <div class="cell" id="map-player"><p class="muted">Click a player on the map.</p></div>`;
  },
  pick(p) {
    this.sel = p.id; this.render();
    const xi = +$("#map-x").value, yi = +$("#map-y").value;
    $("#map-player").innerHTML = `<h3>${esc(p.name)} <small>${p.rating}</small></h3>
      <p>Type ${p.type + 1}. Position from half A of their games: (${p.pcA[xi].toFixed(1)}, ${p.pcA[yi].toFixed(1)}), half B: (${p.pcB[xi].toFixed(1)}, ${p.pcB[yi].toFixed(1)}).</p>
      <button class="primary" id="map-open">Open profile</button>`;
    $("#map-open").addEventListener("click", async () => { showView("players"); await selectPlayer(p.id); });
  },
};

// ------------------------------------------------------------ the field
// A field of annotation marks: "?" (missed) on red, "!" (seen) on black.
// Smooth value noise drifts over time; a click sends a ring that flips cells.
const field = {
  cell: 20, waves: [], t: 0, running: true,
  init() {
    const c = $("#field");
    if (!c) return;
    this.c = c; this.ctx = c.getContext("2d");
    this.reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const resize = () => {
      const r = c.getBoundingClientRect(), dpr = Math.min(2, window.devicePixelRatio || 1);
      c.width = Math.max(1, r.width * dpr); c.height = Math.max(1, r.height * dpr);
      this.cell = r.width / 64;  // 8 field cells per page grid cell, like squares on a board
      this.dpr = dpr; this.cols = Math.ceil(r.width / this.cell) + 1; this.rows = Math.ceil(r.height / this.cell) + 1;
      this.draw();
    };
    window.addEventListener("resize", resize);
    resize();
    c.addEventListener("click", (e) => {
      const r = c.getBoundingClientRect();
      this.waves.push({ x: (e.clientX - r.left) / this.cell, y: (e.clientY - r.top) / this.cell, t0: this.t });
      if (this.reduce) this.draw();
    });
    new IntersectionObserver(([en]) => { this.running = en.isIntersecting; }).observe(c);
    if (!this.reduce) { const loop = () => { if (this.running && !document.hidden) { this.t += 0.016; this.draw(); } requestAnimationFrame(loop); }; requestAnimationFrame(loop); }
  },
  hash(x, y, z) { let h = x * 374761393 + y * 668265263 + z * 2147483647; h = (h ^ (h >>> 13)) * 1274126177; return ((h ^ (h >>> 16)) >>> 0) / 4294967295; },
  noise(x, y, z) {
    const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z), xf = x - xi, yf = y - yi, zf = z - zi;
    const s = (t) => t * t * (3 - 2 * t), u = s(xf), v = s(yf), w = s(zf);
    const L = (a, b, t) => a + (b - a) * t, H = (a, b, c) => this.hash(xi + a, yi + b, zi + c);
    return L(L(L(H(0,0,0), H(1,0,0), u), L(H(0,1,0), H(1,1,0), u), v), L(L(H(0,0,1), H(1,0,1), u), L(H(0,1,1), H(1,1,1), u), v), w);
  },
  draw() {
    const { ctx, cell, dpr } = this;
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = WHITE; ctx.fillRect(0, 0, this.c.width, this.c.height);
    ctx.font = `700 ${Math.round(cell * 0.5)}px "JetBrains Mono", monospace`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    this.waves = this.waves.filter((w) => this.t - w.t0 < 4);
    const z = this.t * 0.25;
    for (let y = 0; y < this.rows; y++) for (let x = 0; x < this.cols; x++) {
      let on = this.noise(x * 0.11, y * 0.16, z) + 0.5 * this.noise(x * 0.3, y * 0.4, z * 1.7) > 0.95;
      for (const w of this.waves) {
        const d = Math.hypot(x - w.x, y - w.y), front = (this.t - w.t0) * 22;
        if (Math.abs(d - front) < 1.6) on = !on;
      }
      const px = x * cell, py = y * cell;
      if (on) { ctx.fillStyle = BLUE; ctx.fillRect(px, py, cell + 0.5, cell + 0.5); ctx.fillStyle = WHITE; ctx.fillText("?", px + cell / 2, py + cell / 2 + 1); }
      else { ctx.fillStyle = "rgba(0,0,255,.45)"; ctx.fillText("!", px + cell / 2, py + cell / 2 + 1); }
      if (x % 8 === 0 && x) { ctx.fillStyle = BLUE; ctx.fillRect(px - 0.5, 0, 1, this.c.height); }
    }
  },
};

// -------------------------------------------------------------------- boot
(async function boot() {
  play.init();
  analyse.init();
  live.init();
  field.init();
  try {
    const h = await api("/api/health");
    $("#engine-badge").textContent = `${h.engine ? "Stockfish: on" : "Stockfish: off"} · ${h.neural ? "Maia-2: on" : "Maia-2: off"}`;
  } catch (e) { /* ignore */ }
  await loadPlayers();
})();
