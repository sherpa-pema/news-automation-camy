/**
 * Entry point wiring HUD, Canvas Engine, WebSocket, Replay, and Boss Drawers.
 */
import { OfficeCanvasEngine } from "./canvas_engine.js";
import { DrawerManager, OfficeAudio } from "./drawers.js";
import { OfficeWsClient, ReplayController } from "./ws_client.js";

const runDateSelect = document.getElementById("runDateSelect");
const runKindBadge = document.getElementById("runKindBadge");
const overallStatusBadge = document.getElementById("overallStatusBadge");
const liveDot = document.getElementById("liveDot");
const primaryActionBtn = document.getElementById("primaryActionBtn");
const tickerLatest = document.getElementById("tickerLatest");
const hoverTooltip = document.getElementById("hoverTooltip");
const replayBanner = document.getElementById("replayBanner");
const replayLabel = document.getElementById("replayLabel");

let currentState = null;
let currentLogs = [];
let currentDate = null;

const audio = new OfficeAudio();

const engine = new OfficeCanvasEngine(document.getElementById("officeCanvas"), {
  onSelectStation: (stationId) => {
    drawers.openStation(stationId);
  },
  onHoverInfo: (info) => {
    if (!info) {
      hoverTooltip.classList.add("hidden");
      return;
    }
    hoverTooltip.innerHTML = `<strong>${info.title}</strong><br><span style="color:#8ea5c7">${info.subtitle}</span>`;
    hoverTooltip.style.left = `${Math.min(window.innerWidth - 290, info.x + 14)}px`;
    hoverTooltip.style.top = `${Math.max(54, info.y - 44)}px`;
    hoverTooltip.classList.remove("hidden");
  },
});

const drawers = new DrawerManager({
  drawerEl: document.getElementById("sideDrawer"),
  titleEl: document.getElementById("drawerTitle"),
  subtitleEl: document.getElementById("drawerSubtitle"),
  tabsEl: document.getElementById("drawerTabs"),
  bodyEl: document.getElementById("drawerBody"),
  modalOverlay: document.getElementById("modalOverlay"),
  modalTitle: document.getElementById("modalTitle"),
  modalBody: document.getElementById("modalBody"),
  onAction: handleAction,
});

const replay = new ReplayController(engine, {
  onTickLabel: (txt) => {
    replayLabel.textContent = txt;
    tickerLatest.textContent = txt;
  },
  onDone: () => {
    replayBanner.classList.add("hidden");
    if (currentState) updateHudFromState(currentState);
  },
});

const wsClient = new OfficeWsClient({
  onConnectionChange: (connected) => {
    liveDot.classList.toggle("offline", !connected);
  },
  onSnapshot: (state) => {
    currentState = state;
    currentLogs = state.recent_logs || [];
    engine.setState(state);
    drawers.setState(state, currentLogs);
    updateHudFromState(state);
  },
  onOfficeEvent: (ev) => {
    if (ev.type === "packet_handoff" && ev.from && ev.to) {
      engine.spawnPacket({
        from: ev.from,
        to: ev.to,
        label: ev.label || ev.artifact,
        artifact: ev.artifact,
        sizeBytes: ev.size_bytes,
        color: ev.status === "alert" ? "#ef4444" : "#f5a623",
      });
    } else if (ev.type === "stage_status" && ev.stage) {
      engine.showBubble(ev.stage, ev.message || ev.status, 4000);
      if (ev.status === "done") audio.chime("done");
      if (ev.status === "failed") audio.chime("error");
    } else if (ev.type === "boss_alert") {
      engine.showBubble("boss", ev.message, 6000, "#f5a623");
      audio.chime("alert");
    }
    tickerLatest.textContent = ev.message || ev.label || `${ev.type}: ${ev.stage || ""}`;
  },
  onLogLines: (lines) => {
    currentLogs.push(...lines);
    if (currentLogs.length > 400) currentLogs = currentLogs.slice(-400);
    if (lines.length) tickerLatest.textContent = lines[lines.length - 1];
    drawers.setState(currentState, currentLogs);
  },
});

function updateHudFromState(state) {
  const manifest = state.manifest || {};
  const rk = (manifest.run_kind || "test").toUpperCase();
  const st = (manifest.status || "pending").toUpperCase();
  const ed = manifest.stages?.editor?.status || "pending";
  const mediaSt = manifest.stages?.media?.status || "pending";
  const qaSt = manifest.stages?.qa?.status || "pending";
  const busy = state.active_process || state.active_job?.status === "running";

  liveDot.classList.toggle("running", Boolean(busy));

  runKindBadge.textContent = rk;
  runKindBadge.className = `badge ${rk === "PRODUCTION" ? "badge-production" : "badge-test"}`;

  overallStatusBadge.textContent = busy ? "RUNNING" : st;
  overallStatusBadge.className = `badge badge-${busy ? "running" : manifest.status === "done" ? "done" : manifest.status === "paused" ? "paused" : manifest.status === "failed" ? "failed" : "neutral"}`;

  // Context-aware primary button
  primaryActionBtn.disabled = false;
  if (busy) {
    primaryActionBtn.textContent = "⏳ Pipeline Running...";
    primaryActionBtn.disabled = true;
  } else if (!state.exists || manifest.status === "pending") {
    primaryActionBtn.textContent = "▶ Start Morning Prep";
    primaryActionBtn.onclick = () => drawers.openStation("boss", "boss");
  } else if (ed === "prepared" || ed === "needs_revision") {
    primaryActionBtn.textContent = "✍ Draft Script";
    primaryActionBtn.onclick = () => drawers.openStation("boss", state.llm_writer?.configured ? "boss" : "manual");
  } else if (ed === "awaiting_approval") {
    primaryActionBtn.textContent = "✅ Review & Approve Script";
    primaryActionBtn.onclick = () => drawers.openStation("boss", "boss");
  } else if (ed === "done" && (mediaSt === "awaiting_approval" || mediaSt === "prepared") && state.artifacts?.has_picker_html) {
    primaryActionBtn.textContent = "🎨 Pick Visuals & Render";
    primaryActionBtn.onclick = () => drawers._openIframeModal("media/review/picker.html", `Media Shot Picker (${state.date})`);
  } else if (ed === "done" && !state.artifacts?.has_picker_html && qaSt !== "done") {
    primaryActionBtn.textContent = "🎙 Prepare Audio & Visuals";
    primaryActionBtn.onclick = () => handleAction("prepare-media", { force: false });
  } else if (ed === "done" && manifest.status !== "done") {
    primaryActionBtn.textContent = "🎬 Finish Broadcast";
    primaryActionBtn.onclick = () => handleAction("finish", {});
  } else if (state.artifacts?.has_final_mp4) {
    primaryActionBtn.textContent = "🎥 Watch Final Video";
    primaryActionBtn.onclick = () => drawers._openVideoModal("render/final.mp4", `Final Broadcast (${state.date})`);
  } else {
    primaryActionBtn.textContent = "👔 Boss Desk";
    primaryActionBtn.onclick = () => drawers.openStation("boss", "boss");
  }

  if (currentLogs.length) {
    tickerLatest.textContent = currentLogs[currentLogs.length - 1];
  } else if (state.exists) {
    tickerLatest.textContent = `Loaded run ${state.date} (${st})`;
  } else {
    tickerLatest.textContent = `Run ${state.date} not started yet — click Start Morning Prep!`;
  }
}

async function handleAction(action, payload) {
  if (!currentDate) return;
  try {
    const res = await fetch(`/api/runs/${encodeURIComponent(currentDate)}/${action}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload || {}),
    });
    const data = await res.json();
    if (!res.ok) {
      alert(data.detail || `Action '${action}' failed.`);
      return;
    }
    if (action === "draft-manual") {
      if (data.ok) {
        alert("Draft passed validation! Ready to Approve.");
        drawers.openStation("boss", "boss");
      } else {
        alert("Validation failed — see error list in Manual Draft tab.");
      }
    }
    if (action === "approve") {
      audio.chime("done");
    }
    // Refresh snapshot immediately
    const stateRes = await fetch(`/api/runs/${encodeURIComponent(currentDate)}/state`);
    if (stateRes.ok) {
      const st = await stateRes.json();
      currentState = st;
      engine.setState(st);
      drawers.setState(st, st.recent_logs || currentLogs);
      updateHudFromState(st);
    }
  } catch (err) {
    alert("Request failed: " + err.message);
  }
}

async function loadRunsAndConnect() {
  const res = await fetch("/api/runs");
  const data = await res.json();
  const runs = data.runs || [];

  runDateSelect.innerHTML = runs
    .map((r) => `<option value="${r.date}">${r.date} (${r.status})</option>`)
    .join("");

  // Prefer most recent existing run or today
  const chosen = runs.find((r) => r.exists) || runs[0];
  if (chosen) {
    currentDate = chosen.date;
    runDateSelect.value = currentDate;
    wsClient.connect(currentDate);
  }
}

runDateSelect.addEventListener("change", () => {
  replay.stop();
  currentDate = runDateSelect.value;
  wsClient.connect(currentDate);
});

document.getElementById("replayBtn").addEventListener("click", async () => {
  if (!currentDate) return;
  const res = await fetch(`/api/runs/${encodeURIComponent(currentDate)}/replay`);
  const data = await res.json();
  if (!data.events || !data.events.length) {
    alert("No stage events recorded for this date yet.");
    return;
  }
  const speed = Number(document.getElementById("replaySpeedSelect").value) || 3;
  replayBanner.classList.remove("hidden");
  replay.start(data.events, speed);
});

document.getElementById("stopReplayBtn").addEventListener("click", () => replay.stop());

document.getElementById("soundToggleBtn").addEventListener("click", (e) => {
  const on = audio.toggle();
  e.target.textContent = on ? "🔊 Sound On" : "🔇 Sound Off";
});

document.getElementById("bossDrawerBtn").addEventListener("click", () => {
  drawers.openStation("boss", "boss");
});

document.getElementById("toggleLogDrawerBtn").addEventListener("click", () => {
  drawers.openStation("orchestrator", "logs");
});

document.getElementById("closeDrawerBtn").addEventListener("click", () => drawers.close());
document.getElementById("closeModalBtn").addEventListener("click", () => drawers.closeModal());
document.getElementById("modalOverlay").addEventListener("click", (e) => {
  if (e.target === e.currentTarget) drawers.closeModal();
});

window.addEventListener("message", async (e) => {
  if (e.origin !== window.location.origin) return;
  if (e.data?.type === "office_picks_submitted") {
    drawers.closeModal();
    audio.chime("done");
    tickerLatest.textContent = `Visual picks applied for ${e.data.date} — rendering & publishing...`;
    if (currentDate) {
      const stateRes = await fetch(`/api/runs/${encodeURIComponent(currentDate)}/state`);
      if (stateRes.ok) {
        const st = await stateRes.json();
        currentState = st;
        engine.setState(st);
        drawers.setState(st, st.recent_logs || currentLogs);
        updateHudFromState(st);
      }
    }
  }
});

document.getElementById("doctorBtn").addEventListener("click", async () => {
  drawers.openModal("System Readiness (--doctor)", `<p class="muted">Running diagnostics...</p>`);
  const res = await fetch("/api/doctor");
  const data = await res.json();
  const rows = (data.checks || [])
    .map(
      (c) => `
      <div class="artifact-item">
        <span><strong>${c.name}</strong>: ${c.detail}</span>
        <span class="badge ${c.status === "PASS" ? "badge-done" : c.status === "WARN" ? "badge-paused" : "badge-failed"}">${c.status}</span>
      </div>`
    )
    .join("");
  drawers.openModal(`System Readiness (${data.ok ? "READY" : "ISSUES FOUND"})`, `<div class="artifact-list">${rows}</div>`);
});

loadRunsAndConnect();
