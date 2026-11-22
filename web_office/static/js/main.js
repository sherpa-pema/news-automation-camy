/**
 * Entry point wiring HUD, Canvas Engine, WebSocket, Replay, and Boss Drawers.
 */
import { OfficeCanvasEngine } from "./canvas_engine.js";
import { DrawerManager, OfficeAudio } from "./drawers.js";
import { OfficeWsClient, ReplayController } from "./ws_client.js";

const runDateSelect = document.getElementById("runDateSelect");
const runKindBadge = document.getElementById("runKindBadge");
const overallStatusBadge = document.getElementById("overallStatusBadge");
const categoryBadge = document.getElementById("categoryBadge");
const liveDot = document.getElementById("liveDot");
const primaryActionBtn = document.getElementById("primaryActionBtn");
const tickerLatest = document.getElementById("tickerLatest");
const hoverTooltip = document.getElementById("hoverTooltip");
const replayBanner = document.getElementById("replayBanner");
const replayLabel = document.getElementById("replayLabel");
const stageRail = document.getElementById("stageRail");
const nextActionBar = document.getElementById("nextActionBar");
const nextActionText = document.getElementById("nextActionText");
const nextActionActions = document.getElementById("nextActionActions");
const hudStopBtn = document.getElementById("hudStopBtn");
const toastContainer = document.getElementById("toastContainer");

const STAGE_ORDER = ["scraper", "editor", "voice", "timeline", "media", "motion", "render", "qa"];

let currentState = null;
let currentLogs = [];
let currentDate = null;
let inspectingHistoricalRun = false;
let substageProgress = {}; // stage -> { current, total, percent, message, eta }
let liveSourceHealth = {}; // source_name -> stats

const audio = new OfficeAudio();

export function showToast(message, kind = "info", durationMs = 5000) {
  if (!toastContainer) return;
  const toast = document.createElement("div");
  toast.className = `toast toast-${kind}`;
  const icon = kind === "done" ? "✅" : kind === "error" ? "❌" : "ℹ️";
  toast.innerHTML = `<span>${icon}</span><span>${escapeHtml(message)}</span>`;
  toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transition = "opacity 0.3s ease";
    setTimeout(() => toast.remove(), 320);
  }, durationMs);
}
window.showOfficeToast = showToast;

function notifyBoss(title, body) {
  if (document.hidden) {
    document.title = "🔔 Action Needed — Nepal News Bot HQ";
    if ("Notification" in window && Notification.permission === "granted") {
      try {
        new Notification(title, { body, icon: "/favicon.ico" });
      } catch (_) {}
    }
  }
}

document.addEventListener("visibilitychange", () => {
  if (!document.hidden) {
    document.title = "Nepal News Bot HQ";
  }
});

function escapeHtml(str) {
  return String(str ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function formatElapsed(startTime, endTime) {
  if (!startTime) return "";
  const start = new Date(startTime).getTime();
  const end = endTime ? new Date(endTime).getTime() : Date.now();
  const sec = Math.max(0, Math.floor((end - start) / 1000));
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}:${s < 10 ? "0" : ""}${s}`;
}

function updateStageRail(state, isStandby = false) {
  if (!stageRail) return;
  const manifest = state?.manifest || {};
  const stages = manifest.stages || {};

  stageRail.innerHTML = STAGE_ORDER.map((sKey) => {
    const meta = stages[sKey] || {};
    const status = isStandby ? "pending" : (meta.status || "pending");
    const retryCount = isStandby ? 0 : (meta.retry_count || 0);
    const elapsed = isStandby ? "" : formatElapsed(meta.start_time, meta.end_time);
    const prog = isStandby ? null : substageProgress[sKey];

    let progressHtml = "";
    let progBarHtml = "";
    if (status === "running" && prog && prog.total > 0) {
      const pct = Math.min(100, Math.round((prog.current / prog.total) * 100));
      const etaStr = prog.eta ? ` • ~${prog.eta}s` : "";
      progressHtml = `<span class="chip-progress-text">${prog.current}/${prog.total}${etaStr}</span>`;
      progBarHtml = `<div class="chip-progress-bar" style="width: ${pct}%"></div>`;
    }

    const retryHtml = retryCount > 0 ? `<span class="chip-retry">↺ ${retryCount}</span>` : "";
    const clockHtml = elapsed ? `<span class="chip-clock">${elapsed}</span>` : "";

    return `
      <div class="stage-chip chip-${status}" data-stage="${sKey}" role="button" tabindex="0" title="${sKey} (${status})">
        <span class="chip-dot ${status}"></span>
        <span class="chip-name">${sKey}</span>
        ${clockHtml}
        ${retryHtml}
        ${progressHtml}
        ${progBarHtml}
      </div>
    `;
  }).join("");

  stageRail.querySelectorAll(".stage-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      drawers.openStation(chip.dataset.stage);
    });
  });
}

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
  onReconnect: async () => {
    if (currentDate) {
      const res = await authFetch(`/api/runs/${encodeURIComponent(currentDate)}/state`);
      if (res.ok) {
        const st = await res.json();
        currentState = st;
        if (!replay.active) {
          engine.setState(st);
          drawers.setState(st, st.recent_logs || currentLogs);
        }
        updateStageRail(st);
        updateHudFromState(st);
      }
    }
  },
  onJobOutput: (data) => {
    if (data?.line) {
      currentLogs.push(`[${data.action || "job"}] ${data.line}`);
      if (currentLogs.length > 400) currentLogs = currentLogs.slice(-400);
      tickerLatest.textContent = data.line;
      drawers.setState(currentState, currentLogs);
    }
  },
  onSnapshot: (state) => {
    currentState = state;
    currentLogs = state.recent_logs || [];
    if (!replay.active) {
      engine.setState(state);
    }
    drawers.setState(state, currentLogs);
    updateStageRail(state);
    updateHudFromState(state);
  },
  onOfficeEvent: (ev) => {
    if (replay.active) return; // Pause live events while replay is active
    if (ev.type === "packet_handoff" && ev.from && ev.to) {
      engine.spawnPacket({
        from: ev.from,
        to: ev.to,
        label: ev.label || ev.artifact,
        artifact: ev.artifact,
        sizeBytes: ev.size_bytes,
        color: ev.status === "alert" ? "#ef4444" : "#f5a623",
      });
    } else if (ev.type === "substage_progress" && ev.stage) {
      substageProgress[ev.stage] = {
        current: ev.current,
        total: ev.total,
        message: ev.message,
        eta: ev.extra?.eta_seconds,
      };
      const etaText = ev.extra?.eta_seconds ? ` • ~${ev.extra.eta_seconds}s ETA` : "";
      engine.showBubble(ev.stage, `${ev.message || `Progress ${ev.current}/${ev.total}`}${etaText}`, 3500);
      if (currentState) updateStageRail(currentState);
    } else if (ev.type === "scraper_source_health") {
      if (ev.extra?.source) {
        liveSourceHealth[ev.extra.source] = {
          name: ev.extra.source,
          status: ev.status || "ok",
          ...ev.extra,
        };
      }
    } else if (ev.type === "publish_complete") {
      audio.chime("done");
      showToast(ev.message || "Broadcast successfully published to data/latest & OneDrive!", "done", 6000);
      if (currentState) {
        currentState.artifacts = currentState.artifacts || {};
        currentState.artifacts.has_final_mp4 = true;
        updateHudFromState(currentState);
      }
    } else if (ev.type === "pipeline_stopped") {
      audio.chime("alert");
      showToast(ev.message || "Pipeline execution stopped by operator.", "info", 5000);
    } else if (ev.type === "stage_status" && ev.stage) {
      engine.showBubble(ev.stage, ev.message || ev.status, 4000);
      if (ev.status === "done") {
        audio.chime("done");
        delete substageProgress[ev.stage];
      } else if (ev.status === "failed") {
        audio.chime("error");
        showToast(`Stage '${ev.stage}' failed: ${ev.message || ""}`, "error", 6000);
      }
      if (currentState) updateStageRail(currentState);
    } else if (ev.type === "boss_alert") {
      engine.showBubble("boss", ev.message, 6000, "#f5a623");
      audio.chime("alert");
      showToast(ev.message, "info", 5000);
    }
    tickerLatest.textContent = ev.message || ev.label || `${ev.type}: ${ev.stage || ""}`;
  },
  onLogLines: (lines) => {
    currentLogs.push(...lines);
    if (currentLogs.length > 400) currentLogs = currentLogs.slice(-400);

    // Fail-safe log-derived progress parser for older/legacy runs
    for (const l of lines) {
      const matchRender = l.match(/Rendering scene (\d+)\/(\d+)/i) || l.match(/scene (\d+)\/(\d+)/i);
      if (matchRender && (!substageProgress.render || !substageProgress.render.total)) {
        substageProgress.render = {
          current: Number(matchRender[1]),
          total: Number(matchRender[2]),
          message: l,
        };
        if (currentState) updateStageRail(currentState);
      }
      const matchVoice = l.match(/Voice scene (\d+)\/(\d+)/i);
      if (matchVoice) {
        substageProgress.voice = {
          current: Number(matchVoice[1]),
          total: Number(matchVoice[2]),
          message: l,
        };
        if (currentState) updateStageRail(currentState);
      }
    }

    if (lines.length) tickerLatest.textContent = lines[lines.length - 1];
    drawers.setState(currentState, currentLogs);
  },
});

export function getOfficeToken() {
  const urlParams = new URLSearchParams(window.location.search);
  const token = urlParams.get("token") || sessionStorage.getItem("office_token");
  if (urlParams.get("token")) {
    sessionStorage.setItem("office_token", urlParams.get("token"));
  }
  return token || "";
}

function authFetch(url, options = {}) {
  const token = getOfficeToken();
  const headers = new Headers(options.headers || {});
  if (token && !headers.has("X-Office-Token")) {
    headers.set("X-Office-Token", token);
  }
  return fetch(url, { ...options, headers });
}

function updateHudFromState(state) {
  const manifest = state.manifest || {};
  const rk = (manifest.run_kind || "test").toUpperCase();
  const st = (manifest.status || "pending").toUpperCase();
  const ed = manifest.stages?.editor?.status || "pending";
  const mediaSt = manifest.stages?.media?.status || "pending";
  const qaSt = manifest.stages?.qa?.status || "pending";
  const busy = Boolean(state.active_process || state.active_job?.status === "running");
  const gateClosed = state.gate && state.gate.ok === false;
  const isFinished = manifest.status === "done" || Boolean(state.artifacts?.has_final_mp4);
  const isStandby = !busy && !inspectingHistoricalRun && isFinished;

  engine.setFreshStandbyMode(isStandby);
  updateStageRail(state, isStandby);

  liveDot.classList.toggle("running", busy);

  runKindBadge.textContent = rk;
  runKindBadge.className = `badge ${rk === "PRODUCTION" ? "badge-production" : "badge-test"}`;

  const catDisplay = state.category_display || (Array.isArray(state.selected_categories) && state.selected_categories.length ? state.selected_categories.join(", ") : "");
  if (categoryBadge) {
    if (catDisplay) {
      categoryBadge.textContent = `CATEGORY: ${catDisplay.toUpperCase()}`;
      categoryBadge.title = `Finished / Selected Category: ${catDisplay}`;
      categoryBadge.classList.remove("hidden");
    } else {
      categoryBadge.classList.add("hidden");
    }
  }

  if (isStandby) {
    overallStatusBadge.textContent = "STANDBY";
    overallStatusBadge.className = "badge badge-standby";
    overallStatusBadge.title = "No active run in progress. Agents standing by for fresh run.";
  } else {
    overallStatusBadge.textContent = busy ? "RUNNING" : st;
    overallStatusBadge.className = `badge badge-${busy ? "running" : manifest.status === "done" ? "done" : manifest.status === "paused" ? "paused" : manifest.status === "failed" ? "failed" : "neutral"}`;
  }

  // Context-aware primary button
  primaryActionBtn.disabled = false;
  primaryActionBtn.title = "";
  if (busy) {
    primaryActionBtn.textContent = "⏳ Pipeline Running...";
    primaryActionBtn.disabled = true;
  } else if (isStandby) {
    primaryActionBtn.textContent = "▶ Start Morning Prep";
    primaryActionBtn.onclick = () => drawers.openStation("boss", "boss");
  } else if (!state.exists || manifest.status === "pending") {
    primaryActionBtn.textContent = "▶ Start Morning Prep";
    primaryActionBtn.onclick = () => drawers.openStation("boss", "boss");
  } else if (ed === "prepared" || ed === "needs_revision") {
    primaryActionBtn.textContent = "✍ Draft Script";
    primaryActionBtn.onclick = () => drawers.openStation("boss", state.llm_writer?.configured ? "boss" : "manual");
  } else if (ed === "awaiting_approval") {
    primaryActionBtn.textContent = "✅ Review & Approve Script";
    primaryActionBtn.onclick = () => drawers.openStation("boss", "boss");
  } else if (ed === "done" && gateClosed && qaSt !== "done") {
    primaryActionBtn.textContent = "🔒 Gate Closed: Review Script";
    primaryActionBtn.disabled = true;
    primaryActionBtn.title = state.gate.reason || "Editor gate refusal";
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
    primaryActionBtn.textContent = `🎥 Watch Video (${catDisplay || "Final"})`;
    primaryActionBtn.onclick = () => drawers._openVideoModal("render/final.mp4", `Final Broadcast — ${catDisplay || "Done"} (${state.date})`);
  } else {
    primaryActionBtn.textContent = "👔 Boss Desk";
    primaryActionBtn.onclick = () => drawers.openStation("boss", "boss");
  }

  // Stop button visibility & action in HUD
  if (hudStopBtn) {
    hudStopBtn.classList.toggle("hidden", !busy);
    hudStopBtn.onclick = async () => {
      if (confirm(`Stop running pipeline for ${state.date}?`)) {
        await handleAction("stop");
      }
    };
  }

  // Next Action Banner
  if (nextActionBar && nextActionText && nextActionActions) {
    if (isStandby) {
      nextActionBar.classList.remove("hidden");
      nextActionText.innerHTML = `Previous broadcast finished: <strong>${escapeHtml(catDisplay || "Headlines")}</strong> (${state.date}). Agents standing by for new run.`;
      nextActionActions.innerHTML = "";

      if (state.artifacts?.has_final_mp4) {
        const watchBtn = document.createElement("button");
        watchBtn.className = "btn btn-primary btn-sm";
        watchBtn.textContent = `🎥 Watch Video (${catDisplay || "Final"})`;
        watchBtn.onclick = () => drawers._openVideoModal("render/final.mp4", `Final Broadcast — ${catDisplay || "Done"} (${state.date})`);
        nextActionActions.appendChild(watchBtn);
      }

      const resetBtn = document.createElement("button");
      resetBtn.className = "btn btn-warning btn-sm";
      resetBtn.textContent = "🔄 Reset for New Run";
      resetBtn.onclick = () => drawers.openStation("boss", "boss");
      nextActionActions.appendChild(resetBtn);

      const inspectBtn = document.createElement("button");
      inspectBtn.className = "btn btn-ghost btn-sm";
      inspectBtn.textContent = "🔍 Inspect Previous Run";
      inspectBtn.onclick = () => {
        inspectingHistoricalRun = true;
        updateHudFromState(state);
      };
      nextActionActions.appendChild(inspectBtn);
    } else if (inspectingHistoricalRun && isFinished && !busy) {
      nextActionBar.classList.remove("hidden");
      nextActionText.innerHTML = `Inspecting completed run: <strong>${escapeHtml(catDisplay || "Headlines")}</strong> (${state.date}).`;
      nextActionActions.innerHTML = "";

      const returnBtn = document.createElement("button");
      returnBtn.className = "btn btn-primary btn-sm";
      returnBtn.textContent = "↩ Return to Standby";
      returnBtn.onclick = () => {
        inspectingHistoricalRun = false;
        updateHudFromState(state);
      };
      nextActionActions.appendChild(returnBtn);

      if (state.artifacts?.has_final_mp4) {
        const watchBtn = document.createElement("button");
        watchBtn.className = "btn btn-sm";
        watchBtn.textContent = `🎥 Watch Video`;
        watchBtn.onclick = () => drawers._openVideoModal("render/final.mp4", `Final Broadcast — ${catDisplay || "Done"} (${state.date})`);
        nextActionActions.appendChild(watchBtn);
      }

      const resetBtn = document.createElement("button");
      resetBtn.className = "btn btn-warning btn-sm";
      resetBtn.textContent = "🔄 Reset for New Run";
      resetBtn.onclick = () => drawers.openStation("boss", "boss");
      nextActionActions.appendChild(resetBtn);
    } else if (state.next_action) {
      nextActionBar.classList.remove("hidden");
      nextActionText.textContent = state.next_action;
      nextActionActions.innerHTML = "";

      if (busy) {
        const stopBtn = document.createElement("button");
        stopBtn.className = "btn btn-danger btn-sm";
        stopBtn.textContent = "⏹ Stop Pipeline";
        stopBtn.onclick = () => {
          if (confirm(`Stop running pipeline for ${state.date}?`)) handleAction("stop");
        };
        nextActionActions.appendChild(stopBtn);
      } else if (ed === "awaiting_approval") {
        const revBtn = document.createElement("button");
        revBtn.className = "btn btn-primary btn-sm";
        revBtn.textContent = "Review Script";
        revBtn.onclick = () => drawers.openStation("boss", "boss");
        nextActionActions.appendChild(revBtn);
      } else if (ed === "done" && (mediaSt === "awaiting_approval" || mediaSt === "prepared") && state.artifacts?.has_picker_html) {
        const pickBtn = document.createElement("button");
        pickBtn.className = "btn btn-primary btn-sm";
        pickBtn.textContent = "Open Picker";
        pickBtn.onclick = () => drawers._openIframeModal("media/review/picker.html", `Media Shot Picker (${state.date})`);
        nextActionActions.appendChild(pickBtn);
      } else if (qaSt === "failed" || manifest.status === "failed") {
        const qaBtn = document.createElement("button");
        qaBtn.className = "btn btn-danger btn-sm";
        qaBtn.textContent = "Open QA Station";
        qaBtn.onclick = () => drawers.openStation("qa");
        nextActionActions.appendChild(qaBtn);
      } else if (manifest.status === "done" && state.artifacts?.has_final_mp4) {
        const watchBtn = document.createElement("button");
        watchBtn.className = "btn btn-primary btn-sm";
        watchBtn.textContent = `Watch Final Video (${catDisplay || "Completed"})`;
        watchBtn.onclick = () => drawers._openVideoModal("render/final.mp4", `Final Broadcast — ${catDisplay || "Done"} (${state.date})`);
        nextActionActions.appendChild(watchBtn);
      }
    } else {
      nextActionBar.classList.add("hidden");
    }
  }

  if (isStandby) {
    tickerLatest.textContent = `Standing by — previous ${catDisplay || 'run'} finished. Ready to start next run.`;
  } else if (state.next_action) {
    tickerLatest.textContent = state.next_action;
  } else if (currentLogs.length) {
    tickerLatest.textContent = currentLogs[currentLogs.length - 1];
  } else if (state.exists) {
    tickerLatest.textContent = `Loaded run ${state.date} (${st})`;
  } else {
    tickerLatest.textContent = `Run ${state.date} not started yet — click Start Morning Prep!`;
  }
}

async function handleAction(action, payload) {
  if (!currentDate) return;
  const activeBtn = document.activeElement;
  const prevText = activeBtn?.textContent;
  if (activeBtn && activeBtn.tagName === "BUTTON") {
    activeBtn.disabled = true;
    activeBtn.classList.add("btn-loading");
  }

  try {
    const res = await authFetch(`/api/runs/${encodeURIComponent(currentDate)}/${action}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload || {}),
    });
    const data = await res.json();
    if (!res.ok) {
      showToast(data.detail || `Action '${action}' failed.`, "error", 6000);
      return;
    }
    if (action === "stop") {
      showToast(data.message || "Pipeline stopped.", "info", 5000);
    } else if (action === "reset-stage") {
      showToast(`Stage reset complete.`, "info", 4000);
    } else if (action === "reset-run") {
      audio.chime("done");
      inspectingHistoricalRun = false;
      showToast("Agents reset to standby. Ready for new run.", "done", 5000);
      await loadRunsAndConnect();
      return;
    } else if (action === "draft-manual") {
      if (data.ok) {
        showToast("Draft passed validation! Ready to Approve.", "done", 5000);
        drawers.openStation("boss", "boss");
      } else {
        showToast("Validation failed — see error list in Manual Draft tab.", "error", 6000);
      }
    } else if (action === "approve") {
      audio.chime("done");
      if (data.job_id) {
        showToast("Script approved! Parallel prep (voice + visuals) started automatically.", "done", 5000);
      } else {
        showToast("Script approved! Ready to render broadcast.", "done", 5000);
      }
    } else {
      showToast(`Action '${action}' dispatched successfully.`, "done", 4000);
    }

    // Refresh snapshot immediately
    const stateRes = await authFetch(`/api/runs/${encodeURIComponent(currentDate)}/state`);
    if (stateRes.ok) {
      const st = await stateRes.json();
      currentState = st;
      engine.setState(st);
      drawers.setState(st, st.recent_logs || currentLogs);
      updateHudFromState(st);
    }
  } catch (err) {
    showToast("Request failed: " + err.message, "error", 6000);
  } finally {
    if (activeBtn && activeBtn.tagName === "BUTTON") {
      activeBtn.disabled = false;
      activeBtn.classList.remove("btn-loading");
      if (prevText) activeBtn.textContent = prevText;
    }
  }
}

async function loadRunsAndConnect() {
  const res = await authFetch("/api/runs");
  const data = await res.json();
  const runs = data.runs || [];

  runDateSelect.innerHTML = runs
    .map((r) => {
      const catLabel = r.category_display ? ` [${r.category_display}]` : "";
      return `<option value="${r.date}">${r.date}${catLabel} (${r.status})</option>`;
    })
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
  inspectingHistoricalRun = false;
  currentDate = runDateSelect.value;
  wsClient.connect(currentDate);
});

document.getElementById("replayBtn").addEventListener("click", async () => {
  if (!currentDate) return;
  const res = await authFetch(`/api/runs/${encodeURIComponent(currentDate)}/replay`);
  const data = await res.json();
  if (!data.events || !data.events.length) {
    showToast("No stage events recorded for this date yet.", "info", 4000);
    return;
  }
  const speed = Number(document.getElementById("replaySpeedSelect").value) || 3;
  replayBanner.classList.remove("hidden");
  replay.start(data.events, speed);
});

// Global Keyboard Shortcuts (Esc, 1-8, Space, ?)
window.addEventListener("keydown", (e) => {
  if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) {
    if (e.key === "Escape") document.activeElement.blur();
    return;
  }

  if (e.key === "Escape") {
    drawers.closeModal();
    drawers.close();
  } else if (e.key >= "1" && e.key <= "8") {
    const idx = parseInt(e.key, 10) - 1;
    if (idx >= 0 && idx < STAGE_ORDER.length) {
      drawers.openStation(STAGE_ORDER[idx]);
    }
  } else if (e.key === "?") {
    drawers.openModal(
      "⌨ Keyboard Shortcuts",
      `
      <div style="font-size:13px; line-height:2.0; padding:10px;">
        <div><kbd style="background:#1e293b; padding:2px 6px; border-radius:4px; border:1px solid #475569;">1</kbd> - <kbd style="background:#1e293b; padding:2px 6px; border-radius:4px; border:1px solid #475569;">8</kbd> : Open Station drawer (1=Scraper ... 8=QA)</div>
        <div><kbd style="background:#1e293b; padding:2px 6px; border-radius:4px; border:1px solid #475569;">Esc</kbd> : Close open drawer or modal</div>
        <div><kbd style="background:#1e293b; padding:2px 6px; border-radius:4px; border:1px solid #475569;">Space</kbd> : Toggle audio chimes on/off</div>
        <div><kbd style="background:#1e293b; padding:2px 6px; border-radius:4px; border:1px solid #475569;">?</kbd> : Show keyboard shortcuts guide</div>
      </div>
      `
    );
  } else if (e.key === " " && !e.target.matches("button")) {
    e.preventDefault();
    const on = audio.toggle();
    const btn = document.getElementById("soundToggleBtn");
    if (btn) btn.textContent = on ? "🔊 Sound On" : "🔇 Sound Off";
    showToast(on ? "Sound enabled" : "Sound muted", "info", 2000);
  }
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
      const stateRes = await authFetch(`/api/runs/${encodeURIComponent(currentDate)}/state`);
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
  drawers.openModal("System Maintenance & Diagnostics", `<p class="muted">Loading system diagnostics & maintenance stats...</p>`);
  try {
    const res = await authFetch("/api/system/maintenance");
    const data = await res.json();
    const doc = data.doctor || {};
    const disk = data.disk || {};
    const mirrors = data.mirrors || {};
    const services = data.services || {};

    const checkRows = (doc.checks || [])
      .map(
        (c) => `
        <div class="artifact-item" style="padding:6px 0; border-bottom:1px solid #1a2a44;">
          <div>
            <strong>${escapeHtml(c.name)}</strong>: <span style="color:#94a3b8">${escapeHtml(c.detail || "")}</span>
            ${c.fix_hint ? `<div style="color:#f5a623; font-size:11px; margin-top:2px;">Tip: ${escapeHtml(c.fix_hint)}</div>` : ""}
          </div>
          <span class="badge ${c.status === "PASS" ? "badge-done" : c.status === "WARN" ? "badge-paused" : "badge-failed"}">${c.status}</span>
        </div>`
      )
      .join("");

    const runDiskRows = (disk.runs || [])
      .map(
        (r) => `
        <tr>
          <td style="padding:4px 8px; font-family:monospace;">${r.date}</td>
          <td style="padding:4px 8px;">${r.size_mb} MB</td>
          <td style="padding:4px 8px;"><span class="badge badge-${r.status === "done" ? "done" : r.status === "failed" ? "failed" : "neutral"}">${r.status}</span></td>
          <td style="padding:4px 8px;">${r.has_final_mp4 ? "✅ MP4" : "—"}</td>
        </tr>`
      )
      .join("");

    const html = `
      <div style="display:flex; flex-direction:column; gap:16px;">
        <!-- Doctor Readiness -->
        <section>
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <h4 style="margin:0; font-size:14px; color:#e2e8f0;">🩺 System Readiness Checks</h4>
            <span class="badge ${doc.ok ? "badge-done" : "badge-failed"}">${doc.ok ? "ALL PASS" : "ATTENTION NEEDED"}</span>
          </div>
          <div class="artifact-list" style="max-height:180px; overflow-y:auto; background:#0b1322; padding:8px 12px; border-radius:6px;">
            ${checkRows}
          </div>
        </section>

        <!-- Disk Usage & Storage -->
        <section>
          <h4 style="margin:0 0 8px 0; font-size:14px; color:#e2e8f0;">💾 Storage & Run Breakdown (${disk.free_gb || 0} GB Free of ${disk.total_gb || 0} GB)</h4>
          <div style="max-height:160px; overflow-y:auto; background:#0b1322; padding:6px; border-radius:6px;">
            <table style="width:100%; font-size:12px; border-collapse:collapse;">
              <thead>
                <tr style="color:#64748b; border-bottom:1px solid #1a2a44; text-align:left;">
                  <th style="padding:4px 8px;">Run Date</th>
                  <th style="padding:4px 8px;">Disk Size</th>
                  <th style="padding:4px 8px;">Status</th>
                  <th style="padding:4px 8px;">Deliverable</th>
                </tr>
              </thead>
              <tbody>${runDiskRows || `<tr><td colspan="4" style="padding:8px; text-align:center;" class="muted">No runs found</td></tr>`}</tbody>
            </table>
          </div>
        </section>

        <!-- Delivery & Services -->
        <section style="display:grid; grid-template-columns:1fr 1fr; gap:10px;">
          <div style="background:#0b1322; padding:10px; border-radius:6px; font-size:12px;">
            <div style="font-weight:600; color:#cbd5e1; margin-bottom:6px;">🌐 Mirror Delivery</div>
            <div>OneDrive: ${mirrors.onedrive?.exists ? `<span class="badge badge-done">Synced</span>` : `<span class="badge badge-neutral">Not Mounted</span>`}</div>
            <div style="margin-top:4px;">Local Latest: <span class="badge badge-${mirrors.latest?.has_final_mp4 ? "done" : "neutral"}">${mirrors.latest?.has_final_mp4 ? "final.mp4 ready" : "Empty"}</span></div>
          </div>
          <div style="background:#0b1322; padding:10px; border-radius:6px; font-size:12px;">
            <div style="font-weight:600; color:#cbd5e1; margin-bottom:6px;">⚙️ Services & Daemons</div>
            <div>Ming TTS Daemon: ${services.tts_daemon ? "✅ Online" : "❌ Offline / Not Started"}</div>
            <div style="margin-top:4px;">Win Scheduler: <span class="badge badge-neutral">${services.windows_scheduler || "Unknown"}</span></div>
          </div>
        </section>

        <!-- Cleanup Tools -->
        <section style="background:#0b1322; padding:12px; border-radius:6px;">
          <h4 style="margin:0 0 8px 0; font-size:14px; color:#e2e8f0;">🧹 Storage Cleanup Tool</h4>
          <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
            <label style="font-size:12px; color:#94a3b8;">
              Keep last
              <input type="number" id="cleanupKeepDays" value="7" min="1" max="60" style="width:50px; padding:3px 6px; background:#101e38; border:1px solid #25436e; color:#fff; border-radius:4px; margin:0 4px;" />
              runs
            </label>
            <button class="btn btn-sm btn-ghost" id="btnCleanupDryRun">🔍 Preview (Dry Run)</button>
            <button class="btn btn-sm btn-danger" id="btnCleanupExecute">🗑 Execute Cleanup</button>
          </div>
          <pre id="cleanupOutput" class="hidden" style="margin-top:10px; padding:8px; background:#060d19; border:1px solid #1e3a64; font-size:11px; color:#38bdf8; max-height:140px; overflow-y:auto; border-radius:4px; white-space:pre-wrap;"></pre>
        </section>
      </div>
    `;

    drawers.openModal("🛠 System Maintenance & Health", html);

    const outEl = document.getElementById("cleanupOutput");
    const keepInput = document.getElementById("cleanupKeepDays");

    document.getElementById("btnCleanupDryRun")?.addEventListener("click", async () => {
      const keepRuns = Number(keepInput.value) || 7;
      outEl.classList.remove("hidden");
      outEl.textContent = "Calculating cleanup plan...";
      try {
        const cRes = await authFetch("/api/system/cleanup", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ dry_run: true, keep_runs: keepRuns }),
        });
        const cData = await cRes.json();
        outEl.textContent = cData.formatted_plan || JSON.stringify(cData, null, 2);
      } catch (err) {
        outEl.textContent = "Preview error: " + err.message;
      }
    });

    document.getElementById("btnCleanupExecute")?.addEventListener("click", async () => {
      const keepRuns = Number(keepInput.value) || 7;
      if (!confirm(`Are you sure you want to permanently delete run targets and cache older than ${keepRuns} runs?`)) return;
      outEl.classList.remove("hidden");
      outEl.textContent = "Executing cleanup...";
      try {
        const cRes = await authFetch("/api/system/cleanup", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ dry_run: false, keep_runs: keepRuns }),
        });
        const cData = await cRes.json();
        const deletedCount = (cData.deleted_paths && Array.isArray(cData.deleted_paths)) ? cData.deleted_paths.length : (cData.deleted_runs_count || 0);
        const freedMb = cData.freed_run_mb != null ? cData.freed_run_mb : Math.round((cData.freed_bytes || 0) / (1024 * 1024));
        outEl.textContent = `Cleanup Complete!\nDeleted ${deletedCount} path(s), freed ${freedMb} MB.\n` + (cData.formatted_plan || "");
        showToast("Storage cleanup completed successfully.", "done", 5000);
      } catch (err) {
        outEl.textContent = "Cleanup execution error: " + err.message;
      }
    });
  } catch (err) {
    drawers.openModal("Maintenance Error", `<p class="badge-failed">Failed loading maintenance stats: ${escapeHtml(err.message)}</p>`);
  }
});

loadRunsAndConnect();
