/**
 * Slide-out Inspector, Boss Office Controls, Modals, and WebAudio Chimes.
 */
import { STATION_BY_ID } from "./office_map.js";

export class OfficeAudio {
  constructor() {
    this.enabled = false;
    this.ctx = null;
  }

  toggle() {
    this.enabled = !this.enabled;
    if (this.enabled && !this.ctx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) this.ctx = new AudioCtx();
    }
    return this.enabled;
  }

  chime(kind = "done") {
    if (!this.enabled || !this.ctx) return;
    try {
      const now = this.ctx.currentTime;
      const osc = this.ctx.createOscillator();
      const gain = this.ctx.createGain();
      osc.connect(gain);
      gain.connect(this.ctx.destination);

      if (kind === "alert") {
        osc.frequency.setValueAtTime(523.25, now);
        osc.frequency.setValueAtTime(659.25, now + 0.12);
      } else if (kind === "error") {
        osc.frequency.setValueAtTime(260, now);
        osc.frequency.setValueAtTime(196, now + 0.15);
      } else {
        osc.frequency.setValueAtTime(587.33, now);
        osc.frequency.setValueAtTime(880, now + 0.1);
      }
      gain.gain.setValueAtTime(0.06, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.3);
      osc.start(now);
      osc.stop(now + 0.32);
    } catch {
      // Ignore audio errors
    }
  }
}

function escapeHtml(str) {
  return String(str ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

export class DrawerManager {
  constructor({ drawerEl, titleEl, subtitleEl, tabsEl, bodyEl, modalOverlay, modalTitle, modalBody, onAction }) {
    this.drawerEl = drawerEl;
    this.titleEl = titleEl;
    this.subtitleEl = subtitleEl;
    this.tabsEl = tabsEl;
    this.bodyEl = bodyEl;
    this.modalOverlay = modalOverlay;
    this.modalTitle = modalTitle;
    this.modalBody = modalBody;
    this.onAction = onAction;

    this.currentStationId = null;
    this.currentTab = "overview";
    this.state = null;
    this.logs = [];
  }

  setState(state, logs) {
    this.state = state;
    this.logs = logs || [];
    if (!this.drawerEl.classList.contains("hidden") && this.currentStationId) {
      // Avoid blowing away textarea edits if user is typing manual draft
      const activeTag = document.activeElement?.tagName;
      if (activeTag === "TEXTAREA" || activeTag === "INPUT") return;
      this.render();
    }
  }

  openStation(stationId, defaultTab = null) {
    this.currentStationId = stationId;
    this.currentTab = defaultTab || (stationId === "boss" ? "boss" : "overview");
    this.drawerEl.classList.remove("hidden");
    this.render();
  }

  close() {
    this.drawerEl.classList.add("hidden");
  }

  openModal(title, htmlContent) {
    this.modalTitle.textContent = title;
    this.modalBody.innerHTML = htmlContent;
    this.modalOverlay.classList.remove("hidden");
  }

  closeModal() {
    this.modalOverlay.classList.add("hidden");
    this.modalBody.innerHTML = "";
  }

  render() {
    if (!this.state) return;
    const st = STATION_BY_ID[this.currentStationId] || STATION_BY_ID.boss;
    this.titleEl.textContent = `${st.title} — ${st.agentName}`;
    this.subtitleEl.textContent = st.role;

    const tabs =
      st.id === "boss"
        ? [
            { id: "boss", label: "Boss Controls" },
            { id: "manual", label: "Manual Draft" },
            { id: "logs", label: "Run Logs" },
          ]
        : [
            { id: "overview", label: "Status & Metrics" },
            { id: "artifacts", label: "Files & Previews" },
            { id: "logs", label: "Logs" },
          ];

    if (!tabs.some((t) => t.id === this.currentTab)) {
      this.currentTab = tabs[0].id;
    }

    this.tabsEl.innerHTML = tabs
      .map(
        (t) =>
          `<button class="tab-btn ${t.id === this.currentTab ? "active" : ""}" data-tab="${t.id}">${t.label}</button>`
      )
      .join("");

    this.tabsEl.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        this.currentTab = btn.dataset.tab;
        this.render();
      });
    });

    if (this.currentTab === "boss") this._renderBossTab();
    else if (this.currentTab === "manual") this._renderManualDraftTab();
    else if (this.currentTab === "overview") this._renderOverviewTab(st);
    else if (this.currentTab === "artifacts") this._renderArtifactsTab(st);
    else if (this.currentTab === "logs") this._renderLogsTab(st);
  }

  _renderBossTab() {
    const d = this.state.date;
    const manifest = this.state.manifest || {};
    const stages = manifest.stages || {};
    const edStatus = stages.editor?.status || "pending";
    const mediaStatus = stages.media?.status || "pending";
    const qaStatus = stages.qa?.status || "pending";
    const busy = this.state.active_process || this.state.active_job?.status === "running";
    const llm = this.state.llm_writer || {};
    const arts = this.state.artifacts || {};
    const mediaSum = this.state.media_summary || null;

    let html = `
      <div class="card">
        <h4>1. Morning Prep & Run Mode</h4>
        <p class="muted" style="margin-top:0">Target date: <strong>${escapeHtml(d)}</strong> • Editor status: <strong>${escapeHtml(edStatus)}</strong></p>
        <div class="btn-row">
          <select id="morningKindSelect">
            <option value="test" ${manifest.run_kind !== "production" ? "selected" : ""}>Test Run (TEST RUN badge)</option>
            <option value="production">Production Broadcast</option>
          </select>
          <button class="btn btn-primary" id="btnMorning" ${busy ? "disabled" : ""}>▶ Start Morning Prep</button>
        </div>
      </div>
    `;

    html += `
      <div class="card">
        <h4>2. Script Writing (${escapeHtml(edStatus)})</h4>
        ${
          llm.configured
            ? `<p class="muted" style="margin-top:0">AI Writer ready: <strong>${escapeHtml(llm.provider)} / ${escapeHtml(llm.model)}</strong></p>`
            : `<p class="muted" style="margin-top:0">No API key in <code>.env</code> yet (set <code>ZHIPU_API_KEY</code>, <code>GEMINI_API_KEY</code>, <code>ANTHROPIC_API_KEY</code>, <code>OPENAI_API_KEY</code>, or <code>OLLAMA_MODEL</code>) — or paste JSON in the <strong>Manual Draft</strong> tab.</p>`
        }
        <div class="btn-row">
          <button class="btn btn-primary" id="btnDraftAi" ${busy || !arts.has_editor_brief || !llm.configured ? "disabled" : ""}>✨ Auto-Draft Script with AI</button>
          ${arts.has_editor_brief ? `<button class="btn btn-sm" id="btnViewBrief">📄 Read editor_brief.md</button>` : ""}
        </div>
      </div>
    `;

    html += `
      <div class="card">
        <h4>3. Boss Sign-Off Gate (${edStatus === "done" ? "APPROVED ✓" : edStatus === "awaiting_approval" ? "READY FOR REVIEW" : "Waiting on script"})</h4>
        ${
          this.state.script_preview_md
            ? `<div class="preview-md">${escapeHtml(this.state.script_preview_md)}</div>`
            : `<p class="muted">No validated <code>script_preview.md</code> yet.</p>`
        }
        <div class="btn-row">
          <button class="btn btn-success" id="btnApprove" ${busy || edStatus !== "awaiting_approval" ? "disabled" : ""}>✅ Approve Script</button>
        </div>
      </div>
    `;

    const voiceStatus = stages.voice?.status || "pending";
    const timelineStatus = stages.timeline?.status || "pending";
    const mediaSummaryText = mediaSum
      ? `${mediaSum.auto_clips_count} auto-matched clip(s), ${mediaSum.auto_cards_count} card(s) across ${mediaSum.total_shots} shots`
      : `Media stage: ${mediaStatus}`;

    // Dual-track status badges
    const audioBadge = voiceStatus === "done" && timelineStatus === "done"
      ? `<span style="background:#0d9488;color:#fff;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;">Audio: Synced ✓</span>`
      : (voiceStatus === "running" || timelineStatus === "running")
      ? `<span style="background:#db2777;color:#fff;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;">Audio: Synthesizing ⏳</span>`
      : `<span style="background:#334155;color:#94a3b8;padding:2px 8px;border-radius:10px;font-size:11px;">Audio: ${escapeHtml(voiceStatus)}</span>`;

    const visualBadge = mediaStatus === "done"
      ? `<span style="background:#16a34a;color:#fff;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;">Visuals: Done ✓</span>`
      : mediaStatus === "awaiting_approval"
      ? `<span style="background:#22c55e;color:#06101f;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:700;">Visuals: Picker Ready 🎨</span>`
      : mediaStatus === "running"
      ? `<span style="background:#0284c7;color:#fff;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;">Visuals: Searching ⏳</span>`
      : `<span style="background:#334155;color:#94a3b8;padding:2px 8px;border-radius:10px;font-size:11px;">Visuals: ${escapeHtml(mediaStatus)}</span>`;

    html += `
      <div class="card">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
          <h4 style="margin:0;">4. Visual Selection & Render</h4>
          <div style="display:flex;gap:6px;">${audioBadge} ${visualBadge}</div>
        </div>
        <p class="muted" style="margin-top:0">${edStatus !== "done" ? "Approve script first to start parallel audio synthesis and visual candidate discovery." : escapeHtml(mediaSummaryText)}</p>
        <div class="btn-row">
          ${
            arts.has_picker_html
              ? `<button class="btn ${mediaStatus === "awaiting_approval" ? "btn-primary" : ""}" id="btnOpenPicker" ${busy ? "disabled" : ""}>🎨 Open Visual Picker & Render</button>
                 <button class="btn btn-sm" id="btnPrepareMediaForce" ${busy || edStatus !== "done" ? "disabled" : ""}>↻ Re-scan Visuals</button>`
              : `<button class="btn btn-primary" id="btnPrepareMedia" ${busy || edStatus !== "done" ? "disabled" : ""}>⚡ Parallel Prep (Voice + Picker)</button>`
          }
          <button class="btn btn-sm" id="btnFinish" ${busy || edStatus !== "done" ? "disabled" : ""} title="Skip manual picking and auto-render full broadcast">⚡ Auto-Finish (--finish)</button>
        </div>
      </div>
    `;

    if (arts.has_final_mp4 || qaStatus === "done") {
      html += `
        <div class="card">
          <h4>5. Final Broadcast Output</h4>
          <div class="btn-row">
            ${arts.has_final_mp4 ? `<button class="btn btn-primary" id="btnWatchVideo">🎥 Watch final.mp4</button>` : ""}
            ${arts.has_storyboard ? `<button class="btn" id="btnOpenStoryboard">🖼 Storyboard</button>` : ""}
            ${arts.has_description_md ? `<button class="btn" id="btnCopyDesc">📋 Description</button>` : ""}
          </div>
          ${this.state.mirror_dir ? `<p class="muted" style="margin-bottom:0;font-size:11px">OneDrive Mirror: <code>${escapeHtml(this.state.mirror_dir)}</code></p>` : ""}
        </div>
      `;
    }

    this.bodyEl.innerHTML = html;

    this.bodyEl.querySelector("#btnMorning")?.addEventListener("click", () => {
      const rk = this.bodyEl.querySelector("#morningKindSelect").value;
      if (rk === "production" && !confirm("Start a PRODUCTION run (omits TEST RUN watermark)?")) return;
      this.onAction("morning", { run_kind: rk });
    });
    this.bodyEl.querySelector("#btnDraftAi")?.addEventListener("click", () => this.onAction("draft-ai", {}));
    this.bodyEl.querySelector("#btnApprove")?.addEventListener("click", () => this.onAction("approve", {}));
    this.bodyEl.querySelector("#btnPrepareMedia")?.addEventListener("click", () => this.onAction("prepare-media", { force: false }));
    this.bodyEl.querySelector("#btnPrepareMediaForce")?.addEventListener("click", () => {
      if (confirm("Re-scan online visual candidates for all shots?")) {
        this.onAction("prepare-media", { force: true });
      }
    });
    this.bodyEl.querySelector("#btnFinish")?.addEventListener("click", () => this.onAction("finish", {}));
    this.bodyEl.querySelector("#btnViewBrief")?.addEventListener("click", () => this._previewRunFile("script/editor_brief.md", "Editor Brief"));
    this.bodyEl.querySelector("#btnWatchVideo")?.addEventListener("click", () => this._openVideoModal("render/final.mp4", `Final Broadcast (${d})`));
    this.bodyEl.querySelector("#btnOpenStoryboard")?.addEventListener("click", () => this._openIframeModal("qa/storyboard.html", `QA Storyboard (${d})`));
    this.bodyEl.querySelector("#btnOpenPicker")?.addEventListener("click", () => this._openIframeModal("media/review/picker.html", `Media Shot Picker (${d})`));
    this.bodyEl.querySelector("#btnCopyDesc")?.addEventListener("click", () => {
      this.openModal(`YouTube Description (${d})`, `<div class="preview-md">${escapeHtml(this.state.description_md)}</div>`);
    });
  }

  _renderManualDraftTab() {
    const valRep = this.state.validation_report;
    const errs = valRep?.errors || [];
    this.bodyEl.innerHTML = `
      <div class="card">
        <h4>Paste draft_script.json & Validate</h4>
        <p class="muted" style="margin-top:0">Paste valid <code>draft_script.json</code> below to run deterministic validation outside Antigravity.</p>
        <textarea id="manualDraftJson" class="code-input" placeholder='{"language": "English", "origin": "editor", "scenes": [...]}'></textarea>
        <p class="muted" style="margin:8px 0 4px">Optional <code>selection.json</code>:</p>
        <textarea id="manualSelectionJson" class="code-input" style="min-height:90px" placeholder='{"selected_clusters": [], "skipped_clusters": []}'></textarea>
        <div class="btn-row">
          <button class="btn btn-primary" id="btnValidateManual">✔ Save & Validate Draft</button>
        </div>
      </div>
      ${
        errs.length
          ? `<div class="card"><h4>Latest Validation Errors (${errs.length})</h4><ul>${errs.map((e) => `<li>${escapeHtml(e)}</li>`).join("")}</ul></div>`
          : ""
      }
    `;

    this.bodyEl.querySelector("#btnValidateManual")?.addEventListener("click", () => {
      try {
        const draftObj = JSON.parse(this.bodyEl.querySelector("#manualDraftJson").value);
        const selRaw = this.bodyEl.querySelector("#manualSelectionJson").value.trim();
        const selObj = selRaw ? JSON.parse(selRaw) : null;
        this.onAction("draft-manual", { draft_script: draftObj, selection: selObj });
      } catch (err) {
        alert("Invalid JSON: " + err.message);
      }
    });
  }

  _renderOverviewTab(st) {
    const stages = this.state.manifest?.stages || {};
    const meta = st.stageKey ? stages[st.stageKey] : null;
    if (!meta) {
      this.bodyEl.innerHTML = `
        <div class="card">
          <h4>${escapeHtml(st.title)}</h4>
          <p>Overall Run Status: <strong>${escapeHtml(this.state.manifest?.status || "pending")}</strong></p>
          <div class="preview-md">${escapeHtml(this.state.status_md || "Run not initialized yet.")}</div>
        </div>
      `;
      return;
    }

    const metrics = meta.metrics ? JSON.stringify(meta.metrics, null, 2) : "None yet";
    this.bodyEl.innerHTML = `
      <div class="card">
        <h4>Stage Status: ${escapeHtml(meta.status)}</h4>
        <div class="kv-grid">
          <span class="kv-label">Start Time</span><span>${escapeHtml(meta.start_time || "-")}</span>
          <span class="kv-label">End Time</span><span>${escapeHtml(meta.end_time || "-")}</span>
          <span class="kv-label">Retries</span><span>${meta.retry_count ?? 0} / ${meta.max_retries ?? 2}</span>
          <span class="kv-label">Error</span><span style="color:#fca5a5">${escapeHtml(meta.error_message || "None")}</span>
        </div>
        <div class="btn-row" style="margin-top:12px">
          <button class="btn btn-danger btn-sm" id="btnResetStage">↺ Reset Stage (${escapeHtml(st.stageKey)})</button>
        </div>
      </div>
      <div class="card">
        <h4>Metrics</h4>
        <div class="log-box" style="max-height:220px">${escapeHtml(metrics)}</div>
      </div>
    `;

    this.bodyEl.querySelector("#btnResetStage")?.addEventListener("click", () => {
      if (confirm(`Reset stage '${st.stageKey}' back to pending?`)) {
        this.onAction("reset-stage", { stage: st.stageKey });
      }
    });
  }

  _renderArtifactsTab(st) {
    const stages = this.state.manifest?.stages || {};
    const meta = st.stageKey ? stages[st.stageKey] : null;
    const files = [...(meta?.input_files || []).map((f) => ({ kind: "IN", path: f })), ...(meta?.output_files || []).map((f) => ({ kind: "OUT", path: f }))];

    if (st.id === "publish") {
      files.push({ kind: "OUT", path: "render/final.mp4" }, { kind: "OUT", path: "qa/storyboard.html" }, { kind: "OUT", path: "render/description.md" });
    }

    if (!files.length) {
      this.bodyEl.innerHTML = `<p class="muted">No stage files recorded yet.</p>`;
      return;
    }

    this.bodyEl.innerHTML = `
      <div class="card">
        <h4>Inputs & Outputs</h4>
        <div class="artifact-list">
          ${files
            .map(
              (f, i) => `
            <div class="artifact-item">
              <span><strong class="muted">[${f.kind}]</strong> <code>${escapeHtml(f.path)}</code></span>
              <button class="btn btn-xs" data-file-idx="${i}">Open</button>
            </div>`
            )
            .join("")}
        </div>
      </div>
    `;

    this.bodyEl.querySelectorAll("[data-file-idx]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const path = files[Number(btn.dataset.fileIdx)].path;
        if (path.endsWith(".mp4")) this._openVideoModal(path, path);
        else if (path.endsWith(".html")) this._openIframeModal(path, path);
        else if (path.endsWith(".wav")) this._openAudioModal(path, path);
        else if (path.endsWith(".jpg") || path.endsWith(".png")) this._openImageModal(path, path);
        else this._previewRunFile(path, path);
      });
    });
  }

  _renderLogsTab(st) {
    const key = st.stageKey ? st.stageKey.toLowerCase() : "";
    const filtered = key
      ? this.logs.filter((l) => l.toLowerCase().includes(key) || l.toLowerCase().includes("orchestrator"))
      : this.logs;
    const display = (filtered.length ? filtered : this.logs).slice(-120).join("\n");
    this.bodyEl.innerHTML = `
      <div class="card">
        <h4>Live Run Log (${escapeHtml(this.state.date)})</h4>
        <div class="log-box">${escapeHtml(display || "No log lines recorded yet.")}</div>
      </div>
    `;
  }

  async _previewRunFile(relPath, title) {
    const url = `/api/runs/${encodeURIComponent(this.state.date)}/file?path=${encodeURIComponent(relPath)}`;
    const res = await fetch(url);
    if (!res.ok) {
      alert(`File not available yet: ${relPath}`);
      return;
    }
    const text = await res.text();
    this.openModal(title, `<div class="preview-md" style="max-height:70vh">${escapeHtml(text)}</div>`);
  }

  _openVideoModal(relPath, title) {
    const url = `/api/runs/${encodeURIComponent(this.state.date)}/file?path=${encodeURIComponent(relPath)}`;
    this.openModal(title, `<video controls autoplay src="${url}"></video>`);
  }

  _openAudioModal(relPath, title) {
    const url = `/api/runs/${encodeURIComponent(this.state.date)}/file?path=${encodeURIComponent(relPath)}`;
    this.openModal(title, `<audio controls autoplay style="width:100%" src="${url}"></audio>`);
  }

  _openImageModal(relPath, title) {
    const url = `/api/runs/${encodeURIComponent(this.state.date)}/file?path=${encodeURIComponent(relPath)}`;
    this.openModal(title, `<img src="${url}" style="max-width:100%;border-radius:8px" />`);
  }

  _openIframeModal(relPath, title) {
    const url = `/api/runs/${encodeURIComponent(this.state.date)}/file?path=${encodeURIComponent(relPath)}`;
    this.openModal(title, `<iframe src="${url}"></iframe>`);
  }
}
