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
    this.isLogPaused = false;
    this.logSearchTerm = "";
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

    const availableCategories = Array.isArray(this.state.available_categories) && this.state.available_categories.length
      ? this.state.available_categories
      : [
          { key: "headlines", label: "Headlines & Breaking News" },
          { key: "politics", label: "National Politics" },
          { key: "society_local", label: "Society & Local News" },
          { key: "business_economics", label: "Business & Economics" },
          { key: "health", label: "Health & Medicine" },
          { key: "culture_entertainment", label: "Culture & Entertainment" },
          { key: "science_technology", label: "Science & Technology" },
          { key: "sports", label: "Sports" },
          { key: "international", label: "International News" },
        ];

    let savedCats = ["headlines"];
    try {
      const raw = localStorage.getItem("nnb_morning_categories");
      if (raw) {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed) && parsed.length) savedCats = parsed;
      }
    } catch (_) {}

    const selectedSet = new Set(savedCats);
    const runScrapedCats =
      manifest.selected_categories ||
      stages.scraper?.metrics?.selected_categories ||
      null;

    const categoryChipsHtml = availableCategories
      .map((cat) => {
        const checked = selectedSet.has(cat.key) || selectedSet.has("all");
        return `
          <label class="category-chip ${checked ? "selected" : ""}" data-cat-chip="${escapeHtml(cat.key)}">
            <input type="checkbox" class="cat-checkbox" value="${escapeHtml(cat.key)}" ${checked ? "checked" : ""} ${busy ? "disabled" : ""} />
            <span>${escapeHtml(cat.label)}</span>
          </label>
        `;
      })
      .join("");

    let html = `
      <div class="card">
        <h4>1. Morning Prep & Run Mode</h4>
        <p class="muted" style="margin-top:0">Target date: <strong>${escapeHtml(d)}</strong> • Editor status: <strong>${escapeHtml(edStatus)}</strong></p>
        ${
          Array.isArray(runScrapedCats) && runScrapedCats.length
            ? `<p class="muted" style="margin:4px 0 8px;font-size:11px">Last scraped categories: <code>${escapeHtml(runScrapedCats.join(", "))}</code></p>`
            : ""
        }
        <div class="category-picker-wrap">
          <div class="category-picker-header">
            <span><strong>Categories to Scrape</strong> (auto-fallback: Headlines)</span>
            <div class="category-picker-actions">
              <button type="button" class="btn btn-xs btn-ghost" id="btnCatHeadlines" ${busy ? "disabled" : ""}>Headlines Only</button>
              <button type="button" class="btn btn-xs btn-ghost" id="btnCatAll" ${busy ? "disabled" : ""}>Select All</button>
            </div>
          </div>
          <div class="category-grid" id="morningCategoryGrid">
            ${categoryChipsHtml}
          </div>
        </div>
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
        <div class="btn-row" style="align-items:center;">
          <button class="btn btn-primary" id="btnDraftAi" ${busy || !arts.has_editor_brief || !llm.configured ? "disabled" : ""}>✨ Auto-Draft Script with AI</button>
          ${arts.has_editor_brief ? `<button class="btn btn-sm" id="btnViewBrief">📄 Read editor_brief.md</button>` : ""}
          <label class="inline-check" title="Automatically approve if validation passes with zero errors">
            <input type="checkbox" id="chkAutoApproveAi" ${busy ? "disabled" : ""} />
            <span>Auto-approve on pass</span>
          </label>
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

    const gate = this.state.gate || { ok: edStatus === "done", reason: "" };
    const gateClosed = !gate.ok;

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
        ${
          gateClosed
            ? `<div class="gate-warning" style="background:rgba(239, 68, 68, 0.15);border:1px solid #ef4444;border-radius:6px;padding:8px 12px;margin:8px 0 12px;font-size:12px;color:#fca5a5;">
                <strong>🔒 Gate Closed:</strong> ${escapeHtml(gate.reason)}
               </div>`
            : ""
        }
        <p class="muted" style="margin-top:0">${edStatus !== "done" ? "Approve script first to start parallel audio synthesis and visual candidate discovery." : escapeHtml(mediaSummaryText)}</p>
        <div class="btn-row">
          ${
            arts.has_picker_html
              ? `<button class="btn ${mediaStatus === "awaiting_approval" ? "btn-primary" : ""}" id="btnOpenPicker" ${busy ? "disabled" : ""}>🎨 Open Visual Picker & Render</button>
                 <button class="btn btn-sm" id="btnPrepareMediaForce" ${busy || gateClosed ? "disabled" : ""} title="${escapeHtml(gateClosed ? gate.reason : 'Re-scan online visual candidates')}">↻ Re-scan Visuals</button>`
              : `<button class="btn btn-primary" id="btnPrepareMedia" ${busy || gateClosed ? "disabled" : ""} title="${escapeHtml(gateClosed ? gate.reason : 'Parallel Prep')}">⚡ Parallel Prep (Voice + Picker)</button>`
          }
          <button class="btn btn-sm" id="btnFinish" ${busy || gateClosed ? "disabled" : ""} title="${escapeHtml(gateClosed ? gate.reason : 'Skip manual picking and auto-render full broadcast')}">⚡ Auto-Finish (--finish)</button>
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

    const syncCategoryChips = () => {
      const chosen = [];
      this.bodyEl.querySelectorAll(".cat-checkbox").forEach((cb) => {
        const chip = cb.closest(".category-chip");
        if (chip) chip.classList.toggle("selected", cb.checked);
        if (cb.checked) chosen.push(cb.value);
      });
      try {
        localStorage.setItem("nnb_morning_categories", JSON.stringify(chosen.length ? chosen : ["headlines"]));
      } catch (_) {}
      return chosen;
    };

    this.bodyEl.querySelectorAll(".cat-checkbox").forEach((cb) => {
      cb.addEventListener("change", syncCategoryChips);
    });
    this.bodyEl.querySelector("#btnCatHeadlines")?.addEventListener("click", () => {
      this.bodyEl.querySelectorAll(".cat-checkbox").forEach((cb) => {
        cb.checked = cb.value === "headlines";
      });
      syncCategoryChips();
    });
    this.bodyEl.querySelector("#btnCatAll")?.addEventListener("click", () => {
      this.bodyEl.querySelectorAll(".cat-checkbox").forEach((cb) => {
        cb.checked = true;
      });
      syncCategoryChips();
    });

    this.bodyEl.querySelector("#btnMorning")?.addEventListener("click", () => {
      const rk = this.bodyEl.querySelector("#morningKindSelect").value;
      const chosenCats = syncCategoryChips();
      const effectiveCats = chosenCats.length ? chosenCats : ["headlines"];
      if (rk === "production" && !confirm("Start a PRODUCTION run (omits TEST RUN watermark)?")) return;
      this.onAction("morning", { run_kind: rk, categories: effectiveCats });
    });
    this.bodyEl.querySelector("#btnDraftAi")?.addEventListener("click", () => {
      const autoApprove = Boolean(this.bodyEl.querySelector("#chkAutoApproveAi")?.checked);
      this.onAction("draft-ai", { auto_approve: autoApprove });
    });
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
        if (window.showOfficeToast) {
          window.showOfficeToast("Invalid JSON: " + err.message, "error");
        } else {
          alert("Invalid JSON: " + err.message);
        }
      }
    });
  }

  _renderOverviewTab(st) {
    const stages = this.state.manifest?.stages || {};
    const meta = st.stageKey ? stages[st.stageKey] : null;
    const arts = this.state.artifacts || {};
    const mediaSum = this.state.media_summary || null;

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
    const isDone = meta.status === "done";
    const isFailed = meta.status === "failed";
    const displayError = isDone ? "None" : (meta.error_message || "None");

    const STAGE_ORDER = ["scraper", "editor", "voice", "timeline", "media", "motion", "render", "qa"];
    const curIdx = STAGE_ORDER.indexOf(st.stageKey);
    const downstream = curIdx >= 0 ? STAGE_ORDER.slice(curIdx + 1) : [];

    let specificCardHtml = "";

    // 1. Scraper Station: Source Health Panel
    if (st.stageKey === "scraper") {
      const health = this.state.source_health || [];
      if (health.length) {
        specificCardHtml += `
          <div class="card">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
              <h4 style="margin:0;">Source Health (${health.length} Sources)</h4>
              <span class="muted" style="font-size:11px;">48h Staleness Gate</span>
            </div>
            <table class="source-table">
              <thead>
                <tr>
                  <th>Source</th>
                  <th>Status</th>
                  <th>Found</th>
                  <th>Kept</th>
                  <th>Dropped</th>
                  <th>Stale</th>
                  <th>Detail</th>
                </tr>
              </thead>
              <tbody>
                ${health.map((src) => {
                  const statusClass = `source-status-${(src.status || "ok").toLowerCase()}`;
                  return `
                    <tr>
                      <td><strong>${escapeHtml(src.name)}</strong></td>
                      <td><span class="source-status-badge ${statusClass}">${escapeHtml(src.status)}</span></td>
                      <td>${src.found ?? 0}</td>
                      <td style="color:#22c55e;font-weight:700">${src.kept ?? 0}</td>
                      <td>${src.dropped ?? 0}</td>
                      <td>${src.stale ?? 0}</td>
                      <td class="muted" style="max-width:140px;overflow:hidden;text-overflow:ellipsis;" title="${escapeHtml(src.detail || src.error || '')}">${escapeHtml(src.detail || src.error || '-')}</td>
                    </tr>
                  `;
                }).join("")}
              </tbody>
            </table>
          </div>
        `;
      }
    }

    // 2. Media Station: Media & Visual Picker Panel
    if (st.stageKey === "media") {
      specificCardHtml += `
        <div class="card">
          <h4>Media Candidates & Visual Selection</h4>
          <p class="muted" style="margin-top:0">
            ${mediaSum ? `Planned ${mediaSum.total_shots} shots (${mediaSum.auto_clips_count} auto-matched clips, ${mediaSum.auto_cards_count} headline cards)` : "No candidate plan generated yet."}
          </p>
          <div class="btn-row">
            ${arts.has_picker_html ? `<button class="btn btn-primary btn-sm" id="btnMediaPicker">🎨 Open Visual Picker</button>` : ""}
            <button class="btn btn-sm" id="btnMediaForce">↻ Re-scan Visuals (--force)</button>
          </div>
        </div>
      `;
    }

    // 3. QA Station: Inspection Report Panel & Redo Action
    if (st.stageKey === "qa") {
      const qaRep = this.state.qa_report;
      if (qaRep) {
        const checks = qaRep.checks || [];
        const warnings = qaRep.warnings || [];
        const redoStage = qaRep.redo_stage;
        const passed = qaRep.passed;

        specificCardHtml += `
          <div class="card">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
              <h4 style="margin:0;">QA Inspection: ${passed ? '<span style="color:#22c55e">PASSED ✓</span>' : '<span style="color:#ef4444">FAILED ✗</span>'}</h4>
              ${redoStage ? `<span class="badge badge-failed">REDO: ${escapeHtml(redoStage)}</span>` : ""}
            </div>
            ${
              redoStage
                ? `
              <div style="margin-bottom:12px;padding:8px 12px;background:rgba(239, 68, 68, 0.15);border:1px solid #ef4444;border-radius:6px;font-size:12px;">
                <strong>Remediation:</strong> QA identified root failure in <code>${escapeHtml(redoStage)}</code> stage.<br>
                <button class="btn btn-primary btn-sm" style="margin-top:8px" id="btnQaRedo">↺ Fix & Retry from ${escapeHtml(redoStage)}</button>
              </div>`
                : ""
            }
            <table class="source-table">
              <thead><tr><th>Check</th><th>Result</th><th>Detail</th></tr></thead>
              <tbody>
                ${checks.map((c) => `
                  <tr>
                    <td><strong>${escapeHtml(c.name)}</strong></td>
                    <td><span class="badge ${c.passed ? "badge-done" : "badge-failed"}">${c.passed ? "PASS" : "FAIL"}</span></td>
                    <td class="muted">${escapeHtml(c.detail || "")}</td>
                  </tr>
                `).join("")}
              </tbody>
            </table>
            ${
              warnings.length
                ? `
              <h4 style="margin:12px 0 6px;color:#f5a623;">Warnings & Freezes (${warnings.length})</h4>
              <ul style="padding-left:18px;margin:4px 0;">${warnings.map((w) => `<li style="font-size:11px;color:#fef3c7;">${escapeHtml(w)}</li>`).join("")}</ul>
            `
                : ""
            }
            <div class="btn-row" style="margin-top:12px">
              ${arts.has_storyboard ? `<button class="btn btn-sm" id="btnQaStoryboard">🖼 Storyboard</button>` : ""}
              ${arts.has_contact_sheet ? `<button class="btn btn-sm" id="btnQaContactSheet">📸 Contact Sheet</button>` : ""}
              ${arts.has_description_md ? `<button class="btn btn-sm" id="btnQaDescription">📋 Description</button>` : ""}
              ${arts.has_final_mp4 ? `<button class="btn btn-sm" id="btnQaVideo">🎥 Watch Video</button>` : ""}
            </div>
          </div>
        `;
      }
    }

    this.bodyEl.innerHTML = `
      ${specificCardHtml}
      <div class="card">
        <h4>Stage Status: ${escapeHtml(meta.status)}</h4>
        <div class="kv-grid">
          <span class="kv-label">Start Time</span><span>${escapeHtml(meta.start_time || "-")}</span>
          <span class="kv-label">End Time</span><span>${escapeHtml(meta.end_time || "-")}</span>
          <span class="kv-label">Retries</span><span>${meta.retry_count ?? 0} / ${meta.max_retries ?? 2}</span>
          <span class="kv-label">Error</span><span style="${isDone || !meta.error_message ? "color:#94a3b8" : "color:#fca5a5"}">${escapeHtml(displayError)}</span>
        </div>
        <div class="btn-row" style="margin-top:12px">
          ${isFailed ? `<button class="btn btn-primary btn-sm" id="btnRetryStage">↺ Retry ${escapeHtml(st.stageKey)}</button>` : ""}
          <button class="btn btn-sm" id="btnResetStage">↺ Reset Stage only</button>
          ${downstream.length ? `<button class="btn btn-danger btn-sm" id="btnResetDownstream">⏭ Reset from ${escapeHtml(st.stageKey)} (Downstream)</button>` : ""}
        </div>
      </div>
      <div class="card">
        <h4>Metrics</h4>
        <div class="log-box" style="max-height:220px">${escapeHtml(metrics)}</div>
      </div>
    `;

    this.bodyEl.querySelector("#btnResetStage")?.addEventListener("click", () => {
      if (confirm(`Reset stage '${st.stageKey}' back to pending?`)) {
        this.onAction("reset-stage", { stage: st.stageKey, reset_downstream: false });
      }
    });

    this.bodyEl.querySelector("#btnResetDownstream")?.addEventListener("click", () => {
      const msg = `Resetting from stage '${st.stageKey}' will also reset downstream stages:\n  ${downstream.join(", ")}\n\nAre you sure you want to proceed?`;
      if (confirm(msg)) {
        this.onAction("reset-stage", { stage: st.stageKey, reset_downstream: true });
      }
    });

    this.bodyEl.querySelector("#btnRetryStage")?.addEventListener("click", () => {
      this.onAction("reset-stage", { stage: st.stageKey, reset_downstream: false });
    });

    // Wire QA Panel buttons
    this.bodyEl.querySelector("#btnQaRedo")?.addEventListener("click", () => {
      const qaRep = this.state.qa_report;
      const redoStage = qaRep?.redo_stage || "render";
      const redoIdx = STAGE_ORDER.indexOf(redoStage);
      const affected = redoIdx >= 0 ? STAGE_ORDER.slice(redoIdx + 1) : [];
      const msg = `Fix & retry from '${redoStage}'?\nDownstream stages that will be reset: ${affected.join(", ") || "none"}.\n\nProceed?`;
      if (confirm(msg)) {
        this.onAction("reset-stage", { stage: redoStage, reset_downstream: true });
      }
    });
    this.bodyEl.querySelector("#btnQaStoryboard")?.addEventListener("click", () => this._openIframeModal("qa/storyboard.html", `QA Storyboard (${this.state.date})`));
    this.bodyEl.querySelector("#btnQaContactSheet")?.addEventListener("click", () => this._openImageModal("qa/contact_sheet.jpg", `QA Contact Sheet (${this.state.date})`));
    this.bodyEl.querySelector("#btnQaDescription")?.addEventListener("click", () => {
      this.openModal(`YouTube Description (${this.state.date})`, `<div class="preview-md">${escapeHtml(this.state.description_md)}</div>`);
    });
    this.bodyEl.querySelector("#btnQaVideo")?.addEventListener("click", () => this._openVideoModal("render/final.mp4", `Final Broadcast (${this.state.date})`));

    // Wire Media Panel buttons
    this.bodyEl.querySelector("#btnMediaPicker")?.addEventListener("click", () => this._openIframeModal("media/review/picker.html", `Media Shot Picker (${this.state.date})`));
    this.bodyEl.querySelector("#btnMediaForce")?.addEventListener("click", () => {
      if (confirm("Re-scan online visual candidates for all shots?")) {
        this.onAction("prepare-media", { force: true });
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

  _formatLogLine(line) {
    const esc = escapeHtml(line);
    if (/\[ERROR\]|FAILED|Exception|Traceback/i.test(line)) {
      return `<div class="log-line log-err">${esc}</div>`;
    }
    if (/\[WARN\]|WARNING/i.test(line)) {
      return `<div class="log-line log-warn">${esc}</div>`;
    }
    if (/\[OK\]|SUCCESS|done|Completed/i.test(line)) {
      return `<div class="log-line log-ok">${esc}</div>`;
    }
    if (/\[INFO\]/i.test(line)) {
      return `<div class="log-line log-info">${esc}</div>`;
    }
    return `<div class="log-line">${esc}</div>`;
  }

  _renderLogsTab(st) {
    const key = st.stageKey ? st.stageKey.toLowerCase() : "";
    let lines = key
      ? this.logs.filter((l) => l.toLowerCase().includes(key) || l.toLowerCase().includes("orchestrator"))
      : this.logs;

    if (this.logSearchTerm) {
      const q = this.logSearchTerm.toLowerCase();
      lines = lines.filter((l) => l.toLowerCase().includes(q));
    }

    const visibleLines = lines.slice(-250);
    const formattedHtml = visibleLines.length
      ? visibleLines.map((l) => this._formatLogLine(l)).join("")
      : `<div class="muted" style="padding:10px;">No matching log lines.</div>`;

    const existingBox = this.bodyEl.querySelector("#activeLogBox") || this.bodyEl.querySelector(".log-box");
    let wasAtBottom = true;
    let prevScrollTop = 0;
    if (existingBox) {
      wasAtBottom = existingBox.scrollHeight - existingBox.scrollTop <= existingBox.clientHeight + 40;
      prevScrollTop = existingBox.scrollTop;
    }

    this.bodyEl.innerHTML = `
      <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; flex-wrap:wrap; gap:6px;">
          <h4 style="margin:0; font-size:13px;">Live Run Log (${escapeHtml(this.state.date)})</h4>
          <div style="display:flex; gap:6px; align-items:center;">
            <input type="text" id="logSearchInput" placeholder="Filter..." value="${escapeHtml(this.logSearchTerm)}" style="width:110px; padding:3px 6px; font-size:11px; background:#101e38; border:1px solid #25436e; color:#fff; border-radius:4px;" />
            <button class="btn btn-sm btn-ghost" id="btnToggleLogPause" title="${this.isLogPaused ? "Resume auto-scroll" : "Pause auto-scroll"}">${this.isLogPaused ? "▶ Follow" : "⏸ Pause"}</button>
            <button class="btn btn-sm btn-ghost" id="btnCopyLogs" title="Copy visible logs to clipboard">📋 Copy</button>
            <button class="btn btn-sm btn-ghost" id="btnDownloadLogs" title="Download run.log">⬇</button>
          </div>
        </div>
        <div class="log-box" id="activeLogBox" style="font-family:monospace; font-size:11px; max-height:480px; overflow-y:auto; line-height:1.45;">${formattedHtml}</div>
      </div>
    `;

    const newBox = this.bodyEl.querySelector("#activeLogBox");
    if (newBox) {
      if (!this.isLogPaused && wasAtBottom) {
        newBox.scrollTop = newBox.scrollHeight;
      } else {
        newBox.scrollTop = prevScrollTop;
      }
    }

    const searchInput = this.bodyEl.querySelector("#logSearchInput");
    searchInput?.addEventListener("input", (e) => {
      this.logSearchTerm = e.target.value;
      this._renderLogsTab(st);
      const updatedInput = this.bodyEl.querySelector("#logSearchInput");
      if (updatedInput) {
        updatedInput.focus();
        updatedInput.setSelectionRange(updatedInput.value.length, updatedInput.value.length);
      }
    });

    this.bodyEl.querySelector("#btnToggleLogPause")?.addEventListener("click", () => {
      this.isLogPaused = !this.isLogPaused;
      this._renderLogsTab(st);
    });

    this.bodyEl.querySelector("#btnCopyLogs")?.addEventListener("click", () => {
      const rawText = lines.join("\n");
      navigator.clipboard.writeText(rawText).then(() => {
        if (window.showOfficeToast) {
          window.showOfficeToast("Copied logs to clipboard!", "done", 3000);
        }
      });
    });

    this.bodyEl.querySelector("#btnDownloadLogs")?.addEventListener("click", () => {
      const rawText = this.logs.join("\n");
      const blob = new Blob([rawText], { type: "text/plain" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `run-${this.state.date}.log`;
      a.click();
    });
  }

  _getTokenQuery() {
    const token = new URLSearchParams(window.location.search).get("token") || sessionStorage.getItem("office_token");
    return token ? `&token=${encodeURIComponent(token)}` : "";
  }

  async _previewRunFile(relPath, title) {
    const url = `/api/runs/${encodeURIComponent(this.state.date)}/file?path=${encodeURIComponent(relPath)}${this._getTokenQuery()}`;
    const token = new URLSearchParams(window.location.search).get("token") || sessionStorage.getItem("office_token");
    const headers = token ? { "X-Office-Token": token } : {};
    const res = await fetch(url, { headers });
    if (!res.ok) {
      if (window.showOfficeToast) {
        window.showOfficeToast(`File not available yet: ${relPath}`, "info");
      } else {
        alert(`File not available yet: ${relPath}`);
      }
      return;
    }
    const text = await res.text();
    this.openModal(title, `<div class="preview-md" style="max-height:70vh">${escapeHtml(text)}</div>`);
  }

  _openVideoModal(relPath, title) {
    const url = `/api/runs/${encodeURIComponent(this.state.date)}/file?path=${encodeURIComponent(relPath)}${this._getTokenQuery()}`;
    this.openModal(title, `<video controls autoplay src="${url}"></video>`);
  }

  _openAudioModal(relPath, title) {
    const url = `/api/runs/${encodeURIComponent(this.state.date)}/file?path=${encodeURIComponent(relPath)}${this._getTokenQuery()}`;
    this.openModal(title, `<audio controls autoplay style="width:100%" src="${url}"></audio>`);
  }

  _openImageModal(relPath, title) {
    const url = `/api/runs/${encodeURIComponent(this.state.date)}/file?path=${encodeURIComponent(relPath)}${this._getTokenQuery()}`;
    this.openModal(title, `<img src="${url}" style="max-width:100%;border-radius:8px" />`);
  }

  _openIframeModal(relPath, title) {
    const url = `/api/runs/${encodeURIComponent(this.state.date)}/file?path=${encodeURIComponent(relPath)}${this._getTokenQuery()}`;
    this.openModal(title, `<iframe src="${url}" sandbox="allow-scripts allow-same-origin allow-forms allow-popups"></iframe>`);
  }
}
