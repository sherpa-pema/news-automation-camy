/**
 * 60fps HTML5 2D Canvas renderer for Nepal News Bot HQ.
 */
import {
  STATION_BY_ID,
  STATIONS,
  TRACK_EDGES,
  VIRTUAL_HEIGHT,
  VIRTUAL_WIDTH,
} from "./office_map.js";

const STATUS_COLORS = {
  pending: { bg: "#10223d", border: "#25436e", badge: "#64748b", text: "IDLE" },
  running: { bg: "#132c4e", border: "#38bdf8", badge: "#38bdf8", text: "WORKING" },
  prepared: { bg: "#2b2412", border: "#f5a623", badge: "#f5a623", text: "NEEDS WRITER" },
  needs_revision: { bg: "#311a18", border: "#f97316", badge: "#f97316", text: "FIX DRAFT" },
  awaiting_approval: { bg: "#33240e", border: "#f5a623", badge: "#f5a623", text: "SIGN OFF" },
  done: { bg: "#0f2c27", border: "#22c55e", badge: "#22c55e", text: "DONE" },
  failed: { bg: "#34141b", border: "#ef4444", badge: "#ef4444", text: "FAILED" },
};

export class OfficeCanvasEngine {
  constructor(canvas, { onSelectStation, onHoverInfo }) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.onSelectStation = onSelectStation;
    this.onHoverInfo = onHoverInfo;

    this.state = null;
    this.selectedStationId = null;
    this.hoveredStationId = null;
    this.hoveredPacket = null;

    // Visual overrides used by Replay Mode
    this.overrideStatuses = null;
    this.bubbles = new Map(); // stationId -> { text, expiresAt, color }
    this.packets = []; // { from, to, label, artifact, progress, speed, color, sizeBytes }

    this.scale = 1;
    this.offsetX = 0;
    this.offsetY = 0;
    this.mouseX = -999;
    this.mouseY = -999;

    this._bindEvents();
    this._loop = this._loop.bind(this);
    this._lastTs = performance.now();
    requestAnimationFrame(this._loop);
  }

  setState(state) {
    this.state = state;
  }

  setOverrideStatuses(mapOrNull) {
    this.overrideStatuses = mapOrNull;
  }

  setSelectedStation(id) {
    this.selectedStationId = id;
  }

  spawnPacket({ from, to, label, artifact, color = "#f5a623", sizeBytes = null, durationMs = 1500 }) {
    if (!STATION_BY_ID[from] || !STATION_BY_ID[to]) return;
    this.packets.push({
      from,
      to,
      label: label || artifact || "data",
      artifact: artifact || "",
      color,
      sizeBytes,
      progress: 0,
      speed: 1000 / Math.max(500, durationMs),
      x: STATION_BY_ID[from].x,
      y: STATION_BY_ID[from].y,
    });
  }

  showBubble(stationId, text, durationMs = 4500, color = "#38bdf8") {
    if (!text) return;
    this.bubbles.set(stationId, {
      text: String(text).slice(0, 48),
      expiresAt: performance.now() + durationMs,
      color,
    });
  }

  clearEffects() {
    this.packets = [];
    this.bubbles.clear();
  }

  _bindEvents() {
    const toVirtual = (clientX, clientY) => {
      const rect = this.canvas.getBoundingClientRect();
      const vx = (clientX - rect.left - this.offsetX) / this.scale;
      const vy = (clientY - rect.top - this.offsetY) / this.scale;
      return { vx, vy, clientX, clientY };
    };

    this.canvas.addEventListener("mousemove", (e) => {
      const { vx, vy, clientX, clientY } = toVirtual(e.clientX, e.clientY);
      this.mouseX = vx;
      this.mouseY = vy;
      this._updateHover(vx, vy, clientX, clientY);
    });

    this.canvas.addEventListener("mouseleave", () => {
      this.hoveredStationId = null;
      this.hoveredPacket = null;
      this.onHoverInfo?.(null);
    });

    this.canvas.addEventListener("click", (e) => {
      const { vx, vy } = toVirtual(e.clientX, e.clientY);
      const hit = this._hitStation(vx, vy);
      if (hit) {
        this.selectedStationId = hit.id;
        this.onSelectStation?.(hit.id);
      }
    });
  }

  _hitStation(vx, vy) {
    for (const st of STATIONS) {
      if (
        vx >= st.x - st.w / 2 &&
        vx <= st.x + st.w / 2 &&
        vy >= st.y - st.h / 2 &&
        vy <= st.y + st.h / 2
      ) {
        return st;
      }
    }
    return null;
  }

  _updateHover(vx, vy, clientX, clientY) {
    // Check moving packets first
    for (const p of this.packets) {
      const dx = vx - p.x;
      const dy = vy - p.y;
      if (dx * dx + dy * dy <= 24 * 24) {
        this.hoveredPacket = p;
        const sizeTxt = p.sizeBytes ? ` (${(p.sizeBytes / 1024).toFixed(1)} KB)` : "";
        this.onHoverInfo?.({
          x: clientX,
          y: clientY,
          title: `Packet: ${p.label}${sizeTxt}`,
          subtitle: `${STATION_BY_ID[p.from]?.title} → ${STATION_BY_ID[p.to]?.title}`,
        });
        return;
      }
    }
    this.hoveredPacket = null;

    const hit = this._hitStation(vx, vy);
    this.hoveredStationId = hit ? hit.id : null;
    if (hit) {
      const stStatus = this._stationStatus(hit);
      this.onHoverInfo?.({
        x: clientX,
        y: clientY,
        title: `${hit.title} (${hit.agentName})`,
        subtitle: `${hit.role} • ${stStatus.toUpperCase()} — Click to inspect`,
      });
    } else {
      this.onHoverInfo?.(null);
    }
  }

  _stationStatus(station) {
    const stages = this.state?.manifest?.stages || {};
    if (this.overrideStatuses) {
      if (station.stageKey) return this.overrideStatuses[station.stageKey] || "pending";
      if (station.id === "boss") {
        const ed = this.overrideStatuses.editor || "pending";
        if (ed === "awaiting_approval" || ed === "prepared" || ed === "needs_revision") return ed;
        return ed === "done" ? "done" : "pending";
      }
      if (station.id === "publish") {
        return this.overrideStatuses.qa === "done" ? "done" : "pending";
      }
      if (station.id === "orchestrator") {
        const anyRunning = Object.values(this.overrideStatuses).some((s) => s === "running");
        return anyRunning ? "running" : Object.values(this.overrideStatuses).every((s) => s === "done") ? "done" : "pending";
      }
      return "pending";
    }

    if (station.stageKey) {
      return stages[station.stageKey]?.status || "pending";
    }
    if (station.id === "boss") {
      const ed = stages.editor?.status || "pending";
      if (ed === "awaiting_approval" || ed === "prepared" || ed === "needs_revision") return ed;
      return ed === "done" ? "done" : "pending";
    }
    if (station.id === "publish") {
      const qa = stages.qa?.status || "pending";
      return qa === "done" && this.state?.artifacts?.has_final_mp4 ? "done" : "pending";
    }
    if (station.id === "orchestrator") {
      if (this.state?.active_process || this.state?.active_job?.status === "running") return "running";
      const top = this.state?.manifest?.status || "pending";
      if (top === "done") return "done";
      if (top === "failed") return "failed";
      if (top === "paused") return "awaiting_approval";
      return "pending";
    }
    return "pending";
  }

  _resizeIfNeeded() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.parentElement.getBoundingClientRect();
    const w = Math.max(320, Math.floor(rect.width));
    const h = Math.max(240, Math.floor(rect.height));
    if (this.canvas.width !== Math.floor(w * dpr) || this.canvas.height !== Math.floor(h * dpr)) {
      this.canvas.width = Math.floor(w * dpr);
      this.canvas.height = Math.floor(h * dpr);
    }
    this.scale = Math.min(w / VIRTUAL_WIDTH, h / VIRTUAL_HEIGHT);
    this.offsetX = (w - VIRTUAL_WIDTH * this.scale) / 2;
    this.offsetY = (h - VIRTUAL_HEIGHT * this.scale) / 2;
    return { dpr, w, h };
  }

  _loop(now) {
    const dt = Math.min(0.1, (now - this._lastTs) / 1000);
    this._lastTs = now;

    // Advance packet sprites
    const remaining = [];
    for (const p of this.packets) {
      p.progress += p.speed * dt;
      if (p.progress < 1) {
        const a = STATION_BY_ID[p.from];
        const b = STATION_BY_ID[p.to];
        const t = p.progress;
        // Gentle arc
        const mx = (a.x + b.x) / 2;
        const my = (a.y + b.y) / 2 - 22;
        p.x = (1 - t) * (1 - t) * a.x + 2 * (1 - t) * t * mx + t * t * b.x;
        p.y = (1 - t) * (1 - t) * a.y + 2 * (1 - t) * t * my + t * t * b.y;
        remaining.push(p);
      }
    }
    this.packets = remaining;

    this._render(now);
    requestAnimationFrame(this._loop);
  }

  _render(now) {
    const { dpr, w, h } = this._resizeIfNeeded();
    const ctx = this.ctx;
    ctx.save();
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    ctx.translate(this.offsetX, this.offsetY);
    ctx.scale(this.scale, this.scale);

    this._drawFloorAndDecor(ctx, now);
    this._drawTracks(ctx, now);

    for (const st of STATIONS) {
      this._drawStation(ctx, st, now);
    }

    this._drawPackets(ctx, now);
    this._drawBubbles(ctx, now);

    ctx.restore();
  }

  _drawFloorAndDecor(ctx, now) {
    // Office floor perimeter
    ctx.fillStyle = "#09172c";
    this._roundRect(ctx, 22, 22, VIRTUAL_WIDTH - 44, VIRTUAL_HEIGHT - 44, 18, true, false);
    ctx.strokeStyle = "#1c365c";
    ctx.lineWidth = 2;
    this._roundRect(ctx, 22, 22, VIRTUAL_WIDTH - 44, VIRTUAL_HEIGHT - 44, 18, false, true);

    // Subtle tile grid
    ctx.strokeStyle = "rgba(31, 59, 100, 0.24)";
    ctx.lineWidth = 1;
    for (let x = 60; x < VIRTUAL_WIDTH - 40; x += 48) {
      ctx.beginPath();
      ctx.moveTo(x, 30);
      ctx.lineTo(x, VIRTUAL_HEIGHT - 30);
      ctx.stroke();
    }
    for (let y = 60; y < VIRTUAL_HEIGHT - 40; y += 48) {
      ctx.beginPath();
      ctx.moveTo(30, y);
      ctx.lineTo(VIRTUAL_WIDTH - 30, y);
      ctx.stroke();
    }

    // Center command rug
    ctx.fillStyle = "rgba(245, 166, 35, 0.05)";
    ctx.beginPath();
    ctx.ellipse(635, 385, 180, 105, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = "rgba(245, 166, 35, 0.18)";
    ctx.setLineDash([6, 6]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Studio Header Sign at top-center
    ctx.fillStyle = "#0e2240";
    this._roundRect(ctx, 470, 32, 340, 32, 8, true, false);
    ctx.strokeStyle = "#26487a";
    this._roundRect(ctx, 470, 32, 340, 32, 8, false, true);
    ctx.fillStyle = "#f5a623";
    ctx.font = "bold 12px Segoe UI, sans-serif";
    ctx.textAlign = "center";
    const timeStr = new Date().toLocaleTimeString("en-US", {
      timeZone: "Asia/Kathmandu",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
    ctx.fillText(`NEPAL NEWS BOT HQ  •  KTM ${timeStr}`, 640, 52);

    // Decorative office plants & coffee bar
    const plants = [
      [48, 54],
      [1232, 54],
      [48, 706],
      [1232, 706],
      [920, 385],
    ];
    for (const [px, py] of plants) {
      ctx.fillStyle = "#164e3b";
      ctx.beginPath();
      ctx.arc(px, py, 12, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#22c55e";
      ctx.beginPath();
      ctx.arc(px - 3, py - 3, 7, 0, Math.PI * 2);
      ctx.fill();
    }

    // Coffee machine near Copy Desk
    ctx.fillStyle = "#1b3358";
    this._roundRect(ctx, 322, 128, 30, 36, 5, true, false);
    ctx.fillStyle = "#f59e0b";
    ctx.fillRect(331, 134, 12, 8);
  }

  _drawTracks(ctx, now) {
    // Orchestrator dispatch spokes (subtle)
    const orch = STATION_BY_ID.orchestrator;
    ctx.save();
    ctx.strokeStyle = "rgba(245, 166, 35, 0.14)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 6]);
    for (const st of STATIONS) {
      if (st.id === "orchestrator" || st.id === "publish") continue;
      ctx.beginPath();
      ctx.moveTo(orch.x, orch.y);
      ctx.lineTo(st.x, st.y);
      ctx.stroke();
    }
    ctx.restore();

    // Primary inter-agent data tracks
    for (const [fromId, toId] of TRACK_EDGES) {
      const a = STATION_BY_ID[fromId];
      const b = STATION_BY_ID[toId];
      if (!a || !b) continue;
      const aStatus = this._stationStatus(a);
      const active = aStatus === "done" || aStatus === "running";
      ctx.save();
      ctx.strokeStyle = active ? "rgba(56, 189, 248, 0.42)" : "rgba(100, 116, 139, 0.25)";
      ctx.lineWidth = active ? 3 : 2;
      if (fromId === "qa" && toId === "orchestrator") {
        ctx.strokeStyle = "rgba(239, 68, 68, 0.25)";
        ctx.setLineDash([5, 5]);
      }
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();

      // Direction chevron halfway
      const mx = (a.x + b.x) / 2;
      const my = (a.y + b.y) / 2;
      const ang = Math.atan2(b.y - a.y, b.x - a.x);
      ctx.translate(mx, my);
      ctx.rotate(ang);
      ctx.fillStyle = active ? "#38bdf8" : "#475569";
      ctx.beginPath();
      ctx.moveTo(6, 0);
      ctx.lineTo(-4, -4);
      ctx.lineTo(-4, 4);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }
  }

  _drawStation(ctx, st, now) {
    const status = this._stationStatus(st);
    const palette = STATUS_COLORS[status] || STATUS_COLORS.pending;
    const isSelected = this.selectedStationId === st.id;
    const isHovered = this.hoveredStationId === st.id;

    const left = st.x - st.w / 2;
    const top = st.y - st.h / 2;

    ctx.save();

    // Pulse glow when running or awaiting boss action
    if (status === "running" || status === "awaiting_approval" || status === "prepared" || status === "needs_revision") {
      const pulse = 0.4 + 0.35 * Math.sin(now / 260);
      ctx.shadowColor = palette.border;
      ctx.shadowBlur = 18 * pulse;
    }

    // Room / desk card
    ctx.fillStyle = palette.bg;
    this._roundRect(ctx, left, top, st.w, st.h, 12, true, false);
    ctx.shadowBlur = 0;

    ctx.strokeStyle = isSelected ? "#ffffff" : isHovered ? "#f5a623" : palette.border;
    ctx.lineWidth = isSelected ? 3 : isHovered ? 2.5 : 1.8;
    this._roundRect(ctx, left, top, st.w, st.h, 12, false, true);

    // Desk surface
    ctx.fillStyle = "#162d50";
    this._roundRect(ctx, left + 14, top + 36, st.w - 28, 38, 6, true, false);

    // Station-specific visual prop
    this._drawProp(ctx, st, status, now, left, top);

    // Animated Agent Character
    this._drawAgentAvatar(ctx, st, status, now, st.x - 42, top + 56);

    // Header Title
    ctx.fillStyle = "#eef4ff";
    ctx.font = "bold 13px Segoe UI, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(st.title, left + 12, top + 21);

    // Status Pill
    const pillW = Math.min(92, ctx.measureText(palette.text).width + 14);
    ctx.fillStyle = palette.badge;
    this._roundRect(ctx, left + st.w - pillW - 10, top + 8, pillW, 18, 9, true, false);
    ctx.fillStyle = "#06101f";
    ctx.font = "bold 9px Segoe UI, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(palette.text, left + st.w - pillW / 2 - 10, top + 20);

    // Footer role / live metric
    ctx.fillStyle = "#9fb3d1";
    ctx.font = "11px Segoe UI, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(this._stationSubtext(st, status), left + 12, top + st.h - 14);

    ctx.restore();
  }

  _stationSubtext(st, status) {
    const stages = this.state?.manifest?.stages || {};
    const meta = st.stageKey ? stages[st.stageKey] : null;
    const m = meta?.metrics || {};
    if (st.id === "scraper" && m.total_articles) return `${m.total_articles} articles fetched`;
    if (st.id === "voice" && m.total_duration_sec) return `${m.total_duration_sec}s • ${m.voice || "NP_lemi"}`;
    if (st.id === "timeline" && m.total_words) return `${m.total_words} words • ${m.caption_count || 0} caps`;
    if (st.id === "media" && m.total_shots) return `${m.auto_clips_count || 0} clips, ${m.auto_cards_count || 0} cards`;
    if (st.id === "render" && m.encode_speed) return `FFmpeg ${m.encode_speed}x • ${m.total_scenes || 7} scenes`;
    if (st.id === "qa" && m.checks_count) return m.passed ? `PASS ${m.checks_count}/${m.checks_count}` : `QA FAILED`;
    if (st.id === "boss" && (status === "awaiting_approval" || status === "prepared")) return "Click here for Boss action!";
    return st.role.slice(0, 30);
  }

  _drawProp(ctx, st, status, now, left, top) {
    const rx = left + st.w - 76;
    const ry = top + 41;

    if (st.propType === "monitors") {
      // 4 RSS feed screens
      const colors = status === "done" || status === "running" ? ["#22c55e", "#22c55e", "#22c55e", "#22c55e"] : ["#475569", "#475569", "#475569", "#475569"];
      for (let i = 0; i < 4; i++) {
        ctx.fillStyle = "#081426";
        ctx.fillRect(rx + (i % 2) * 28, ry + Math.floor(i / 2) * 14, 24, 11);
        ctx.fillStyle = colors[i];
        ctx.fillRect(rx + (i % 2) * 28 + 2, ry + Math.floor(i / 2) * 14 + 2, 4, 4);
      }
    } else if (st.propType === "booth") {
      // Animated VU waveform bars
      for (let i = 0; i < 6; i++) {
        const bh = status === "running" ? 6 + Math.abs(Math.sin(now / 110 + i)) * 18 : status === "done" ? 12 : 4;
        ctx.fillStyle = status === "running" ? "#ec4899" : "#64748b";
        ctx.fillRect(rx + i * 8, ry + 26 - bh, 5, bh);
      }
    } else if (st.propType === "server_rack") {
      // Spinning cooling fans
      for (let i = 0; i < 2; i++) {
        const fx = rx + 14 + i * 26;
        const fy = ry + 14;
        ctx.strokeStyle = status === "running" ? "#eab308" : "#475569";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(fx, fy, 10, 0, Math.PI * 2);
        ctx.stroke();
        const ang = status === "running" ? (now / 90) * (i ? -1 : 1) : 0;
        ctx.beginPath();
        ctx.moveTo(fx + Math.cos(ang) * 8, fy + Math.sin(ang) * 8);
        ctx.lineTo(fx - Math.cos(ang) * 8, fy - Math.sin(ang) * 8);
        ctx.stroke();
      }
    } else if (st.propType === "boss_desk") {
      // Gold inbox tray
      const alert = status === "awaiting_approval" || status === "prepared" || status === "needs_revision";
      ctx.fillStyle = alert ? "#f5a623" : "#334155";
      this._roundRect(ctx, rx, ry + 4, 52, 22, 4, true, false);
      ctx.fillStyle = "#06101f";
      ctx.font = "bold 9px Segoe UI, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(alert ? "INBOX!" : "CLEAR", rx + 26, ry + 18);
    } else if (st.propType === "command_pod") {
      // Spinning radar ring
      const cx = rx + 26;
      const cy = ry + 14;
      ctx.strokeStyle = status === "running" ? "#f5a623" : "#475569";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(cx, cy, 12, 0, Math.PI * 2);
      ctx.stroke();
      if (status === "running") {
        const a = now / 300;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + Math.cos(a) * 12, cy + Math.sin(a) * 12);
        ctx.stroke();
      }
    } else if (st.propType === "gallery") {
      // Photo frames / gallery cards for Archive Studio
      const alert = status === "awaiting_approval";
      ctx.fillStyle = "#081426";
      this._roundRect(ctx, rx, ry + 2, 54, 26, 3, true, false);
      ctx.fillStyle = alert ? "#22c55e" : status === "running" ? "#38bdf8" : "#334155";
      ctx.fillRect(rx + 4, ry + 6, 20, 16);
      ctx.fillStyle = alert ? "#38bdf8" : "#475569";
      ctx.fillRect(rx + 28, ry + 6, 22, 16);
      if (alert) {
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 8px Segoe UI, sans-serif";
        ctx.textAlign = "center";
        ctx.fillText("PICK!", rx + 27, ry + 17);
      }
    } else {
      // Generic monitor
      ctx.fillStyle = "#081426";
      this._roundRect(ctx, rx + 4, ry + 2, 46, 24, 3, true, false);
      ctx.fillStyle = status === "done" ? "#22c55e" : status === "running" ? "#38bdf8" : "#334155";
      ctx.fillRect(rx + 8, ry + 6, 38, 4);
      ctx.fillRect(rx + 8, ry + 13, 26, 3);
    }
  }

  _drawAgentAvatar(ctx, st, status, now, ax, ay) {
    let bob = 0;
    if (status === "running") {
      bob = Math.sin(now / 110 + st.x) * 3;
    } else if (status === "awaiting_approval" || status === "prepared") {
      bob = Math.abs(Math.sin(now / 220)) * -4;
    } else {
      bob = Math.sin(now / 700 + st.y) * 1;
    }

    ctx.save();
    ctx.translate(ax, ay + bob);

    // Shoulders / body
    ctx.fillStyle = st.avatarColor;
    ctx.beginPath();
    ctx.ellipse(0, 8, 13, 8, 0, 0, Math.PI * 2);
    ctx.fill();

    // Head
    ctx.fillStyle = "#fde68a";
    ctx.beginPath();
    ctx.arc(0, -4, 9, 0, Math.PI * 2);
    ctx.fill();

    // Eyes
    ctx.fillStyle = "#081222";
    ctx.fillRect(-4, -5, 2, 2);
    ctx.fillRect(2, -5, 2, 2);

    // Status emblem above head
    if (status === "done") {
      ctx.fillStyle = "#22c55e";
      ctx.beginPath();
      ctx.arc(11, -11, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#041b0c";
      ctx.font = "bold 9px Segoe UI";
      ctx.textAlign = "center";
      ctx.fillText("✓", 11, -8);
    } else if (status === "failed") {
      ctx.fillStyle = "#ef4444";
      ctx.beginPath();
      ctx.arc(11, -11, 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#fff";
      ctx.font = "bold 9px Segoe UI";
      ctx.textAlign = "center";
      ctx.fillText("!", 11, -8);
    }
    ctx.restore();
  }

  _drawPackets(ctx, now) {
    for (const p of this.packets) {
      ctx.save();
      ctx.translate(p.x, p.y);
      ctx.shadowColor = p.color;
      ctx.shadowBlur = 14;

      // Packet envelope
      ctx.fillStyle = p.color;
      this._roundRect(ctx, -12, -9, 24, 18, 4, true, false);
      ctx.shadowBlur = 0;

      ctx.strokeStyle = "#06101f";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(-11, -7);
      ctx.lineTo(0, 2);
      ctx.lineTo(11, -7);
      ctx.stroke();

      // Label tag
      ctx.font = "bold 10px Segoe UI, sans-serif";
      const tw = ctx.measureText(p.label).width + 10;
      ctx.fillStyle = "rgba(6, 16, 31, 0.9)";
      this._roundRect(ctx, -tw / 2, -28, tw, 16, 4, true, false);
      ctx.fillStyle = "#eef4ff";
      ctx.textAlign = "center";
      ctx.fillText(p.label, 0, -16);
      ctx.restore();
    }
  }

  _drawBubbles(ctx, now) {
    for (const [stationId, b] of this.bubbles.entries()) {
      if (now > b.expiresAt) {
        this.bubbles.delete(stationId);
        continue;
      }
      const st = STATION_BY_ID[stationId];
      if (!st) continue;
      ctx.save();
      ctx.font = "600 11px Segoe UI, sans-serif";
      const bw = Math.min(220, ctx.measureText(b.text).width + 16);
      const bx = st.x - bw / 2;
      const by = st.y - st.h / 2 - 30;
      ctx.fillStyle = "rgba(11, 25, 46, 0.96)";
      this._roundRect(ctx, bx, by, bw, 22, 6, true, false);
      ctx.strokeStyle = b.color;
      ctx.lineWidth = 1.2;
      this._roundRect(ctx, bx, by, bw, 22, 6, false, true);
      ctx.fillStyle = "#eef4ff";
      ctx.textAlign = "center";
      ctx.fillText(b.text, st.x, by + 15);
      ctx.restore();
    }
  }

  _roundRect(ctx, x, y, w, h, r, fill, stroke) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
    if (fill) ctx.fill();
    if (stroke) ctx.stroke();
  }
}
