/**
 * WebSocket stream manager + Historical Replay Engine for Nepal News Bot HQ.
 */

export class OfficeWsClient {
  constructor({ onSnapshot, onOfficeEvent, onLogLines, onConnectionChange }) {
    this.onSnapshot = onSnapshot;
    this.onOfficeEvent = onOfficeEvent;
    this.onLogLines = onLogLines;
    this.onConnectionChange = onConnectionChange;
    this.ws = null;
    this.currentDate = null;
    this.reconnectTimer = null;
    this.closedIntentionally = false;
  }

  connect(runDate) {
    this.disconnect();
    this.currentDate = runDate;
    this.closedIntentionally = false;

    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${proto}//${location.host}/ws/runs/${encodeURIComponent(runDate)}`;
    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      this.onConnectionChange?.(true);
    };

    this.ws.onmessage = (evt) => {
      try {
        const msg = JSON.parse(evt.data);
        if (msg.type === "state_snapshot") this.onSnapshot?.(msg.data);
        else if (msg.type === "office_event") this.onOfficeEvent?.(msg.data);
        else if (msg.type === "log_lines") this.onLogLines?.(msg.data);
      } catch (err) {
        console.error("WS parse error", err);
      }
    };

    this.ws.onclose = () => {
      this.onConnectionChange?.(false);
      if (!this.closedIntentionally && this.currentDate === runDate) {
        this.reconnectTimer = setTimeout(() => this.connect(runDate), 2000);
      }
    };

    this.ws.onerror = () => {
      this.ws?.close();
    };
  }

  disconnect() {
    this.closedIntentionally = true;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }
}

export class ReplayController {
  constructor(engine, { onTickLabel, onDone }) {
    this.engine = engine;
    this.onTickLabel = onTickLabel;
    this.onDone = onDone;
    this.timers = [];
    this.active = false;
  }

  start(events, speed = 3) {
    this.stop();
    if (!events || events.length === 0) {
      this.onDone?.();
      return;
    }

    this.active = true;
    const statuses = {
      scraper: "pending",
      editor: "pending",
      voice: "pending",
      timeline: "pending",
      media: "pending",
      motion: "pending",
      render: "pending",
      qa: "pending",
    };
    this.engine.clearEffects();
    this.engine.setOverrideStatuses({ ...statuses });

    const stepMs = Math.max(250, Math.round(1400 / Math.max(1, speed)));

    events.forEach((ev, idx) => {
      const t = setTimeout(() => {
        if (!this.active) return;
        this.onTickLabel?.(`Replaying (${idx + 1}/${events.length}): ${ev.message || ev.label || ev.stage || ev.type}`);

        if (ev.type === "stage_status" && ev.stage) {
          statuses[ev.stage] = ev.status || "running";
          this.engine.setOverrideStatuses({ ...statuses });
          if (ev.message) {
            this.engine.showBubble(ev.stage, ev.message, stepMs * 1.8);
          }
        } else if (ev.type === "packet_handoff" && ev.from && ev.to) {
          this.engine.spawnPacket({
            from: ev.from,
            to: ev.to,
            label: ev.label || ev.artifact,
            artifact: ev.artifact,
            sizeBytes: ev.size_bytes,
            color: ev.status === "alert" ? "#ef4444" : "#f5a623",
            durationMs: Math.max(600, stepMs * 1.1),
          });
        } else if (ev.type === "substage_progress" && ev.stage) {
          this.engine.showBubble(ev.stage, ev.message, stepMs * 1.5, "#38bdf8");
        } else if (ev.type === "boss_alert") {
          this.engine.showBubble("boss", ev.message || "Boss action needed!", stepMs * 2, "#f5a623");
        }

        if (idx === events.length - 1) {
          const endTimer = setTimeout(() => this.stop(), stepMs + 800);
          this.timers.push(endTimer);
        }
      }, idx * stepMs);

      this.timers.push(t);
    });
  }

  stop() {
    this.active = false;
    for (const t of this.timers) clearTimeout(t);
    this.timers = [];
    this.engine.setOverrideStatuses(null);
    this.onDone?.();
  }
}
