/**
 * WebSocket stream manager + Historical Replay Engine for Nepal News Bot HQ.
 */

export class OfficeWsClient {
  constructor({ onSnapshot, onOfficeEvent, onLogLines, onJobOutput, onConnectionChange, onReconnect }) {
    this.onSnapshot = onSnapshot;
    this.onOfficeEvent = onOfficeEvent;
    this.onLogLines = onLogLines;
    this.onJobOutput = onJobOutput;
    this.onConnectionChange = onConnectionChange;
    this.onReconnect = onReconnect;
    this.ws = null;
    this.currentDate = null;
    this.reconnectTimer = null;
    this.closedIntentionally = false;
    this.retryDelay = 500;
    this.heartbeatTimer = null;
    this.hasConnectedOnce = false;
  }

  _resetHeartbeat() {
    if (this.heartbeatTimer) clearTimeout(this.heartbeatTimer);
    // If no message or ping received within 15s, connection is presumed stale
    this.heartbeatTimer = setTimeout(() => {
      console.warn("WS heartbeat timed out after 15s. Reconnecting...");
      if (this.ws) {
        try {
          this.ws.close();
        } catch (_) {}
      }
    }, 15000);
  }

  connect(runDate) {
    this.disconnect();
    this.currentDate = runDate;
    this.closedIntentionally = false;

    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const urlParams = new URLSearchParams(window.location.search);
    const token = urlParams.get("token") || sessionStorage.getItem("office_token");
    if (urlParams.get("token")) {
      sessionStorage.setItem("office_token", urlParams.get("token"));
    }
    const tokenQuery = token ? `?token=${encodeURIComponent(token)}` : "";
    const url = `${proto}//${location.host}/ws/runs/${encodeURIComponent(runDate)}${tokenQuery}`;
    this.ws = new WebSocket(url);

    this.ws.onopen = () => {
      this.retryDelay = 500;
      this._resetHeartbeat();
      this.onConnectionChange?.(true);
      if (this.hasConnectedOnce) {
        // Resync on reconnection
        this.onReconnect?.();
        this.send({ type: "request_sync" });
      }
      this.hasConnectedOnce = true;
    };

    this.ws.onmessage = (evt) => {
      this._resetHeartbeat();
      try {
        const msg = JSON.parse(evt.data);
        if (msg.type === "ping") {
          this.send({ type: "pong" });
        } else if (msg.type === "state_snapshot") {
          this.onSnapshot?.(msg.data);
        } else if (msg.type === "office_event") {
          this.onOfficeEvent?.(msg.data);
        } else if (msg.type === "log_lines") {
          this.onLogLines?.(msg.data);
        } else if (msg.type === "job_output") {
          this.onJobOutput?.(msg.data);
        }
      } catch (err) {
        console.error("WS parse error", err);
      }
    };

    this.ws.onclose = () => {
      if (this.heartbeatTimer) clearTimeout(this.heartbeatTimer);
      this.onConnectionChange?.(false);
      if (!this.closedIntentionally && this.currentDate === runDate) {
        const nextDelay = this.retryDelay;
        this.retryDelay = Math.min(Math.round(this.retryDelay * 1.6), 10000);
        this.reconnectTimer = setTimeout(() => this.connect(runDate), nextDelay);
      }
    };

    this.ws.onerror = () => {
      this.ws?.close();
    };
  }

  send(data) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      try {
        this.ws.send(JSON.stringify(data));
      } catch (_) {}
    }
  }

  disconnect() {
    this.closedIntentionally = true;
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.heartbeatTimer) {
      clearTimeout(this.heartbeatTimer);
      this.heartbeatTimer = null;
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
