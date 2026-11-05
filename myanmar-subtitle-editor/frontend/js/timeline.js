/* ============================================================
   timeline.js — CapCut စတိုင် timeline
   • block များ ကပ်လျက် (segment.end === next.start)
   • ဘယ်/ညာ handle ဆွဲ၍ အချိန်ပြောင်း (ကပ်လျက် segment ပါ လိုက်ပြင်သည်)
   • block ကို ဆွဲ၍ ရွှေ့၊ ruler ကို နှိပ်၍ seek
   ============================================================ */
const Timeline = (() => {
  let root = null;
  let scrollEl = null;
  let canvasEl = null;
  let rulerEl = null;
  let tracksEl = null;
  let playheadEl = null;
  let zoomInput = null;
  let pxPerSec = 48;
  let timeLabel = null;
  let onSeek = null;
  let onSelect = null;

  const MIN_PX = 10;
  const MAX_PX = 320;

  function mount(element, options = {}) {
    root = element;
    onSeek = options.onSeek || (() => {});
    onSelect = options.onSelect || (() => {});
    scrollEl = root.querySelector(".timeline-scroll");
    canvasEl = root.querySelector(".timeline-canvas");
    rulerEl = root.querySelector(".ruler");
    tracksEl = root.querySelector(".tracks");
    playheadEl = root.querySelector(".playhead");
    zoomInput = root.querySelector(".zoom-range");
    timeLabel = root.querySelector('[data-timeline-time]');

    if (zoomInput) {
      zoomInput.addEventListener("input", () => {
        setPxPerSec(Number(zoomInput.value));
      });
    }
    const fitBtn = root.querySelector("[data-timeline-fit]");
    if (fitBtn) fitBtn.addEventListener("click", () => fit());

    scrollEl.addEventListener("pointerdown", (event) => {
      if (event.target.closest(".seg-block")) return;
      const time = eventToTime(event);
      if (time != null) onSeek(time);
    });

    Project.on("change", () => render());
    Project.on("select", () => render());
    if (window.ResizeObserver) {
      new ResizeObserver(() => {
        if (!pxPerSec) return;
        render();
      }).observe(scrollEl);
    }
    return api;
  }

  function duration() {
    return Math.max(0.1, Project.duration() || 10);
  }

  function widthPx() {
    return Math.max(scrollEl ? scrollEl.clientWidth : 600, duration() * pxPerSec);
  }

  function setPxPerSec(value) {
    pxPerSec = Math.min(MAX_PX, Math.max(MIN_PX, Number(value) || 48));
    if (zoomInput && Number(zoomInput.value) !== Math.round(pxPerSec)) {
      zoomInput.value = String(Math.round(pxPerSec));
    }
    render();
  }

  function fit() {
    if (!scrollEl) return;
    const available = Math.max(200, scrollEl.clientWidth - 24);
    setPxPerSec(available / duration());
  }

  function timeToX(time) {
    return Math.max(0, Number(time) || 0) * pxPerSec;
  }

  function eventToTime(event) {
    if (!tracksEl) return null;
    const rect = tracksEl.getBoundingClientRect();
    const x = event.clientX - rect.left;
    return Math.max(0, Math.min(duration(), x / pxPerSec));
  }

  function tickStep() {
    const candidates = [0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300];
    for (const step of candidates) {
      if (step * pxPerSec >= 66) return step;
    }
    return 600;
  }

  function renderRuler() {
    if (!rulerEl) return;
    const width = widthPx();
    const step = tickStep();
    const parts = [];
    for (let time = 0; time <= duration() + 0.001; time += step) {
      const x = timeToX(time);
      if (x > width + 40) break;
      parts.push(
        `<div class="ruler-tick" style="left:${x}px">${MSE.formatClock(time)}</div>`
      );
    }
    rulerEl.style.width = `${width}px`;
    rulerEl.innerHTML = parts.join("");
  }

  function blockForSegment(segment, index, activeId) {
    const left = timeToX(segment.start);
    const width = Math.max(12, timeToX(segment.end) - left);
    const active = segment.id === activeId ? " active" : "";
    return `
      <div class="seg-block${active}" data-segment="${segment.id}" style="left:${left}px;width:${width}px" title="${MSE.escapeHtml(
      segment.text
    )}">
        <div class="handle left" data-handle="left"></div>
        <div class="label">${index + 1}. ${MSE.escapeHtml(segment.text)}</div>
        <div class="handle right" data-handle="right"></div>
      </div>`;
  }

  function renderTracks() {
    if (!tracksEl) return;
    const doc = Project.doc();
    if (!doc) {
      tracksEl.innerHTML = "";
      return;
    }
    const selection = Project.selection();
    const width = widthPx();
    const activeId = selection.type === "subtitle" ? selection.id : null;
    const segmentHtml = doc.segments.map((segment, index) => blockForSegment(segment, index, activeId)).join("");
    const logoHtml = doc.logos
      .map(
        (logo, index) => `
      <div class="seg-block overlay-track${
        selection.type === "logo" && selection.id === logo.id ? " active" : ""
      }" data-logo="${logo.id}" style="left:0;width:${width}px;top:${18 + index * 20}px;height:22px">
        <div class="label">◈ ${MSE.escapeHtml(logo.name || "Logo")} ${logo.visible ? "" : "(ကွယ်)"}</div>
      </div>`
      )
      .join("");
    const blurHtml = doc.blurs
      .map(
        (box, index) => `
      <div class="seg-block blur-track${
        selection.type === "blur" && selection.id === box.id ? " active" : ""
      }" data-blur="${box.id}" style="left:0;width:${width}px;top:${18 + index * 20}px;height:22px">
        <div class="label">▒ Blur ${MSE.escapeHtml(String(box.intensity))}</div>
      </div>`
      )
      .join("");

    tracksEl.style.width = `${width}px`;
    tracksEl.innerHTML = `
      <div class="track x-track" style="height:${Math.max(78, 30 + doc.segments.length * 0 + 78)}px">
        <span class="track-label">စာတန်းထိုး (${doc.segments.length})</span>
        ${segmentHtml}
      </div>
      <div class="track-divider"></div>
      <div class="track" style="height:${Math.max(30, 26 + doc.logos.length * 20)}px">
        <span class="track-label">Logo (${doc.logos.length})</span>
        ${logoHtml}
      </div>
      <div class="track" style="height:${Math.max(30, 26 + doc.blurs.length * 20)}px">
        <span class="track-label">Blur (${doc.blurs.length})</span>
        ${blurHtml}
      </div>
      <div style="height:12px"></div>`;

    tracksEl.querySelectorAll(".seg-block").forEach((block) => {
      block.addEventListener("pointerdown", (event) => onBlockPointerDown(event, block));
    });
  }

  function onBlockPointerDown(event, block) {
    const segmentId = block.dataset.segment;
    const logoId = block.dataset.logo;
    const blurId = block.dataset.blur;
    event.stopPropagation();
    if (logoId) {
      Project.select({ type: "logo", id: logoId });
      return;
    }
    if (blurId) {
      Project.select({ type: "blur", id: blurId });
      return;
    }
    if (!segmentId) return;
    const segment = Project.doc().segments.find((item) => item.id === segmentId);
    if (!segment) return;
    Project.select({ type: "subtitle", id: segmentId });
    onSelect(segment);
    const handle = event.target.closest("[data-handle]");
    const mode = handle ? handle.dataset.handle : "move";
    if (!handle) onSeek(segment.start);

    const startX = event.clientX;
    const original = { start: segment.start, end: segment.end };
    let moved = false;
    Project.checkpoint();

    const move = (moveEvent) => {
      const delta = (moveEvent.clientX - startX) / pxPerSec;
      if (Math.abs(moveEvent.clientX - startX) > 2) moved = true;
      if (!moved) return;
      if (mode === "left") {
        Project.setSegmentStart(segmentId, original.start + delta, { history: false });
      } else if (mode === "right") {
        Project.setSegmentEnd(segmentId, original.end + delta, { history: false });
      } else {
        Project.moveSegment(segmentId, delta, { history: false });
      }
    };
    const up = () => {
      document.removeEventListener("pointermove", move);
      document.removeEventListener("pointerup", up);
      document.removeEventListener("pointercancel", up);
      document.body.style.cursor = "";
      Project.markDirty();
    };
    document.addEventListener("pointermove", move);
    document.addEventListener("pointerup", up);
    document.addEventListener("pointercancel", up);
    document.body.style.cursor = mode === "move" ? "grabbing" : "ew-resize";
  }

  function render() {
    if (!canvasEl) return;
    const doc = Project.doc();
    if (!doc) return;
    renderRuler();
    renderTracks();
    if (timeLabel) {
      timeLabel.textContent = `${MSE.formatTime(Project.playhead(), true)} / ${MSE.formatTime(
        Project.duration()
      )}`;
    }
    updatePlayhead(Project.playhead());
  }

  function updatePlayhead(time) {
    if (!playheadEl) return;
    const x = timeToX(time);
    playheadEl.style.left = `${x}px`;
    playheadEl.style.display = "block";
    if (timeLabel) {
      timeLabel.textContent = `${MSE.formatTime(time, true)} / ${MSE.formatTime(Project.duration())}`;
    }
  }

  function followPlayhead(time, force = false) {
    if (!scrollEl) return;
    const x = timeToX(time);
    const left = scrollEl.scrollLeft;
    const right = left + scrollEl.clientWidth;
    if (force || x < left + 20 || x > right - 40) {
      scrollEl.scrollLeft = Math.max(0, x - scrollEl.clientWidth * 0.35);
    }
  }

  const api = {
    mount,
    render,
    fit,
    setPxPerSec,
    updatePlayhead,
    followPlayhead,
    timeToX,
    pxPerSec: () => pxPerSec,
    scrollEl: () => scrollEl,
  };
  return api;
})();
