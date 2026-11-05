/* ============================================================
   project.js — ပရောဂျက် မော်ဒယ် (segments / logos / blurs)
   • CapCut စည်းကမ်း: segment[i].end === segment[i+1].start
   • Undo / Redo, Auto-save (30s), JSON save/load
   ============================================================ */
const Project = (() => {
  const MIN_DURATION = 0.2;
  const MAX_UNDO = 80;
  const AUTOSAVE_MS = 30000;

  const listeners = new Map();
  let history = [];
  let future = [];
  let autosaveTimer = null;
  let saveInFlight = null;

  const state = {
    doc: null,
    selection: { type: "none", id: null },
    dirty: false,
    saving: false,
    savedAt: 0,
    capabilities: null,
    playhead: 0,
  };

  /* ------------------------------- events ----------------------------- */
  function on(event, handler) {
    if (!listeners.has(event)) listeners.set(event, new Set());
    listeners.get(event).add(handler);
    return () => listeners.get(event).delete(handler);
  }
  function emit(event, payload) {
    const set = listeners.get(event);
    if (set) set.forEach((handler) => handler(payload));
  }

  /* ------------------------------- helpers ---------------------------- */
  const clone = (value) => JSON.parse(JSON.stringify(value));
  const round = (value, digits = 3) => Number(Number(value).toFixed(digits));
  const clamp = (value, low, high) => Math.min(Math.max(value, low), high);

  function defaultStyle() {
    const base = (state.capabilities && state.capabilities.config.default_style) || {};
    return {
      fontSize: base.fontSize ?? 0.055,
      x: base.x ?? 0.5,
      y: base.y ?? 0.86,
      width: base.width ?? 0.82,
      color: base.color ?? "#FFFFFF",
      outlineColor: base.outlineColor ?? "#000000",
      outlineWidth: base.outlineWidth ?? 0.08,
      opacity: base.opacity ?? 1,
      bold: base.bold ?? true,
      rotation: 0,
    };
  }

  function normaliseStyle(style) {
    const base = defaultStyle();
    const merged = { ...base, ...(style || {}) };
    return {
      fontSize: clamp(Number(merged.fontSize) || base.fontSize, 0.015, 0.3),
      x: clamp(Number(merged.x ?? 0.5), 0, 1),
      y: clamp(Number(merged.y ?? 0.86), 0, 1),
      width: clamp(Number(merged.width ?? 0.82), 0.05, 1),
      color: String(merged.color || "#FFFFFF"),
      outlineColor: String(merged.outlineColor || "#000000"),
      outlineWidth: clamp(Number(merged.outlineWidth) || 0, 0, 0.3),
      opacity: clamp(Number(merged.opacity ?? 1), 0, 1),
      bold: Boolean(merged.bold),
      rotation: Number(merged.rotation) || 0,
    };
  }

  function duration() {
    return Number((state.doc && state.doc.video && state.doc.video.duration) || 0);
  }

  function normaliseDoc(doc) {
    const target = doc || {};
    const video = target.video || {};
    target.video = {
      id: String(video.id || ""),
      filename: String(video.filename || ""),
      width: Number(video.width) || 0,
      height: Number(video.height) || 0,
      duration: Number(video.duration) || 0,
      has_audio: video.has_audio !== false,
    };
    target.name = String(target.name || "အမည်မသိ ပရောဂျက်");
    target.segments = (Array.isArray(target.segments) ? target.segments : [])
      .map((segment) => ({
        id: segment.id || MSE.uid("s"),
        start: Math.max(0, Number(segment.start) || 0),
        end: Math.max(0, Number(segment.end) || 0),
        text: MyanmarText.normalize(String(segment.text || "")),
        style: normaliseStyle(segment.style),
      }))
      .filter((segment) => segment.text && segment.end > segment.start)
      .sort((a, b) => a.start - b.start);
    target.logos = (Array.isArray(target.logos) ? target.logos : []).map((logo) => ({
      id: logo.id || MSE.uid("l"),
      type: "logo",
      src: String(logo.src || ""),
      url: logo.url || (logo.src ? `/api/media/${logo.src}` : ""),
      name: String(logo.name || "Logo"),
      x: clamp(Number(logo.x) || 0, -0.2, 1.2),
      y: clamp(Number(logo.y) || 0, -0.2, 1.2),
      w: clamp(Number(logo.w) || 0.15, 0.02, 1.5),
      h: clamp(Number(logo.h) || 0.15, 0.02, 1.5),
      rotation: Number(logo.rotation) || 0,
      opacity: clamp(Number(logo.opacity ?? 1), 0, 1),
      visible: logo.visible !== false,
      locked: Boolean(logo.locked),
    }));
    target.blurs = (Array.isArray(target.blurs) ? target.blurs : []).map((box) => ({
      id: box.id || MSE.uid("b"),
      type: "blur",
      name: String(box.name || "Blur"),
      x: clamp(Number(box.x) || 0, 0, 1),
      y: clamp(Number(box.y) || 0, 0, 1),
      w: clamp(Number(box.w ?? box.width) || 0.2, 0.01, 1),
      h: clamp(Number(box.h ?? box.height) || 0.1, 0.01, 1),
      intensity: clamp(Number(box.intensity) || 8, 1, 20),
      visible: box.visible !== false,
      locked: Boolean(box.locked),
    }));
    target.layer_order = Array.isArray(target.layer_order)
      ? target.layer_order
      : ["subtitle", ...target.logos.map((logo) => logo.id), ...target.blurs.map((box) => box.id)];
    target.style_defaults = normaliseStyle(target.style_defaults || {});
    // CapCut chain
    for (let index = 0; index < target.segments.length - 1; index += 1) {
      target.segments[index].end = round(target.segments[index + 1].start);
    }
    if (target.segments.length && target.video.duration > 0) {
      const last = target.segments[target.segments.length - 1];
      last.end = round(Math.min(last.end, target.video.duration));
      if (last.end <= last.start) last.end = round(Math.min(target.video.duration, last.start + 0.3));
    }
    return target;
  }

  /* ------------------------------ history ----------------------------- */
  function checkpoint() {
    if (!state.doc) return;
    history.push(clone(state.doc));
    if (history.length > MAX_UNDO) history.shift();
    future = [];
    emit("history", { canUndo: canUndo(), canRedo: canRedo() });
  }

  function canUndo() {
    return history.length > 0;
  }
  function canRedo() {
    return future.length > 0;
  }

  function undo() {
    if (!state.doc || !history.length) return false;
    future.push(clone(state.doc));
    state.doc = normaliseDoc(history.pop());
    ensureSelection();
    touch("undo");
    emit("history", { canUndo: canUndo(), canRedo: canRedo() });
    return true;
  }

  function redo() {
    if (!state.doc || !future.length) return false;
    history.push(clone(state.doc));
    state.doc = normaliseDoc(future.pop());
    ensureSelection();
    touch("redo");
    emit("history", { canUndo: canUndo(), canRedo: canRedo() });
    return true;
  }

  function touch(reason) {
    state.dirty = true;
    emit("change", { reason, doc: state.doc });
    emit("dirty", { dirty: true });
  }

  /* ------------------------------ selection --------------------------- */
  function select(selection) {
    const next = selection || { type: "none", id: null };
    state.selection = next;
    emit("select", next);
  }
  function selection() {
    return state.selection;
  }
  function selectedSegment() {
    if (!state.doc || state.selection.type !== "subtitle") return null;
    return state.doc.segments.find((segment) => segment.id === state.selection.id) || null;
  }
  function selectedLogo() {
    if (!state.doc || state.selection.type !== "logo") return null;
    return state.doc.logos.find((logo) => logo.id === state.selection.id) || null;
  }
  function selectedBlur() {
    if (!state.doc || state.selection.type !== "blur") return null;
    return state.doc.blurs.find((box) => box.id === state.selection.id) || null;
  }
  function ensureSelection() {
    if (!state.doc) return;
    const { type, id } = state.selection;
    if (type === "subtitle" && !state.doc.segments.some((segment) => segment.id === id)) {
      select({ type: "none", id: null });
    }
    if (type === "logo" && !state.doc.logos.some((logo) => logo.id === id)) {
      select({ type: "none", id: null });
    }
    if (type === "blur" && !state.doc.blurs.some((box) => box.id === id)) {
      select({ type: "none", id: null });
    }
  }

  /* ------------------------------ segments ---------------------------- */
  function segmentIndex(id) {
    if (!state.doc) return -1;
    return state.doc.segments.findIndex((segment) => segment.id === id);
  }

  function setSegmentStart(id, time, { history: withHistory = true } = {}) {
    const index = segmentIndex(id);
    if (index < 0) return;
    const segments = state.doc.segments;
    const segment = segments[index];
    const previous = index > 0 ? segments[index - 1] : null;
    const low = Math.max(
      0,
      previous ? previous.start + MIN_DURATION : 0,
      segment.end - Math.max(MIN_DURATION, Math.min(30, segment.end - segment.start))
    );
    const high = segment.end - MIN_DURATION;
    const value = round(clamp(time, low, high));
    if (previous && value - previous.start < MIN_DURATION) return;
    if (withHistory) checkpoint();
    segment.start = value;
    if (previous) previous.end = value;
    touch("segment-start");
  }

  function setSegmentEnd(id, time, { history: withHistory = true } = {}) {
    const index = segmentIndex(id);
    if (index < 0) return;
    const segments = state.doc.segments;
    const segment = segments[index];
    const next = index < segments.length - 1 ? segments[index + 1] : null;
    const low = segment.start + MIN_DURATION;
    const high = next
      ? next.end - MIN_DURATION
      : Math.max(low, duration() || segment.start + 30);
    const value = round(clamp(time, low, high));
    if (withHistory) checkpoint();
    segment.end = value;
    if (next) next.start = value;
    touch("segment-end");
  }

  function moveSegment(id, deltaSeconds, { history: withHistory = true } = {}) {
    const index = segmentIndex(id);
    if (index < 0) return;
    const segments = state.doc.segments;
    const segment = segments[index];
    const previous = index > 0 ? segments[index - 1] : null;
    const next = index < segments.length - 1 ? segments[index + 1] : null;
    const length = segment.end - segment.start;
    let low = previous ? previous.start + MIN_DURATION : 0;
    let high = next ? next.end - MIN_DURATION : duration() || segment.end + 30;
    let start = clamp(segment.start + deltaSeconds, low, high - length);
    start = round(Math.max(low, start));
    const end = round(start + length);
    if (withHistory) checkpoint();
    segment.start = start;
    segment.end = end;
    if (previous) previous.end = start;
    if (next) next.start = end;
    touch("segment-move");
  }

  function setSegmentText(id, text, { history: withHistory = false } = {}) {
    const segment = state.doc && state.doc.segments.find((item) => item.id === id);
    if (!segment) return;
    const clean = MyanmarText.normalize(text);
    if (clean === segment.text) return;
    if (withHistory) checkpoint();
    segment.text = clean;
    touch("segment-text");
  }

  function setSegmentStyle(id, patch, { history: withHistory = true } = {}) {
    const segment = state.doc && state.doc.segments.find((item) => item.id === id);
    if (!segment) return;
    if (withHistory) checkpoint();
    segment.style = normaliseStyle({ ...segment.style, ...patch });
    touch("segment-style");
  }

  function addSegmentAt(time, { text = "စာတန်းအသစ်", history: withHistory = true } = {}) {
    if (!state.doc) return null;
    const at = clamp(Number(time) || 0, 0, duration() || 3600);
    if (withHistory) checkpoint();
    const segments = state.doc.segments;
    const index = segments.findIndex((segment) => at >= segment.start && at < segment.end);
    let id;
    if (index >= 0) {
      const segment = segments[index];
      const ratio = (at - segment.start) / Math.max(0.001, segment.end - segment.start);
      const pieces = MyanmarText.splitAtSyllable(segment.text, ratio);
      const originalEnd = segment.end;
      segment.end = round(at);
      segment.text = pieces[0] || segment.text;
      id = MSE.uid("s");
      segments.splice(index + 1, 0, {
        id,
        start: round(at),
        end: round(originalEnd),
        text: pieces[1] || text,
        style: normaliseStyle(segment.style),
      });
    } else {
      const next = segments.find((segment) => segment.start > at);
      const start = round(at);
      const end = round(Math.min(next ? next.start : duration() || at + 2, start + 2));
      id = MSE.uid("s");
      const newSegment = {
        id,
        start,
        end: Math.max(start + MIN_DURATION, end),
        text,
        style: defaultStyle(),
      };
      const insertAt = segments.findIndex((segment) => segment.start > at);
      if (insertAt < 0) segments.push(newSegment);
      else segments.splice(insertAt, 0, newSegment);
    }
    touch("segment-add");
    select({ type: "subtitle", id });
    return id;
  }

  function splitSegment(id, time) {
    const index = segmentIndex(id);
    if (index < 0) return null;
    const segment = state.doc.segments[index];
    if (time <= segment.start + MIN_DURATION || time >= segment.end - MIN_DURATION) return null;
    checkpoint();
    const ratio = (time - segment.start) / Math.max(0.001, segment.end - segment.start);
    const pieces = MyanmarText.splitAtSyllable(segment.text, ratio);
    if (!pieces[1]) return null;
    const originalEnd = segment.end;
    segment.end = round(time);
    segment.text = pieces[0];
    const id2 = MSE.uid("s");
    state.doc.segments.splice(index + 1, 0, {
      id: id2,
      start: round(time),
      end: round(originalEnd),
      text: pieces[1],
      style: normaliseStyle(segment.style),
    });
    touch("segment-split");
    select({ type: "subtitle", id: id2 });
    return id2;
  }

  function mergeWithNext(id) {
    const index = segmentIndex(id);
    if (index < 0 || index >= state.doc.segments.length - 1) return false;
    checkpoint();
    const segment = state.doc.segments[index];
    const next = state.doc.segments[index + 1];
    segment.text = MyanmarText.joinTokens(segment.text, next.text);
    segment.end = next.end;
    state.doc.segments.splice(index + 1, 1);
    touch("segment-merge");
    select({ type: "subtitle", id });
    return true;
  }

  function deleteSegment(id) {
    const index = segmentIndex(id);
    if (index < 0) return false;
    checkpoint();
    const segments = state.doc.segments;
    const removed = segments[index];
    const previous = index > 0 ? segments[index - 1] : null;
    const next = index + 1 < segments.length ? segments[index + 1] : null;
    segments.splice(index, 1);
    if (previous) {
      previous.end = next ? round(next.start) : round(Math.min(duration() || removed.end + 3, removed.end + 3));
    }
    touch("segment-delete");
    if (state.selection.id === id) select({ type: "none", id: null });
    return true;
  }

  function replaceSegments(list, { keepStyle = true } = {}) {
    if (!state.doc) return [];
    checkpoint();
    const previous = state.doc.segments;
    state.doc.segments = (list || [])
      .map((item, index) => ({
        id: item.id || MSE.uid("s"),
        start: Math.max(0, Number(item.start) || 0),
        end: Math.max(0, Number(item.end) || 0),
        text: MyanmarText.normalize(String(item.text || "")),
        style:
          keepStyle && previous[index]
            ? previous[index].style
            : keepStyle && previous.length
            ? previous[previous.length - 1].style
            : normaliseStyle(item.style || {}),
      }))
      .filter((segment) => segment.text && segment.end > segment.start)
      .sort((a, b) => a.start - b.start);
    for (let index = 0; index < state.doc.segments.length - 1; index += 1) {
      state.doc.segments[index].end = round(state.doc.segments[index + 1].start);
    }
    select({ type: "subtitle", id: state.doc.segments[0] ? state.doc.segments[0].id : null });
    touch("segments-replace");
    return state.doc.segments;
  }

  function segmentAt(time) {
    if (!state.doc) return null;
    return (
      state.doc.segments.find((segment) => time >= segment.start - 1e-4 && time < segment.end) || null
    );
  }

  /* -------------------------------- logos ----------------------------- */
  function addLogo(logo) {
    if (!state.doc) return null;
    checkpoint();
    const id = logo.id || MSE.uid("l");
    const aspect = (state.doc.video.width || 16) / (state.doc.video.height || 9);
    let width = Number(logo.w) || 0.2;
    let height = Number(logo.h) || width;
    if (logo.naturalWidth && logo.naturalHeight) {
      height = (width * logo.naturalHeight) / logo.naturalWidth * aspect;
      height = clamp(height, 0.02, 0.95);
    }
    const item = {
      id,
      type: "logo",
      src: logo.src,
      url: logo.url || `/api/media/${logo.src}`,
      name: logo.name || "Logo",
      x: clamp(logo.x ?? 0.04, -0.2, 1.2),
      y: clamp(logo.y ?? 0.04, -0.2, 1.2),
      w: clamp(width, 0.02, 1.5),
      h: clamp(height, 0.02, 1.5),
      rotation: 0,
      opacity: 1,
      visible: true,
      locked: false,
    };
    state.doc.logos.push(item);
    state.doc.layer_order = ["subtitle", ...state.doc.logos.map((entry) => entry.id), ...state.doc.blurs.map((entry) => entry.id)];
    touch("logo-add");
    select({ type: "logo", id });
    return id;
  }

  function updateLogo(id, patch, { history: withHistory = true } = {}) {
    const logo = state.doc && state.doc.logos.find((item) => item.id === id);
    if (!logo) return;
    if (withHistory) checkpoint();
    Object.assign(logo, patch);
    logo.x = clamp(Number(logo.x) || 0, -0.2, 1.2);
    logo.y = clamp(Number(logo.y) || 0, -0.2, 1.2);
    logo.w = clamp(Number(logo.w) || 0.05, 0.02, 1.5);
    logo.h = clamp(Number(logo.h) || 0.05, 0.02, 1.5);
    logo.opacity = clamp(Number(logo.opacity ?? 1), 0, 1);
    logo.rotation = Number(logo.rotation) || 0;
    touch("logo-update");
  }

  function deleteLogo(id) {
    const index = state.doc.logos.findIndex((logo) => logo.id === id);
    if (index < 0) return false;
    checkpoint();
    state.doc.logos.splice(index, 1);
    state.doc.layer_order = state.doc.layer_order.filter((item) => item !== id);
    touch("logo-delete");
    if (state.selection.id === id) select({ type: "none", id: null });
    return true;
  }

  /* -------------------------------- blurs ----------------------------- */
  function addBlur(box = {}) {
    if (!state.doc) return null;
    checkpoint();
    const id = box.id || MSE.uid("b");
    const item = {
      id,
      type: "blur",
      name: box.name || "Blur",
      x: clamp(box.x ?? 0.32, 0, 0.99),
      y: clamp(box.y ?? 0.36, 0, 0.99),
      w: clamp(box.w ?? 0.28, 0.02, 1),
      h: clamp(box.h ?? 0.16, 0.02, 1),
      intensity: clamp(box.intensity ?? 10, 1, 20),
      visible: true,
      locked: false,
    };
    if (item.x + item.w > 1) item.x = 1 - item.w;
    if (item.y + item.h > 1) item.y = 1 - item.h;
    state.doc.blurs.push(item);
    state.doc.layer_order = ["subtitle", ...state.doc.logos.map((entry) => entry.id), ...state.doc.blurs.map((entry) => entry.id)];
    touch("blur-add");
    select({ type: "blur", id });
    return id;
  }

  function updateBlur(id, patch, { history: withHistory = true } = {}) {
    const box = state.doc && state.doc.blurs.find((item) => item.id === id);
    if (!box) return;
    if (withHistory) checkpoint();
    Object.assign(box, patch);
    box.x = clamp(Number(box.x) || 0, 0, 1);
    box.y = clamp(Number(box.y) || 0, 0, 1);
    box.w = clamp(Number(box.w) || 0.05, 0.01, 1);
    box.h = clamp(Number(box.h) || 0.05, 0.01, 1);
    box.intensity = clamp(Number(box.intensity) || 8, 1, 20);
    if (box.x + box.w > 1) box.x = Math.max(0, 1 - box.w);
    if (box.y + box.h > 1) box.y = Math.max(0, 1 - box.h);
    touch("blur-update");
  }

  function deleteBlur(id) {
    const index = state.doc.blurs.findIndex((box) => box.id === id);
    if (index < 0) return false;
    checkpoint();
    state.doc.blurs.splice(index, 1);
    state.doc.layer_order = state.doc.layer_order.filter((item) => item !== id);
    touch("blur-delete");
    if (state.selection.id === id) select({ type: "none", id: null });
    return true;
  }

  function toggleVisibility(type, id) {
    if (type === "logo") {
      const logo = state.doc.logos.find((item) => item.id === id);
      if (!logo) return;
      updateLogo(id, { visible: !logo.visible });
    } else if (type === "blur") {
      const box = state.doc.blurs.find((item) => item.id === id);
      if (!box) return;
      updateBlur(id, { visible: !box.visible });
    }
  }

  function toggleLock(type, id) {
    if (type === "logo") {
      const logo = state.doc.logos.find((item) => item.id === id);
      if (!logo) return;
      updateLogo(id, { locked: !logo.locked });
    } else if (type === "blur") {
      const box = state.doc.blurs.find((item) => item.id === id);
      if (!box) return;
      updateBlur(id, { locked: !box.locked });
    }
  }

  function reorder(id, direction) {
    const order = state.doc.layer_order.filter((item) => item !== "subtitle");
    const index = order.indexOf(id);
    if (index < 0) return;
    const target = direction === "up" ? index + 1 : index - 1;
    if (target < 0 || target >= order.length) return;
    checkpoint();
    order.splice(index, 1);
    order.splice(target, 0, id);
    state.doc.layer_order = ["subtitle", ...order];
    touch("layer-order");
  }

  function setDocument(doc, { resetHistory = true } = {}) {
    state.doc = normaliseDoc(doc);
    if (resetHistory) {
      history = [];
      future = [];
    }
    state.dirty = false;
    state.savedAt = 0;
    emit("change", { reason: "load", doc: state.doc });
    emit("history", { canUndo: canUndo(), canRedo: canRedo() });
    emit("dirty", { dirty: false });
    ensureSelection();
  }

  async function load(projectId) {
    const data = await MSE.json(`/api/project/${encodeURIComponent(projectId)}`);
    setDocument(data.project);
    return state.doc;
  }

  async function save({ silent = false } = {}) {
    if (!state.doc) return null;
    if (saveInFlight) return saveInFlight;
    const payload = toPayload();
    state.saving = true;
    emit("saving", { saving: true });
    saveInFlight = MSE.json("/api/project/save", { method: "POST", body: payload })
      .then((data) => {
        state.doc.id = data.project.id;
        state.doc.updated_at = data.project.updated_at;
        state.doc.warnings = data.project.warnings || [];
        state.dirty = false;
        state.savedAt = Date.now();
        state.saving = false;
        emit("saved", { project: data.project });
        emit("dirty", { dirty: false });
        emit("saving", { saving: false });
        if (!silent) MSE.toast("သိမ်းဆည်းပြီးပါပြီ", "ok");
        (data.project.warnings || []).forEach((warning) => MSE.toast(warning, "warn", 5200));
        return data.project;
      })
      .catch((error) => {
        state.saving = false;
        emit("saving", { saving: false });
        if (!silent) MSE.toast(error.message || "သိမ်းဆည်း၍ မရပါ", "bad", 6000);
        throw error;
      })
      .finally(() => {
        saveInFlight = null;
      });
    return saveInFlight;
  }

  function startAutosave() {
    stopAutosave();
    autosaveTimer = setInterval(() => {
      if (state.doc && state.dirty && !state.saving) {
        save({ silent: true }).catch(() => {});
      }
    }, AUTOSAVE_MS);
  }
  function stopAutosave() {
    if (autosaveTimer) clearInterval(autosaveTimer);
    autosaveTimer = null;
  }

  function toPayload() {
    if (!state.doc) return {};
    return {
      id: state.doc.id || undefined,
      name: state.doc.name,
      video: state.doc.video,
      segments: state.doc.segments.map((segment) => ({
        id: segment.id,
        start: round(segment.start),
        end: round(segment.end),
        text: segment.text,
        style: segment.style,
      })),
      logos: state.doc.logos.map((logo) => ({
        id: logo.id,
        src: logo.src,
        name: logo.name,
        x: logo.x,
        y: logo.y,
        w: logo.w,
        h: logo.h,
        rotation: logo.rotation,
        opacity: logo.opacity,
        visible: logo.visible,
        locked: logo.locked,
      })),
      blurs: state.doc.blurs,
      layer_order: state.doc.layer_order,
      style_defaults: state.doc.style_defaults,
      thumbnail: state.doc.thumbnail || "",
    };
  }

  function setCapabilities(capabilities) {
    state.capabilities = capabilities;
  }
  function capabilities() {
    return state.capabilities;
  }
  function doc() {
    return state.doc;
  }
  function isDirty() {
    return state.dirty;
  }
  function markDirty() {
    state.dirty = true;
    emit("dirty", { dirty: true });
  }
  function setPlayhead(time) {
    state.playhead = time;
  }
  function playhead() {
    return state.playhead;
  }

  return {
    MIN_DURATION,
    state,
    on,
    emit,
    doc,
    duration,
    select,
    selection,
    selectedSegment,
    selectedLogo,
    selectedBlur,
    checkpoint,
    undo,
    redo,
    canUndo,
    canRedo,
    segmentIndex,
    segmentAt,
    setSegmentStart,
    setSegmentEnd,
    moveSegment,
    setSegmentText,
    setSegmentStyle,
    addSegmentAt,
    replaceSegments,
    splitSegment,
    mergeWithNext,
    deleteSegment,
    addLogo,
    updateLogo,
    deleteLogo,
    addBlur,
    updateBlur,
    deleteBlur,
    toggleVisibility,
    toggleLock,
    reorder,
    setDocument,
    load,
    save,
    startAutosave,
    stopAutosave,
    toPayload,
    setCapabilities,
    capabilities,
    isDirty,
    markDirty,
    setPlayhead,
    playhead,
    defaultStyle,
    normaliseStyle,
    normaliseDoc,
  };
})();
