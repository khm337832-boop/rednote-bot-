/* ============================================================
   editor.js — တည်းဖြတ်ခန်း (Canvas + Timeline + Properties)
   ============================================================ */
(() => {
  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => Array.from(document.querySelectorAll(selector));

  const state = {
    capabilities: null,
    video: null,
    ticking: false,
    renderQueued: false,
    propsKey: "",
    exportJobId: null,
    transcribeJobId: null,
    started: false,
    lastPlayheadSave: 0,
  };

  /* ------------------------------- boot ------------------------------- */
  document.addEventListener("DOMContentLoaded", boot);

  async function boot() {
    try {
      await MSE.Auth.check();
    } catch (error) {
      /* auth disabled */
    }
    try {
      state.capabilities = await MSE.capabilities();
    } catch (error) {
      state.capabilities = null;
    }
    Project.setCapabilities(state.capabilities);
    bindChrome();
    bindKeyboard();
    await loadProject();
  }

  function projectIdFromUrl() {
    const match = /^\/editor\/([A-Za-z0-9_-]{6,64})\/?$/.exec(location.pathname);
    if (match) return match[1];
    const param = new URLSearchParams(location.search).get("project");
    return param || "";
  }

  async function loadProject() {
    const projectId = projectIdFromUrl();
    if (!projectId) {
      showEmpty("ပရောဂျက် မတွေ့ပါ။ <a href=\"/\">ပင်မ စာမျက်နှာ</a> မှ ပရောဂျက် ဖွင့်ပါ။");
      return;
    }
    try {
      await Project.load(projectId);
    } catch (error) {
      showEmpty(
        `${MSE.escapeHtml(error.message || "ပရောဂျက် ဖွင့်၍ မရပါ")}<br><a href="/">ပင်မ စာမျက်နှာ</a>`
      );
      return;
    }
    await startEditor();
  }

  function showEmpty(html) {
    const empty = $("#stage-empty");
    empty.classList.remove("hidden");
    empty.innerHTML = html;
    $("#stage").style.display = "none";
    Project.select({ type: "none", id: null });
    $$(".navbar-actions .btn").forEach((btn) => {
      if (!btn.dataset.themeToggle) btn.disabled = true;
    });
  }

  /* ---------------------------- editor start -------------------------- */
  async function startEditor() {
    const doc = Project.doc();
    $("#project-name").value = doc.name;
    state.video = $("#video");
    state.video.src = `/api/video/${encodeURIComponent(doc.video.id)}`;
    state.video.volume = 1;

    await CanvasView.waitForFonts();

    try {
      await waitForMetadata(state.video, 8000);
    } catch (error) {
      /* fall back to stored metadata */
    }
    if (!doc.video.duration && state.video.duration) {
      doc.video.duration = state.video.duration;
      doc.video.width = state.video.videoWidth || doc.video.width;
      doc.video.height = state.video.videoHeight || doc.video.height;
    }
    if (!doc.video.width && state.video.videoWidth) {
      doc.video.width = state.video.videoWidth;
      doc.video.height = state.video.videoHeight;
    }

    const stageEl = $("#stage");
    stageEl.style.display = "";
    $("#stage-empty").classList.add("hidden");
    CanvasView.init({ container: stageEl, video: state.video });
    CanvasView.resize();
    CanvasView.setTime(0);

    Timeline.mount($("#timeline"), {
      onSeek: (time) => seek(time),
      onSelect: (segment) => {
        seek(segment.start);
      },
    });
    Timeline.fit();

    bindVideo();
    bindEditorControls();
    subscribeProject();
    Project.startAutosave();

    const seekInput = $("#seek");
    seekInput.max = String(Project.duration() || 0);
    state.started = true;
    renderAll(true);
    updateChip();
    if (Project.capabilities()?.demo && !Project.capabilities()?.whisper?.available) {
      MSE.toast(
        "Whisper မရနိုင်သဖြင့် သရုပ်ပြ စာတန်းများ ဖြင့် စမ်းသပ်နိုင်ပါသည် (စစ်ဆေးရန် သာ)",
        "warn",
        7000
      );
    }
  }

  function waitForMetadata(video, timeout = 8000) {
    return new Promise((resolve, reject) => {
      if (video.readyState >= 1 && video.videoWidth) {
        resolve();
        return;
      }
      const timer = setTimeout(() => {
        cleanup();
        reject(new Error("timeout"));
      }, timeout);
      const onLoaded = () => {
        cleanup();
        resolve();
      };
      const onError = () => {
        cleanup();
        reject(new Error("video error"));
      };
      const cleanup = () => {
        clearTimeout(timer);
        video.removeEventListener("loadedmetadata", onLoaded);
        video.removeEventListener("error", onError);
      };
      video.addEventListener("loadedmetadata", onLoaded);
      video.addEventListener("error", onError);
    });
  }

  /* ------------------------------ chrome ------------------------------ */
  function bindChrome() {
    $("#btn-help")?.addEventListener("click", () => MSE.openModal("help-modal"));
    $$("[data-collapse]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const panel = document.getElementById(btn.dataset.collapse);
        if (panel) panel.classList.toggle("collapsed");
      });
    });
    $("#layer-list")?.addEventListener("click", onLayerClick);
    $("#props-body")?.addEventListener("click", onPropsClick);
    $("#props-body")?.addEventListener("input", onPropsInput);
    $("#props-body")?.addEventListener("focusin", (event) => {
      if (event.target.matches("[data-prop], [data-text]")) {
        if (!event.target.dataset.checkpointed) {
          Project.checkpoint();
          event.target.dataset.checkpointed = "1";
        }
      }
    });
    $("#props-body")?.addEventListener("focusout", (event) => {
      if (event.target.dataset) delete event.target.dataset.checkpointed;
    });
    window.addEventListener("resize", () => {
      if (state.started) CanvasView.resize();
    });
    window.addEventListener("beforeunload", (event) => {
      if (Project.isDirty()) {
        event.preventDefault();
        event.returnValue = "";
      }
    });
  }

  function bindEditorControls() {
    $("#btn-save")?.addEventListener("click", () => {
      Project.save().catch(() => {});
    });
    $("#btn-export")?.addEventListener("click", runExport);
    $("#btn-srt")?.addEventListener("click", downloadSrt);
    $("#btn-transcribe")?.addEventListener("click", runTranscribe);
    const openLogo = () => $("#logo-input")?.click();
    $("#btn-logo")?.addEventListener("click", openLogo);
    $("#btn-logo-2")?.addEventListener("click", openLogo);
    $("#logo-input")?.addEventListener("change", onLogoChosen);
    const addBlur = () => {
      Project.addBlur({ x: 0.34, y: 0.38, w: 0.3, h: 0.16, intensity: 10 });
      MSE.toast("Blur box ထည့်ပြီးပါပြီ — ဆွဲ၍ နေရာချပါ", "ok");
    };
    $("#btn-blur")?.addEventListener("click", addBlur);
    $("#btn-blur-2")?.addEventListener("click", addBlur);
    $("#btn-add-segment")?.addEventListener("click", addSegmentAtPlayhead);
    $("#btn-undo")?.addEventListener("click", () => {
      if (!Project.undo()) MSE.toast("နောက်ပြန်ရန် မရှိပါ", "warn", 1800);
    });
    $("#btn-redo")?.addEventListener("click", () => {
      if (!Project.redo()) MSE.toast("ရှေ့သို့ ရန် မရှိပါ", "warn", 1800);
    });
    $("#tl-split")?.addEventListener("click", splitAtPlayhead);
    $("#tl-merge")?.addEventListener("click", mergeSelected);
    $("#tl-delete")?.addEventListener("click", deleteSelected);
    $("#project-name")?.addEventListener("input", (event) => {
      if (!Project.doc()) return;
      Project.doc().name = event.target.value;
      Project.markDirty();
    });
    $("#project-name")?.addEventListener("change", () => {
      Project.save({ silent: true }).catch(() => {});
    });
    $("#transcribe-cancel")?.addEventListener("click", async () => {
      if (state.transcribeJobId) {
        try {
          await MSE.json(`/api/jobs/${state.transcribeJobId}/cancel`, { method: "POST" });
        } catch (error) {
          /* ignore */
        }
      }
    });
  }

  function bindVideo() {
    const video = state.video;
    video.addEventListener("play", () => {
      updatePlayButton();
      startTicker();
    });
    video.addEventListener("pause", updatePlayButton);
    video.addEventListener("ended", updatePlayButton);
    video.addEventListener("seeked", () => {
      if (!state.ticking) updateFromVideo();
    });
    video.addEventListener("error", () => {
      MSE.toast("ဗီဒီယိုဖိုင် ဖွင့်၍ မရပါ — ပြန်တင်ရန် လိုနိုင်ပါသည်", "bad", 6000);
    });

    $("#btn-play")?.addEventListener("click", togglePlay);
    $("#seek")?.addEventListener("input", (event) => {
      seek(Number(event.target.value));
    });
    $("#volume")?.addEventListener("input", (event) => {
      video.volume = Number(event.target.value);
      $("#btn-mute").textContent = video.volume === 0 ? "🔇" : "🔊";
    });
    $("#btn-mute")?.addEventListener("click", () => {
      video.muted = !video.muted;
      $("#btn-mute").textContent = video.muted ? "🔇" : "🔊";
    });
    updatePlayButton();
  }

  function subscribeProject() {
    Project.on("change", () => scheduleRender());
    Project.on("select", () => scheduleRender());
    Project.on("saving", updateSaveIndicator);
    Project.on("saved", updateSaveIndicator);
    Project.on("dirty", updateSaveIndicator);
    Project.on("history", updateUndoButtons);
  }

  /* ----------------------------- rendering ---------------------------- */
  function scheduleRender() {
    if (state.renderQueued) return;
    state.renderQueued = true;
    requestAnimationFrame(() => {
      state.renderQueued = false;
      renderAll(false);
    });
  }

  function renderAll(force) {
    if (!state.started) return;
    CanvasView.sync();
    Timeline.render();
    renderLayers();
    renderProps(force);
    updateSaveIndicator();
    updateUndoButtons();
    updateChip();
  }

  function updateChip() {
    const chip = $("#stage-chip");
    if (!chip || !Project.doc()) return;
    const doc = Project.doc();
    chip.textContent = `စာတန်း ${doc.segments.length} · Logo ${doc.logos.length} · Blur ${doc.blurs.length}`;
  }

  function renderLayers() {
    const list = $("#layer-list");
    const doc = Project.doc();
    if (!list || !doc) return;
    const selection = Project.selection();
    const segments = doc.segments
      .map(
        (segment, index) => `
      <div class="segment-chip${
        selection.type === "subtitle" && selection.id === segment.id ? " active" : ""
      }" data-segment="${segment.id}">
        <span class="time">${MSE.formatTime(segment.start, true)}</span>
        <span class="txt">${index + 1}. ${MSE.escapeHtml(segment.text)}</span>
      </div>`
      )
      .join("");

    const logoItems = doc.logos
      .map(
        (logo) => `
      <div class="layer-item${logo.visible ? "" : " muted"}${
          selection.type === "logo" && selection.id === logo.id ? " active" : ""
        }" data-layer-logo="${logo.id}">
        <span class="li-icon">◈</span>
        <div class="li-text">
          <div class="li-title">${MSE.escapeHtml(logo.name || "Logo")}</div>
          <div class="li-sub">${Math.round(logo.w * 100)}% · ${Math.round(logo.opacity * 100)}%</div>
        </div>
        <div class="li-actions">
          <button data-act="visible" data-type="logo" data-id="${logo.id}" title="ပြ/ကွယ်">${
          logo.visible ? "👁" : "🚫"
        }</button>
          <button data-act="lock" data-type="logo" data-id="${logo.id}" title="သော့">${
          logo.locked ? "🔒" : "🔓"
        }</button>
          <button data-act="up" data-id="${logo.id}" title="ရှေ့သို့">▲</button>
          <button data-act="down" data-id="${logo.id}" title="နောက်သို့">▼</button>
          <button data-act="delete" data-type="logo" data-id="${logo.id}" title="ဖျက်">✕</button>
        </div>
      </div>`
      )
      .join("");

    const blurItems = doc.blurs
      .map(
        (box) => `
      <div class="layer-item${box.visible ? "" : " muted"}${
          selection.type === "blur" && selection.id === box.id ? " active" : ""
        }" data-layer-blur="${box.id}">
        <span class="li-icon">▒</span>
        <div class="li-text">
          <div class="li-title">Blur ${Math.round(box.intensity)}</div>
          <div class="li-sub">${Math.round(box.w * 100)}% × ${Math.round(box.h * 100)}%</div>
        </div>
        <div class="li-actions">
          <button data-act="visible" data-type="blur" data-id="${box.id}" title="ပြ/ကွယ်">${
          box.visible ? "👁" : "🚫"
        }</button>
          <button data-act="lock" data-type="blur" data-id="${box.id}" title="သော့">${
          box.locked ? "🔒" : "🔓"
        }</button>
          <button data-act="delete" data-type="blur" data-id="${box.id}" title="ဖျက်">✕</button>
        </div>
      </div>`
      )
      .join("");

    list.innerHTML = `
      <div class="layer-group">
        <div class="layer-item${selection.type === "subtitle" ? " active" : ""}" data-layer-subtitle="1">
          <span class="li-icon">字</span>
          <div class="li-text">
            <div class="li-title">စာတန်းထိုး</div>
            <div class="li-sub">${doc.segments.length} ခု · နောက်စကားအထိ ဆက်ပေါ်</div>
          </div>
          <div class="li-actions">
            <button data-act="add-segment" title="စာတန်းအသစ်">＋</button>
          </div>
        </div>
        <div class="segment-chip-row">${segments || '<span class="text-faint" style="font-size:0.76rem">စာတန်း မရှိသေးပါ</span>'}</div>
      </div>
      <div class="layer-group">
        <div class="layer-item" data-layer-logo-group="1">
          <span class="li-icon">◈</span>
          <div class="li-text"><div class="li-title">Logo</div><div class="li-sub">${doc.logos.length} ခု</div></div>
          <div class="li-actions"><button data-act="add-logo" title="Logo ထည့်">＋</button></div>
        </div>
        ${logoItems}
      </div>
      <div class="layer-group">
        <div class="layer-item" data-layer-blur-group="1">
          <span class="li-icon">▒</span>
          <div class="li-text"><div class="li-title">Blur</div><div class="li-sub">${doc.blurs.length} ခု</div></div>
          <div class="li-actions"><button data-act="add-blur" title="Blur ထည့်">＋</button></div>
        </div>
        ${blurItems}
      </div>`;
  }

  /* ----------------------------- properties --------------------------- */
  function fieldNumber(
    label,
    type,
    field,
    value,
    { step = "1", min = 0, max = 100, unit = "%", scale = 100 } = {}
  ) {
    return `<div class="field"><label>${label}</label>
      <div class="num-row">
        <input class="input" type="number" step="${step}" min="${min}" max="${max}"
          data-prop="${type}" data-field="${field}" data-scale="${scale}" data-unit="${unit}"
          value="${Number(value).toFixed(1)}">
        <span class="unit">${unit}</span>
      </div></div>`;
  }

  function fieldRange(label, type, field, value, { min = 0, max = 100, step = "1", scale = 100, unit = "%" } = {}) {
    return `<div class="field">
      <label>${label} <span class="text-faint" data-out="${field}">${Number(value).toFixed(1)}${unit}</span></label>
      <input type="range" min="${min}" max="${max}" step="${step}"
        data-prop="${type}" data-field="${field}" data-scale="${scale}" data-unit="${unit}"
        value="${Number(value)}">
    </div>`;
  }

  function buildSegmentProps(segment) {
    const index = Project.doc().segments.findIndex((item) => item.id === segment.id);
    const style = segment.style;
    return `
      <div class="props-section">
        <div class="props-head">စာတန်းထိုး <span class="chip">#${index + 1}</span></div>
        <div class="field">
          <label>စာသား (မြန်မာ Unicode)</label>
          <textarea class="textarea" rows="3" data-text="1" spellcheck="false">${MSE.escapeHtml(
            segment.text
          )}</textarea>
        </div>
        <div class="field-row">
          <div class="field"><label>စတင်ချိန်</label>
            <input class="input" type="text" data-prop="segment" data-field="start" data-time="1"
              value="${MSE.formatTime(segment.start, true)}"></div>
          <div class="field"><label>ဆုံးချိန်</label>
            <input class="input" type="text" data-prop="segment" data-field="end" data-time="1"
              value="${MSE.formatTime(segment.end, true)}"></div>
        </div>
        <p class="text-faint" style="font-size:0.72rem;margin:-4px 0 8px">
          ဆုံးချိန် ပြောင်းလျှင် နောက် စာတန်း၏ စတင်ချိန် ပါ လိုက်ပြင်ပါသည် (CapCut rule)။
        </p>
        ${fieldRange("ဖောင့် အရွယ်", "segment", "fontSize", style.fontSize * 100, {
          min: 1.5,
          max: 30,
          step: "0.1",
          scale: 100,
        })}
        ${fieldNumber("ဘောင် အထူ", "segment", "outlineWidth", style.outlineWidth * 100, {
          step: "1",
          min: 0,
          max: 30,
        })}
        <div class="field-row">
          <div class="field"><label>စာသား အရောင်</label>
            <input type="color" data-prop="segment" data-field="color" data-color="1" value="${style.color}"></div>
          <div class="field"><label>ဘောင် အရောင်</label>
            <input type="color" data-prop="segment" data-field="outlineColor" data-color="1" value="${style.outlineColor}"></div>
        </div>
        ${fieldNumber("အလျား (box)", "segment", "width", style.width * 100, { step: "1", min: 5, max: 100 })}
        <div class="field-row">
          ${fieldNumber("X", "segment", "x", style.x * 100, { step: "1", min: 0, max: 100 })}
          ${fieldNumber("Y", "segment", "y", style.y * 100, { step: "1", min: 0, max: 100 })}
        </div>
        ${fieldNumber("လှည့်ခြင်း", "segment", "rotation", style.rotation, {
          step: "1",
          min: -180,
          max: 180,
          unit: "°",
          scale: 1,
        })}
        ${fieldRange("အလင်း အနေအထား", "segment", "opacity", style.opacity * 100, {
          min: 0,
          max: 100,
          step: "1",
        })}
        <label class="check" style="margin-bottom:10px">
          <input type="checkbox" data-prop="segment" data-field="bold" data-check="1" ${
            style.bold ? "checked" : ""
          }> စာလုံး အထူ
        </label>
        <div class="row">
          <button class="btn btn-sm" data-action="split">✂ ခွဲရန်</button>
          <button class="btn btn-sm" data-action="merge">⇔ ပူးရန်</button>
          <button class="btn btn-sm btn-danger" data-action="delete">🗑 ဖျက်ရန်</button>
        </div>
      </div>`;
  }

  function buildLogoProps(logo) {
    return `
      <div class="props-section">
        <div class="props-head">Logo <span class="chip">${MSE.escapeHtml(logo.name || "")}</span></div>
        <div class="field-row">
          ${fieldNumber("X", "logo", "x", logo.x * 100, { min: -20, max: 120 })}
          ${fieldNumber("Y", "logo", "y", logo.y * 100, { min: -20, max: 120 })}
        </div>
        <div class="field-row">
          ${fieldNumber("အလျား", "logo", "w", logo.w * 100, { step: "1", min: 2, max: 150 })}
          ${fieldNumber("အမြင့်", "logo", "h", logo.h * 100, { step: "1", min: 2, max: 150 })}
        </div>
        ${fieldNumber("လှည့်ခြင်း", "logo", "rotation", logo.rotation, {
          step: "1",
          min: -180,
          max: 180,
          unit: "°",
          scale: 1,
        })}
        ${fieldRange("အလင်း အနေအထား", "logo", "opacity", logo.opacity * 100, {
          min: 0,
          max: 100,
          step: "1",
        })}
        <div class="row" style="margin-bottom:8px">
          <label class="check"><input type="checkbox" data-prop="logo" data-field="visible" data-check="1" ${
            logo.visible ? "checked" : ""
          }> ပြရန်</label>
          <label class="check"><input type="checkbox" data-prop="logo" data-field="locked" data-check="1" ${
            logo.locked ? "checked" : ""
          }> သော့ခတ်</label>
        </div>
        <div class="row">
          <button class="btn btn-sm" data-action="front">ရှေ့သို့ ပို့</button>
          <button class="btn btn-sm" data-action="back">နောက်သို့ ပို့</button>
          <button class="btn btn-sm btn-danger" data-action="delete">🗑 ဖျက်ရန်</button>
        </div>
      </div>`;
  }

  function buildBlurProps(box) {
    return `
      <div class="props-section">
        <div class="props-head">Blur <span class="chip">${Math.round(box.intensity)}</span></div>
        <div class="field-row">
          ${fieldNumber("X", "blur", "x", box.x * 100, { min: 0, max: 100 })}
          ${fieldNumber("Y", "blur", "y", box.y * 100, { min: 0, max: 100 })}
        </div>
        <div class="field-row">
          ${fieldNumber("အလျား", "blur", "w", box.w * 100, { step: "1", min: 1, max: 100 })}
          ${fieldNumber("အမြင့်", "blur", "h", box.h * 100, { step: "1", min: 1, max: 100 })}
        </div>
        ${fieldRange("ဝါးမှု အားး (1–20)", "blur", "intensity", box.intensity, {
          min: 1,
          max: 20,
          step: "1",
          scale: 1,
          unit: "",
        })}
        <div class="row" style="margin-bottom:8px">
          <label class="check"><input type="checkbox" data-prop="blur" data-field="visible" data-check="1" ${
            box.visible ? "checked" : ""
          }> ပြရန်</label>
          <label class="check"><input type="checkbox" data-prop="blur" data-field="locked" data-check="1" ${
            box.locked ? "checked" : ""
          }> သော့ခတ်</label>
        </div>
        <button class="btn btn-sm btn-danger" data-action="delete">🗑 ဖျက်ရန်</button>
      </div>`;
  }

  function buildEmptyProps() {
    return `
      <p class="props-empty">
        အလွှာတစ်ခုကို ရွေးပါ — Canvas ပေါ်တွင် နှိပ်၍ ဖြစ်စေ၊ Timeline မှ ဖြစ်စေ ရွေးနိုင်သည်။
        ရွေးပြီးပါက နေရာ၊ အရွယ်၊ လှည့်ခြင်း၊ အလင်း အနေအထားနှင့် စာသားကို ဤနေရာတွင်
        အတိအကျ ထည့်နိုင်ပါသည်။
      </p>
      <div class="divider"></div>
      <ul class="status-list">
        <li><span class="icon">字</span><span>စာတန်းထိုး — စာသားနှင့် အချိန်ကို Timeline တွင် ပြင်ပါ</span></li>
        <li><span class="icon">◈</span><span>Logo — PNG/JPG/SVG တင်ပြီး ဆွဲ/ချဲ့/လှည့်နိုင်သည်</span></li>
        <li><span class="icon">▒</span><span>Blur — ဝါးစေလိုသည့် နေရာကို ရွေးပြီး အားး ချိန်နိုင်သည်</span></li>
      </ul>`;
  }

  function renderProps(force) {
    const body = $("#props-body");
    if (!body) return;
    const selection = Project.selection();
    const key = `${selection.type}:${selection.id || ""}`;
    if (!force && key === state.propsKey) {
      updatePropsValues();
      return;
    }
    state.propsKey = key;
    if (selection.type === "subtitle") {
      const segment = Project.selectedSegment();
      body.innerHTML = segment ? buildSegmentProps(segment) : buildEmptyProps();
    } else if (selection.type === "logo") {
      const logo = Project.selectedLogo();
      body.innerHTML = logo ? buildLogoProps(logo) : buildEmptyProps();
    } else if (selection.type === "blur") {
      const box = Project.selectedBlur();
      body.innerHTML = box ? buildBlurProps(box) : buildEmptyProps();
    } else {
      body.innerHTML = buildEmptyProps();
    }
  }

  function updatePropsValues() {
    const body = $("#props-body");
    if (!body) return;
    const selection = Project.selection();
    let source = null;
    if (selection.type === "subtitle") source = Project.selectedSegment();
    if (selection.type === "logo") source = Project.selectedLogo();
    if (selection.type === "blur") source = Project.selectedBlur();
    if (!source) return;
    body.querySelectorAll("[data-prop]").forEach((input) => {
      if (input === document.activeElement) return;
      const field = input.dataset.field;
      let value;
      if (selection.type === "segment") {
        value = field === "start" || field === "end" ? source[field] : source.style[field];
      } else {
        value = source[field];
      }
      if (value == null) return;
      if (input.type === "checkbox") {
        input.checked = Boolean(value);
        return;
      }
      if (input.type === "color") {
        input.value = String(value);
        return;
      }
      if (input.dataset.time) {
        input.value = MSE.formatTime(value, true);
        return;
      }
      const scale = Number(input.dataset.scale || 1);
      const display = value * scale;
      if (input.type === "range" || input.type === "number") {
        input.value = String(Number(display.toFixed(2)));
      } else {
        input.value = String(display);
      }
      const out = body.querySelector(`[data-out="${field}"]`);
      if (out) out.textContent = `${Number(display.toFixed(1))}${input.dataset.unit || ""}`;
    });
    const textarea = body.querySelector("[data-text]");
    if (textarea && textarea !== document.activeElement) textarea.value = source.text || "";
  }

  function onPropsClick(event) {
    const actionBtn = event.target.closest("[data-action]");
    if (!actionBtn) return;
    const action = actionBtn.dataset.action;
    if (action === "split") return splitAtPlayhead();
    if (action === "merge") return mergeSelected();
    if (action === "delete") return deleteSelected();
    const selection = Project.selection();
    if (selection.type === "logo" && (action === "front" || action === "back")) {
      Project.reorder(selection.id, action === "front" ? "up" : "down");
    }
  }

  function onPropsInput(event) {
    const target = event.target;
    if (target.matches("[data-text]")) {
      const segment = Project.selectedSegment();
      if (segment) Project.setSegmentText(segment.id, target.value, { history: false });
      return;
    }
    const type = target.dataset.prop;
    const field = target.dataset.field;
    if (!type || !field) return;
    const selection = Project.selection();
    if (!selection.id) return;

    if (target.dataset.time) {
      const seconds = MSE.parseClock(target.value);
      if (seconds == null) return;
      if (type === "segment") {
        if (field === "start") Project.setSegmentStart(selection.id, seconds, { history: false });
        else Project.setSegmentEnd(selection.id, seconds, { history: false });
        seek(seconds);
      }
      return;
    }
    if (target.type === "checkbox") {
      if (type === "segment") Project.setSegmentStyle(selection.id, { [field]: target.checked }, { history: false });
      if (type === "logo") Project.updateLogo(selection.id, { [field]: target.checked }, { history: false });
      if (type === "blur") Project.updateBlur(selection.id, { [field]: target.checked }, { history: false });
      return;
    }
    if (target.type === "color") {
      Project.setSegmentStyle(selection.id, { [field]: target.value }, { history: false });
      return;
    }
    const raw = Number(target.value);
    if (Number.isNaN(raw)) return;
    const scale = Number(target.dataset.scale || 1);
    const value = scale === 0 ? raw : raw / scale;
    if (type === "segment") {
      Project.setSegmentStyle(selection.id, { [field]: value }, { history: false });
      if (field === "x" || field === "y") CanvasView.applySelection();
      return;
    }
    if (type === "logo") {
      Project.updateLogo(selection.id, { [field]: value }, { history: false });
      return;
    }
    if (type === "blur") {
      Project.updateBlur(selection.id, { [field]: value }, { history: false });
    }
  }

  function onLayerClick(event) {
    const actionBtn = event.target.closest("[data-act]");
    if (actionBtn) {
      event.stopPropagation();
      const action = actionBtn.dataset.act;
      const type = actionBtn.dataset.type;
      const id = actionBtn.dataset.id;
      if (action === "add-logo" || action === "add-segment") {
        if (action === "add-logo") $("#logo-input").click();
        else addSegmentAtPlayhead();
        return;
      }
      if (action === "add-blur") {
        Project.addBlur();
        return;
      }
      if (action === "visible") return Project.toggleVisibility(type, id);
      if (action === "lock") return Project.toggleLock(type, id);
      if (action === "delete") {
        if (type === "logo") Project.deleteLogo(id);
        if (type === "blur") Project.deleteBlur(id);
        return;
      }
      if (action === "up" || action === "down") {
        Project.reorder(id, action);
        return;
      }
    }
    const segmentChip = event.target.closest("[data-segment]");
    if (segmentChip) {
      const id = segmentChip.dataset.segment;
      Project.select({ type: "subtitle", id });
      const segment = Project.doc().segments.find((item) => item.id === id);
      if (segment) seek(segment.start);
      return;
    }
    const logoItem = event.target.closest("[data-layer-logo]");
    if (logoItem) {
      Project.select({ type: "logo", id: logoItem.dataset.layerLogo });
      return;
    }
    const blurItem = event.target.closest("[data-layer-blur]");
    if (blurItem) {
      Project.select({ type: "blur", id: blurItem.dataset.layerBlur });
      return;
    }
    const subtitleLayer = event.target.closest("[data-layer-subtitle]");
    if (subtitleLayer) {
      const first = Project.doc().segments[0];
      if (first) {
        Project.select({ type: "subtitle", id: first.id });
        seek(first.start);
      } else {
        Project.select({ type: "none", id: null });
      }
    }
  }

  /* ------------------------------ playback ---------------------------- */
  function startTicker() {
    if (state.ticking) return;
    state.ticking = true;
    const tick = () => {
      if (!state.video || state.video.paused) {
        state.ticking = false;
        updateFromVideo();
        return;
      }
      updateFromVideo();
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }

  function updateFromVideo() {
    const time = state.video ? state.video.currentTime : 0;
    Project.setPlayhead(time);
    CanvasView.setTime(time);
    Timeline.updatePlayhead(time);
    if (!state.video.paused) Timeline.followPlayhead(time);
    const seekInput = $("#seek");
    if (seekInput && document.activeElement !== seekInput) seekInput.value = String(time);
    const label = $("#time-label");
    if (label) {
      label.textContent = `${MSE.formatTime(time, true)} / ${MSE.formatTime(Project.duration())}`;
    }
  }

  function seek(time) {
    if (!state.video) return;
    const duration = Project.duration() || state.video.duration || 0;
    const target = Math.max(0, Math.min(duration || 0, Number(time) || 0));
    try {
      state.video.currentTime = target;
    } catch (error) {
      /* ignore */
    }
    Project.setPlayhead(target);
    CanvasView.setTime(target);
    Timeline.updatePlayhead(target);
    Timeline.followPlayhead(target, true);
    const seekInput = $("#seek");
    if (seekInput) seekInput.value = String(target);
    const label = $("#time-label");
    if (label) {
      label.textContent = `${MSE.formatTime(target, true)} / ${MSE.formatTime(Project.duration())}`;
    }
    requestAnimationFrame(() => {
      if (Project.selection().type === "none") CanvasView.setTime(target);
    });
  }

  function togglePlay() {
    if (!state.video) return;
    if (state.video.paused) {
      state.video.play().catch(() => MSE.toast("ဗီဒီယို ဖွင့်၍ မရပါ", "bad"));
    } else {
      state.video.pause();
    }
  }

  function updatePlayButton() {
    const btn = $("#btn-play");
    if (btn) btn.textContent = state.video && !state.video.paused ? "❚❚" : "▶";
  }

  /* ------------------------------ actions ----------------------------- */
  function addSegmentAtPlayhead() {
    if (!Project.doc()) return;
    const time = Project.playhead();
    const id = Project.addSegmentAt(time, { text: "စာတန်းအသစ်" });
    if (id) {
      seek(time);
      MSE.toast("စာတန်း အသစ် ထည့်ပြီးပါပြီ — စာသား ပြင်နိုင်ပါသည်", "ok", 2400);
      setTimeout(() => {
        const textarea = document.querySelector("[data-text]");
        if (textarea) textarea.focus();
      }, 120);
    }
  }

  function splitAtPlayhead() {
    const selection = Project.selection();
    const time = Project.playhead();
    let id = selection.type === "subtitle" ? selection.id : null;
    if (!id) {
      const segment = Project.segmentAt(time);
      if (segment) id = segment.id;
    }
    if (!id) {
      MSE.toast("ဤနေရာတွင် ခွဲရန် စာတန်း မရှိပါ", "warn", 2400);
      return;
    }
    const newId = Project.splitSegment(id, time);
    if (newId) MSE.toast("စာတန်း ခွဲပြီးပါပြီ", "ok", 2000);
    else MSE.toast("ခွဲရန် နေရာ မသင့်လျော်ပါ (အနည်းဆုံး ၀.၂ စက္ကန့် လိုသည်)", "warn", 3000);
  }

  function mergeSelected() {
    const selection = Project.selection();
    if (selection.type !== "subtitle") {
      MSE.toast("ပူးရန် စာတန်းကို အရင် ရွေးပါ", "warn", 2200);
      return;
    }
    if (Project.mergeWithNext(selection.id)) MSE.toast("ပူးပြီးပါပြီ", "ok", 1800);
    else MSE.toast("နောက် စာတန်း မရှိပါ", "warn", 2200);
  }

  function deleteSelected() {
    const selection = Project.selection();
    if (selection.type === "segment" || selection.type === "none") return;
    if (selection.type === "subtitle") {
      Project.deleteSegment(selection.id);
      MSE.toast("စာတန်း ဖျက်ပြီးပါပြီ", "ok", 1800);
    } else if (selection.type === "logo") {
      Project.deleteLogo(selection.id);
      MSE.toast("Logo ဖျက်ပြီးပါပြီ", "ok", 1800);
    } else if (selection.type === "blur") {
      Project.deleteBlur(selection.id);
      MSE.toast("Blur ဖျက်ပြီးပါပြီ", "ok", 1800);
    }
  }

  /* ------------------------------- logos ------------------------------ */
  async function onLogoChosen(event) {
    const file = event.target.files && event.target.files[0];
    event.target.value = "";
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    try {
      MSE.toast("Logo တင်နေသည်...", "ok", 1600);
      const data = await MSE.request("/api/logo", { method: "POST", body: form });
      let naturalWidth = data.width || 1;
      let naturalHeight = data.height || 1;
      try {
        const image = new Image();
        image.src = data.url;
        await image.decode();
        naturalWidth = image.naturalWidth || naturalWidth;
        naturalHeight = image.naturalHeight || naturalHeight;
      } catch (error) {
        /* use server dimensions */
      }
      Project.addLogo({
        src: data.src,
        url: data.url,
        name: data.name,
        w: 0.2,
        naturalWidth,
        naturalHeight,
      });
      MSE.toast("Logo ထည့်ပြီးပါပြီ — ဆွဲ/ချဲ့/လှည့်နိုင်ပါသည်", "ok");
    } catch (error) {
      MSE.toast(error.message || "Logo တင်၍ မရပါ", "bad", 6000);
    }
  }

  /* ---------------------------- transcription ------------------------- */
  async function runTranscribe() {
    const doc = Project.doc();
    if (!doc) return;
    const capabilities = Project.capabilities() || state.capabilities || {};
    const whisperOk = capabilities.whisper && capabilities.whisper.available;
    const demo = Boolean(capabilities.demo) && !whisperOk;
    if (!whisperOk && !demo) {
      MSE.toast(
        (capabilities.whisper && capabilities.whisper.reason) ||
          "Whisper အသင့် မဖြစ်ပါ။ backend/requirements.txt ကို ထည့်ပါ။",
        "bad",
        9000
      );
      return;
    }
    if (
      doc.segments.length &&
      !window.confirm("လက်ရှိ စာတန်းများ အားလုံး အစားထိုး ခံရပါမည်။ ဆက်လုပ်မလား?")
    ) {
      return;
    }
    const modal = $("#transcribe-modal");
    const bar = $("#transcribe-progress");
    const pct = $("#transcribe-pct");
    const stage = $("#transcribe-stage");
    $("#transcribe-model").textContent = demo
      ? "သရုပ်ပြ စာတန်း"
      : `${capabilities.whisper.model} · ${capabilities.whisper.backend}`;
    bar.style.width = "2%";
    pct.textContent = "0%";
    stage.textContent = "စတင်နေသည်...";
    MSE.openModal("transcribe-modal");
    try {
      const started = await MSE.json("/api/transcribe", {
        method: "POST",
        body: { video_id: doc.video.id, demo },
      });
      state.transcribeJobId = started.job_id;
      const result = await MSE.pollJob(started.job_id, {
        onProgress: (job) => {
          const progress = Math.round((job.progress || 0) * 100);
          bar.style.width = `${Math.max(2, progress)}%`;
          pct.textContent = `${progress}%`;
          if (job.stage) stage.textContent = job.stage;
        },
      });
      MSE.closeModal("transcribe-modal");
      if (!result.segments || !result.segments.length) {
        MSE.toast("စာတန်း မတွေ့ပါ — ဗီဒီယို အသံကို စစ်ပါ", "warn", 6000);
        return;
      }
      Project.replaceSegments(result.segments);
      await Project.save({ silent: true });
      seek(Project.doc().segments[0].start);
      MSE.toast(
        demo
          ? "သရုပ်ပြ စာတန်း ဖန်တီးပြီးပါပြီ (စမ်းသပ်ရန် သာ — စစ်မှန်သော စာသား မဟုတ်ပါ)"
          : `စာတန်း ${result.segments.length} ခု ဖန်တီးပြီးပါပြီ`,
        demo ? "warn" : "ok",
        6500
      );
    } catch (error) {
      MSE.closeModal("transcribe-modal");
      MSE.toast(error.message || "စာတန်း ဖန်တီး၍ မရပါ", "bad", 8000);
    } finally {
      state.transcribeJobId = null;
    }
  }

  /* ------------------------------- export ----------------------------- */
  async function runExport() {
    const doc = Project.doc();
    if (!doc) return;
    if (!doc.segments.length) {
      MSE.toast("စာတန်းထိုး တစ်ခုမှ မရှိသေးပါ", "warn", 3000);
      return;
    }
    const modal = $("#export-modal");
    const bar = $("#export-progress");
    const pct = $("#export-pct");
    const stage = $("#export-stage");
    const resultBox = $("#export-result");
    const downloadLink = $("#export-download");
    bar.style.width = "2%";
    pct.textContent = "0%";
    stage.textContent = "ပြင်ဆင်နေသည်...";
    resultBox.classList.add("hidden");
    resultBox.className = "export-preview hidden";
    downloadLink.classList.add("hidden");
    downloadLink.removeAttribute("href");
    MSE.openModal("export-modal");
    try {
      await Project.save({ silent: true }).catch(() => {});
      const started = await MSE.json("/api/export", {
        method: "POST",
        body: { project: Project.toPayload() },
      });
      state.exportJobId = started.job_id;
      const result = await MSE.pollJob(started.job_id, {
        onProgress: (job) => {
          const progress = Math.round((job.progress || 0) * 100);
          bar.style.width = `${Math.max(2, progress)}%`;
          pct.textContent = `${progress}%`;
          if (job.stage) stage.textContent = job.stage;
        },
      });
      bar.style.width = "100%";
      pct.textContent = "100%";
      stage.textContent = "ပြီးစီးပါပြီ";
      downloadLink.href = result.download_url;
      downloadLink.setAttribute("download", result.filename || "export.mp4");
      downloadLink.classList.remove("hidden");
      resultBox.className = "export-preview";
      resultBox.innerHTML = `✅ ဗီဒီယို အဆင်သင့် ဖြစ်ပါပြီ — ${MSE.escapeHtml(
        MSE.formatBytes(result.size)
      )} · ${MSE.formatTime(result.duration)} · ${result.segments} စာတန်း`;
      MSE.toast("ဗီဒီယို သိမ်းဆည်းပြီးပါပြီ", "ok", 5000);
    } catch (error) {
      stage.textContent = "အမှား ဖြစ်ပွားသည်";
      resultBox.className = "export-preview bad";
      resultBox.classList.remove("hidden");
      resultBox.textContent = error.message || "ဗီဒီယို သိမ်း၍ မရပါ";
      MSE.toast(error.message || "ဗီဒီယို သိမ်း၍ မရပါ", "bad", 8000);
    } finally {
      state.exportJobId = null;
    }
  }

  async function downloadSrt() {
    const doc = Project.doc();
    if (!doc || !doc.segments.length) {
      MSE.toast("SRT ထုတ်ရန် စာတန်း မရှိသေးပါ", "warn", 2600);
      return;
    }
    try {
      const response = await MSE.request("/api/srt", {
        method: "POST",
        body: { project: Project.toPayload() },
        raw: true,
      });
      const filename = await MSE.downloadResponse(response, `${doc.name || "subtitles"}.srt`);
      MSE.toast(`SRT ရယူပြီးပါပြီ — ${filename}`, "ok");
    } catch (error) {
      MSE.toast(error.message || "SRT ရယူ၍ မရပါ", "bad", 6000);
    }
  }

  /* ------------------------------ indicators -------------------------- */
  function updateSaveIndicator() {
    const el = $("#save-indicator");
    if (!el) return;
    if (Project.state.saving) {
      el.textContent = "သိမ်းဆည်းနေသည်...";
      el.className = "save-indicator saving";
    } else if (Project.isDirty()) {
      el.textContent = "မသိမ်းရသေးပါ";
      el.className = "save-indicator";
    } else if (Project.state.savedAt) {
      const time = new Date(Project.state.savedAt).toLocaleTimeString("en-GB", {
        hour: "2-digit",
        minute: "2-digit",
      });
      el.textContent = `သိမ်းပြီး ${time}`;
      el.className = "save-indicator saved";
    } else {
      el.textContent = "သိမ်းထားပြီး";
      el.className = "save-indicator saved";
    }
  }

  function updateUndoButtons() {
    const undo = $("#btn-undo");
    const redo = $("#btn-redo");
    if (undo) undo.disabled = !Project.canUndo();
    if (redo) redo.disabled = !Project.canRedo();
  }

  /* ------------------------------ keyboard ---------------------------- */
  function isTyping(event) {
    const tag = (event.target && event.target.tagName) || "";
    return (
      tag === "INPUT" ||
      tag === "TEXTAREA" ||
      tag === "SELECT" ||
      (event.target && event.target.isContentEditable)
    );
  }

  function bindKeyboard() {
    document.addEventListener("keydown", (event) => {
      if (!state.started) return;
      const meta = event.ctrlKey || event.metaKey;
      if (meta && event.key.toLowerCase() === "s") {
        event.preventDefault();
        Project.save().catch(() => {});
        return;
      }
      if (meta && event.key.toLowerCase() === "z") {
        if (isTyping(event)) return;
        event.preventDefault();
        Project.undo();
        return;
      }
      if (meta && (event.key.toLowerCase() === "y" || (event.shiftKey && event.key.toLowerCase() === "z"))) {
        if (isTyping(event)) return;
        event.preventDefault();
        Project.redo();
        return;
      }
      if (meta && event.key.toLowerCase() === "b") {
        event.preventDefault();
        splitAtPlayhead();
        return;
      }
      if (meta && event.key.toLowerCase() === "m") {
        event.preventDefault();
        mergeSelected();
        return;
      }
      if (isTyping(event)) return;
      if (event.key === " ") {
        event.preventDefault();
        togglePlay();
        return;
      }
      if (event.key === "ArrowRight" || event.key === "ArrowLeft") {
        const step = event.shiftKey ? 5 : 1;
        seek(Project.playhead() + (event.key === "ArrowRight" ? step : -step));
        return;
      }
      if (event.key === "Home") {
        seek(0);
        return;
      }
      if (event.key === "End") {
        seek(Project.duration());
        return;
      }
      if (event.key === "Delete" || event.key === "Backspace") {
        event.preventDefault();
        deleteSelected();
        return;
      }
      if (event.key === "Escape") {
        Project.select({ type: "none", id: null });
        return;
      }
      if (["ArrowUp", "ArrowDown"].includes(event.key)) {
        nudgeSelected(event.key === "ArrowUp" ? -1 : 1, "y", event.shiftKey);
        return;
      }
    });
  }

  function nudgeSelected(deltaPercent, axis, big) {
    const selection = Project.selection();
    const amount = (deltaPercent * (big ? 5 : 1)) / 100;
    if (selection.type === "subtitle") {
      const segment = Project.selectedSegment();
      if (!segment) return;
      Project.setSegmentStyle(
        selection.id,
        { [axis]: Math.min(1, Math.max(0, segment.style[axis] + amount)) },
        { history: false }
      );
    } else if (selection.type === "logo") {
      const logo = Project.selectedLogo();
      if (!logo) return;
      Project.updateLogo(selection.id, { [axis]: logo[axis] + amount }, { history: false });
    } else if (selection.type === "blur") {
      const box = Project.selectedBlur();
      if (!box) return;
      Project.updateBlur(selection.id, { [axis]: box[axis] + amount }, { history: false });
    }
  }
})();
