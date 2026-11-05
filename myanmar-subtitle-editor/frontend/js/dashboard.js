/* ============================================================
   dashboard.js — ဗီဒီယို တင်၊ စာတန်းထိုး ဖန်တီး၊ ပရောဂျက် စီမံ
   ============================================================ */
(() => {
  const $ = (selector) => document.querySelector(selector);
  const $$ = (selector) => Array.from(document.querySelectorAll(selector));
  const state = {
    capabilities: null,
    busy: false,
    projects: [],
  };

  document.addEventListener("DOMContentLoaded", boot);

  async function boot() {
    bindUi();
    await MSE.Auth.check();
    try {
      state.capabilities = await MSE.capabilities();
    } catch (error) {
      MSE.toast(error.message || "ဆာဗာ ချိတ်ဆက်မှု မအောင်မြင်ပါ", "bad", 6000);
    }
    renderCapabilities();
    await loadProjects();
  }

  /* ------------------------------- chrome ----------------------------- */
  function bindUi() {
    const fileInput = $("#video-input");
    const openPicker = () => {
      if (state.busy) return;
      fileInput.click();
    };
    $("#btn-upload")?.addEventListener("click", openPicker);
    $("#btn-upload-2")?.addEventListener("click", openPicker);
    fileInput?.addEventListener("change", (event) => {
      const file = event.target.files && event.target.files[0];
      event.target.value = "";
      if (file) startWithFile(file);
    });

    const dropzone = $("#dropzone");
    if (dropzone) {
      dropzone.addEventListener("click", openPicker);
      dropzone.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") openPicker();
      });
      ["dragenter", "dragover"].forEach((name) =>
        dropzone.addEventListener(name, (event) => {
          event.preventDefault();
          dropzone.classList.add("drag");
        })
      );
      ["dragleave", "dragend"].forEach((name) =>
        dropzone.addEventListener(name, () => dropzone.classList.remove("drag"))
      );
      dropzone.addEventListener("drop", (event) => {
        event.preventDefault();
        dropzone.classList.remove("drag");
        const file = event.dataTransfer.files && event.dataTransfer.files[0];
        if (file) startWithFile(file);
      });
    }

    $("#btn-demo-video")?.addEventListener("click", startDemo);
    $("#btn-demo-video-2")?.addEventListener("click", startDemo);
    $("#btn-refresh")?.addEventListener("click", () => loadProjects(true));
  }

  function formatDate(value) {
    if (!value) return "";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "";
    return date.toLocaleString("en-GB", {
      day: "2-digit",
      month: "short",
      hour: "2-digit",
      minute: "2-digit",
    });
  }

  /* --------------------------- capabilities --------------------------- */
  function renderCapabilities() {
    const caps = state.capabilities;
    const chip = $("#ffmpeg-chip");
    if (!caps) {
      if (chip) {
        chip.className = "chip bad";
        chip.innerHTML = '<span class="dot"></span> ဆာဗာ မချိတ်ဆက်နိုင်ပါ';
      }
      return;
    }
    const ffmpegOk = caps.ffmpeg && caps.ffmpeg.ok;
    if (chip) {
      chip.className = `chip ${ffmpegOk ? "ok" : "warn"}`;
      chip.innerHTML = `<span class="dot"></span> FFmpeg ${ffmpegOk ? "အသင့်" : "မပြည့်စုံ"}`;
      chip.title = ffmpegOk
        ? caps.ffmpeg.path || ""
        : (caps.ffmpeg.problems || []).join("\n");
    }
    const label = $("#max-upload-label");
    if (label) label.textContent = `${caps.config.max_upload_mb} MB`;

    const list = $("#status-list");
    if (list) {
      const whisper = caps.whisper || {};
      const items = [
        {
          ok: ffmpegOk,
          icon: "🎞",
          title: "FFmpeg (libass + harfbuzz)",
          detail: ffmpegOk
            ? "မြန်မာစာတန်း burn-in အသင့်"
            : (caps.ffmpeg.problems || []).join(" · ") || "မတွေ့ပါ",
        },
        {
          ok: whisper.available,
          icon: "🎙",
          title: `Whisper — ${whisper.backend || "-"} · ${whisper.model || "-"}`,
          detail: whisper.available
            ? "စကားပြော → မြန်မာစာတန်း အသင့်"
            : whisper.reason || "မရနိုင်ပါ",
        },
        {
          ok: caps.fonts && caps.fonts.regular,
          icon: "🔤",
          title: `ဖောင့် — ${caps.fonts.family}`,
          detail: "Canvas နှင့် FFmpeg နှစ်ခုလုံး တူညီသော ဖောင့် အသုံးပြုသည်",
        },
        {
          ok: !caps.auth.required || true,
          icon: "🔒",
          title: caps.auth.required ? "စကားဝှက် ကာကွယ်ထားသည်" : "စကားဝှက် မကာကွယ်ထားပါ",
          detail: caps.auth.required
            ? "သီးသန့် စာမျက်နှာ"
            : "EDITOR_PASSWORD သတ်မှတ်ရန် အကြံပြုပါသည်",
        },
      ];
      list.innerHTML = items
        .map(
          (item) => `
        <li>
          <span class="icon">${item.icon}</span>
          <span>
            <strong style="display:block;color:var(--text)">${MSE.escapeHtml(item.title)} ${
            item.ok ? "✓" : "⚠"
          }</strong>
            <span style="font-size:0.78rem">${MSE.escapeHtml(item.detail)}</span>
          </span>
        </li>`
        )
        .join("");
    }

    const config = $("#config-list");
    if (config) {
      const cfg = caps.config;
      const rows = [
        ["မော်ဒယ်", `${cfg.stt_backend} · ${cfg.model_size}`],
        ["စကားလုံး ကွာဟမှု (gap)", `${cfg.gap_threshold} စက္ကန့်`],
        ["စာလုံး အများဆုံး", `${cfg.max_chars} လုံး`],
        ["ကြာချိန် အများဆုံး", `${cfg.max_duration} စက္ကန့်`],
        ["နောက်ဆုံး စာတန်း ဆက်ပေါ်ချိန်", `${cfg.max_hold_seconds} စက္ကန့်`],
        ["တင်နိုင်သော အများဆုံး", `${cfg.max_upload_mb} MB`],
      ];
      config.innerHTML = rows
        .map(
          ([key, value]) =>
            `<div><span>${MSE.escapeHtml(key)}</span><span>${MSE.escapeHtml(String(value))}</span></div>`
        )
        .join("");
    }

    const warnings = $("#warnings");
    if (warnings) {
      const blocks = [];
      if (!ffmpegOk) {
        blocks.push(
          `<div class="warn-box bad"><strong>FFmpeg အသင့် မဖြစ်ပါ</strong><br>${(caps.ffmpeg.problems || [])
            .map((item) => MSE.escapeHtml(item))
            .join("<br>")}</div>`
        );
      }
      if (!caps.whisper.available) {
        blocks.push(
          `<div class="warn-box"><strong>Whisper မရနိုင်ပါ</strong><br>${MSE.escapeHtml(
            caps.whisper.reason || ""
          )}${
            caps.demo
              ? "<br>သရုပ်ပြ စာတန်းဖြင့် UI ကို စမ်းသပ်နိုင်ပါသည်။"
              : ""
          }</div>`
        );
      }
      if (!caps.auth.required) {
        blocks.push(
          `<div class="warn-box"><strong>သီးသန့် မဟုတ်သေးပါ</strong><br>ဤစာမျက်နှာကို မည်သူမဆို ဝင်နိုင်ပါသည်။ EDITOR_PASSWORD သတ်မှတ်ပါ။</div>`
        );
      }
      warnings.innerHTML = blocks.join("");
    }
  }

  /* ---------------------------- projects ------------------------------ */
  async function loadProjects(showToast = false) {
    const grid = $("#project-grid");
    try {
      const data = await MSE.json("/api/project/list");
      state.projects = data.projects || [];
      renderProjects();
      if (showToast) MSE.toast("ပရောဂျက် စာရင်း ပြန်စစ်ပြီးပါပြီ", "ok", 1800);
    } catch (error) {
      if (grid) grid.innerHTML = `<div class="empty-state">${MSE.escapeHtml(error.message)}</div>`;
    }
  }

  function renderProjects() {
    const grid = $("#project-grid");
    if (!grid) return;
    if (!state.projects.length) {
      grid.innerHTML = `<div class="empty-state">ပရောဂျက် မရှိသေးပါ။ ဗီဒီယို တင်ပြီး စတင်ပါ။</div>`;
      return;
    }
    grid.innerHTML = state.projects
      .map((project) => {
        const thumb = project.video_id
          ? `<img src="/api/video/${project.video_id}/thumbnail" alt="" loading="lazy"
               onerror="this.style.display='none'">`
          : "🎬";
        return `
        <article class="project-card" data-project="${project.id}">
          <div class="project-thumb">
            ${thumb}
            <span class="duration-badge">${MSE.formatTime(project.duration || 0)}</span>
          </div>
          <div class="project-body">
            <div class="project-title">${MSE.escapeHtml(project.name)}</div>
            <div class="project-meta">
              <span>စာတန်း ${project.segment_count}</span>
              <span>Logo ${project.logo_count}</span>
              <span>Blur ${project.blur_count}</span>
            </div>
            <div class="project-meta"><span>${MSE.escapeHtml(formatDate(project.updated_at))}</span></div>
            <div class="project-actions">
              <a class="btn btn-sm btn-primary" href="/editor/${project.id}">ဖွင့်ရန်</a>
              <a class="btn btn-sm" href="/api/srt/${project.id}" title="SRT ရယူရန်">SRT</a>
              <button class="btn btn-sm btn-danger" data-delete="${project.id}" title="ဖျက်ရန်">🗑</button>
            </div>
          </div>
        </article>`;
      })
      .join("");
    grid.querySelectorAll("[data-delete]").forEach((button) => {
      button.addEventListener("click", () => deleteProject(button.dataset.delete));
    });
  }

  async function deleteProject(id) {
    if (!window.confirm("ဤပရောဂျက်ကို ဖျက်မည်လား? ပြန်ယူ၍ မရပါ။")) return;
    try {
      await MSE.json(`/api/project/${id}`, { method: "DELETE" });
      MSE.toast("ပရောဂျက် ဖျက်ပြီးပါပြီ", "ok");
      await loadProjects();
    } catch (error) {
      MSE.toast(error.message || "ဖျက်၍ မရပါ", "bad", 5000);
    }
  }

  /* ----------------------------- progress ----------------------------- */
  function showProgress(title, stage) {
    $("#upload-title").textContent = title;
    $("#upload-stage").textContent = stage;
    $("#upload-progress").style.width = "1%";
    $("#upload-pct").textContent = "0%";
    $("#upload-size").textContent = "";
    const errorBox = $("#upload-error");
    errorBox.classList.add("hidden");
    errorBox.className = "export-preview hidden";
    errorBox.textContent = "";
    MSE.openModal("upload-modal");
  }

  function setProgress(percent, stage, extra) {
    const value = Math.max(1, Math.min(100, Math.round(percent)));
    $("#upload-progress").style.width = `${value}%`;
    $("#upload-pct").textContent = `${value}%`;
    if (stage) $("#upload-stage").textContent = stage;
    if (extra != null) $("#upload-size").textContent = extra;
  }

  function showProgressError(message) {
    const errorBox = $("#upload-error");
    errorBox.className = "export-preview bad";
    errorBox.classList.remove("hidden");
    errorBox.textContent = message;
  }

  /* ------------------------------ upload ------------------------------ */
  const ALLOWED = [".mp4", ".mkv", ".mov", ".m4v", ".webm"];

  async function startWithFile(file) {
    if (state.busy) return;
    const extension = `.${(file.name.split(".").pop() || "").toLowerCase()}`;
    if (!ALLOWED.includes(extension)) {
      MSE.toast(`'${extension}' ဖိုင် အမျိုးအစားကို မပံ့ပိုးပါ (${ALLOWED.join(", ")})`, "bad", 6000);
      return;
    }
    const maxBytes = (state.capabilities?.config?.max_upload_mb || 500) * 1024 * 1024;
    if (file.size > maxBytes) {
      MSE.toast(
        `ဖိုင် ကြီးလွန်းပါသည် (${MSE.formatBytes(file.size)}) — အများဆုံး ${MSE.formatBytes(
          maxBytes
        )}`,
        "bad",
        7000
      );
      return;
    }
    state.busy = true;
    showProgress("ဗီဒီယို တင်နေသည်...", "ဖိုင် ပြင်ဆင်နေသည်...");
    try {
      const video = await uploadVideo(file, (sent, total) => {
        setProgress((sent / total) * 75, "ဗီဒီယို တင်နေသည်...", `${MSE.formatBytes(sent)} / ${MSE.formatBytes(total)}`);
      });
      setProgress(78, "စကားပြော မှတ်မိနေသည်...", "ဗီဒီယို တင်ပြီးပါပြီ");
      await transcribeAndOpen(video, { name: baseName(file.name) });
    } catch (error) {
      state.busy = false;
      showProgressError(error.message || "တင်၍ မရပါ");
      MSE.toast(error.message || "ဗီဒီယို တင်၍ မရပါ", "bad", 8000);
    }
  }

  function baseName(filename) {
    return (filename || "ပရောဂျက်").replace(/\.[^.]+$/, "").slice(0, 100) || "ပရောဂျက်";
  }

  async function uploadVideo(file, onProgress) {
    const start = await MSE.json("/api/upload/start", {
      method: "POST",
      body: { filename: file.name, size: file.size },
    });
    const chunkSize = start.chunk_size || 8 * 1024 * 1024;
    let offset = 0;
    while (offset < file.size) {
      const chunk = file.slice(offset, offset + chunkSize);
      const response = await MSE.json(
        `/api/upload/${start.upload_id}?offset=${offset}`,
        {
          method: "PUT",
          body: chunk,
          headers: { "Content-Type": "application/octet-stream" },
        }
      );
      if (response && response.resync) {
        offset = response.received;
        continue;
      }
      offset = (response && response.received) || offset + chunk.size;
      if (onProgress) onProgress(offset, file.size);
    }
    setProgress(76, "ဖိုင် စစ်ဆေးနေသည်...", "");
    return MSE.json(`/api/upload/${start.upload_id}/complete`, { method: "POST" });
  }

  /* ---------------------------- transcription ------------------------- */
  async function transcribeAndOpen(video, { name, demo = null } = {}) {
    const caps = state.capabilities || {};
    const whisperOk = Boolean(caps.whisper && caps.whisper.available);
    const useDemo = demo == null ? Boolean(caps.demo) && !whisperOk : Boolean(demo);
    if (!whisperOk && !useDemo) {
      // No ASR available: still open the editor so captions can be added by hand.
      await createProject(video, [], { name, demo: false });
      return;
    }
    try {
      const started = await MSE.json("/api/transcribe", {
        method: "POST",
        body: { video_id: video.id, demo: useDemo },
      });
      const result = await MSE.pollJob(started.job_id, {
        onProgress: (job) => {
          setProgress(78 + (job.progress || 0) * 20, job.stage || "စာတန်းထိုး ဖန်တီးနေသည်...", "");
        },
      });
      await createProject(video, result.segments || [], { name, demo: useDemo });
    } catch (error) {
      showProgressError(error.message || "စာတန်း ဖန်တီး၍ မရပါ");
      MSE.toast(
        `${error.message || "စာတန်း ဖန်တီး၍ မရပါ"} — စာတန်းမပါဘဲ တည်းဖြတ်ခန်း ဖွင့်ပေးနေသည်`,
        "warn",
        8000
      );
      await createProject(video, [], { name, demo: false });
    }
  }

  async function createProject(video, segments, { name, demo }) {
    setProgress(99, "ပရောဂျက် သိမ်းနေသည်...", "");
    const payload = {
      name: name || baseName(video.filename),
      video: {
        id: video.id,
        filename: video.filename,
        width: video.width,
        height: video.height,
        duration: video.duration,
        has_audio: video.has_audio,
      },
      segments: segments.map((segment) => ({
        start: segment.start,
        end: segment.end,
        text: segment.text,
      })),
      logos: [],
      blurs: [],
    };
    const saved = await MSE.json("/api/project/save", { method: "POST", body: payload });
    setProgress(100, "ပြီးစီးပါပြီ", "");
    const project = saved.project;
    if (!segments.length) {
      MSE.toast(
        "စာတန်း မပါဘဲ ပရောဂျက် ဖန်တီးပြီးပါပြီ — တည်းဖြတ်ခန်းတွင် ကိုယ်တိုင် ထည့်နိုင်ပါသည်",
        "warn",
        6500
      );
    } else if (demo) {
      MSE.toast(
        "သရုပ်ပြ စာတန်းဖြင့် ပရောဂျက် ဖန်တီးပြီးပါပြီ — ဤစာသားများသည် စမ်းသပ်ရန် သာ ဖြစ်သည်",
        "warn",
        7000
      );
    } else {
      MSE.toast(`စာတန်း ${segments.length} ခု ဖန်တီးပြီးပါပြီ — တည်းဖြတ်ခန်း ဖွင့်နေသည်...`, "ok", 4000);
    }
    state.busy = false;
    window.location.href = `/editor/${project.id}`;
  }

  /* ------------------------------- demo ------------------------------- */
  async function startDemo() {
    if (state.busy) return;
    state.busy = true;
    showProgress("သရုပ်ပြ ဗီဒီယို ပြင်ဆင်နေသည်...", "ဗီဒီယို ဖန်တီးနေသည် (FFmpeg)");
    try {
      setProgress(20, "သရုပ်ပြ ဗီဒီယို ဖန်တီးနေသည်...", "");
      const video = await MSE.json("/api/demo/video", { method: "POST" });
      setProgress(70, "သရုပ်ပြ စာတန်း ဖန်တီးနေသည်...", "");
      await transcribeAndOpen(video, { name: "သရုပ်ပြ ပရောဂျက်", demo: true });
    } catch (error) {
      state.busy = false;
      showProgressError(error.message || "သရုပ်ပြ ဗီဒီယို ဖန်တီး၍ မရပါ");
      MSE.toast(error.message || "သရုပ်ပြ ဗီဒီယို ဖန်တီး၍ မရပါ", "bad", 8000);
    }
  }
})();
