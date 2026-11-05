/* ============================================================
   api.js — fetch helper, toast, theme, auth, job polling
   (Vanilla JS, no build step. All UI strings are Myanmar.)
   ============================================================ */
const MSE = (() => {
  const API = "";

  /* ------------------------------- errors ----------------------------- */
  function extractError(payload, status) {
    const detail = payload && payload.detail;
    if (typeof detail === "string") return detail;
    if (detail && typeof detail === "object") {
      if (detail.message) return detail.error ? `${detail.message}` : detail.message;
      if (detail.detail) return String(detail.detail);
    }
    if (payload && payload.message) return payload.message;
    if (status === 401) return "သီးသန့် စကားဝှက် လိုအပ်ပါသည်";
    return `အမှား ဖြစ်ပွားသည် (${status})`;
  }

  class ApiError extends Error {
    constructor(message, status, payload) {
      super(message);
      this.status = status;
      this.payload = payload;
    }
  }

  async function request(path, options = {}) {
    const { method = "GET", body, headers = {}, raw = false, signal } = options;
    const init = { method, headers: { ...headers }, signal, credentials: "same-origin" };
    if (body !== undefined) {
      if (
        body instanceof FormData ||
        body instanceof Blob ||
        body instanceof ArrayBuffer ||
        ArrayBuffer.isView(body)
      ) {
        init.body = body;
      } else {
        init.headers["Content-Type"] = "application/json";
        init.body = JSON.stringify(body);
      }
    }
    let response;
    try {
      response = await fetch(API + path, init);
    } catch (error) {
      if (error.name === "AbortError") throw error;
      throw new ApiError("ကွန်ရက် ချိတ်ဆက်မှု မအောင်မြင်ပါ", 0, null);
    }
    if (response.status === 401) {
      const payload = await response.clone().json().catch(() => null);
      Auth.showLogin();
      throw new ApiError(extractError(payload, 401), 401, payload);
    }
    if (!response.ok) {
      const payload = await response.json().catch(() => null);
      throw new ApiError(extractError(payload, response.status), response.status, payload);
    }
    if (raw) return response;
    if (response.status === 204) return null;
    const contentType = response.headers.get("content-type") || "";
    if (contentType.includes("application/json")) return response.json();
    return response;
  }

  const json = (path, options) => request(path, options);

  /* ------------------------------- toast ------------------------------ */
  function toastWrap() {
    let wrap = document.querySelector(".toast-wrap");
    if (!wrap) {
      wrap = document.createElement("div");
      wrap.className = "toast-wrap";
      document.body.appendChild(wrap);
    }
    return wrap;
  }

  function toast(message, kind = "ok", timeout = 3600) {
    const wrap = toastWrap();
    const el = document.createElement("div");
    el.className = `toast ${kind}`;
    const icon = kind === "bad" ? "⚠" : kind === "warn" ? "!" : "✓";
    el.innerHTML = `<strong style="flex:0 0 auto">${icon}</strong><div>${escapeHtml(
      message
    )}</div>`;
    wrap.appendChild(el);
    setTimeout(() => {
      el.classList.add("hide");
      setTimeout(() => el.remove(), 260);
    }, timeout);
    return el;
  }

  /* ------------------------------- theme ------------------------------ */
  const THEME_KEY = "mse-theme";
  function applyTheme(theme) {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem(THEME_KEY, theme);
    document.querySelectorAll("[data-theme-toggle]").forEach((btn) => {
      btn.textContent = theme === "dark" ? "☀" : "☾";
      btn.title = theme === "dark" ? "အလင်း မုဒ်" : "အမှောင် မုဒ်";
    });
  }
  function currentTheme() {
    return localStorage.getItem(THEME_KEY) || "dark";
  }
  function toggleTheme() {
    applyTheme(currentTheme() === "dark" ? "light" : "dark");
  }
  function initTheme() {
    applyTheme(currentTheme());
    document.querySelectorAll("[data-theme-toggle]").forEach((btn) => {
      btn.addEventListener("click", toggleTheme);
    });
  }

  /* -------------------------------- auth ------------------------------ */
  const Auth = {
    status: { required: false, authed: true },
    async check() {
      try {
        this.status = await json("/api/auth/status");
      } catch (error) {
        this.status = { required: false, authed: true };
      }
      if (this.status.required && !this.status.authed) this.showLogin();
      return this.status;
    },
    showLogin() {
      const modal = document.getElementById("login-modal");
      if (!modal) return;
      modal.classList.add("open");
      const input = modal.querySelector("input");
      if (input) setTimeout(() => input.focus(), 80);
    },
    async login(password) {
      await json("/api/auth/login", { method: "POST", body: { password } });
      this.status = { required: true, authed: true };
      const modal = document.getElementById("login-modal");
      if (modal) modal.classList.remove("open");
      toast("ဝင်ရောက်ပြီးပါပြီ", "ok");
    },
  };

  function bindLoginForm() {
    const modal = document.getElementById("login-modal");
    if (!modal) return;
    const form = modal.querySelector("form");
    const error = modal.querySelector(".login-error");
    form &&
      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const password = form.querySelector("input").value;
        try {
          await Auth.login(password);
          error.textContent = "";
          location.reload();
        } catch (err) {
          error.textContent = err.message || "စကားဝှက် မှားနေပါသည်";
          error.classList.remove("hidden");
        }
      });
  }

  /* ------------------------------ formatting -------------------------- */
  function formatTime(seconds, withMs = false) {
    const value = Number(seconds) || 0;
    const hours = Math.floor(value / 3600);
    const minutes = Math.floor((value % 3600) / 60);
    const secs = Math.floor(value % 60);
    const ms = Math.floor((value % 1) * 1000);
    const base = hours
      ? `${hours}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`
      : `${minutes}:${String(secs).padStart(2, "0")}`;
    return withMs ? `${base}.${String(ms).padStart(3, "0")}` : base;
  }
  function formatClock(seconds) {
    const value = Math.max(0, Number(seconds) || 0);
    const minutes = Math.floor(value / 60);
    const secs = Math.floor(value % 60);
    return `${minutes}:${String(secs).padStart(2, "0")}`;
  }
  function parseClock(text) {
    const parts = String(text || "")
      .trim()
      .split(":")
      .map((part) => Number(part));
    if (parts.some((part) => Number.isNaN(part))) return null;
    if (parts.length === 3) return parts[0] * 3600 + parts[1] * 60 + parts[2];
    if (parts.length === 2) return parts[0] * 60 + parts[1];
    if (parts.length === 1) return parts[0];
    return null;
  }
  function formatBytes(bytes) {
    const value = Number(bytes) || 0;
    if (value < 1024) return `${value} B`;
    if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
    if (value < 1024 * 1024 * 1024) return `${(value / 1024 / 1024).toFixed(1)} MB`;
    return `${(value / 1024 / 1024 / 1024).toFixed(2)} GB`;
  }
  function escapeHtml(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }
  function uid(prefix = "") {
    const value = Math.random().toString(16).slice(2, 10) + Date.now().toString(16).slice(-6);
    return prefix + value;
  }
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  /* ------------------------------ job polling ------------------------- */
  async function pollJob(jobId, { onProgress, timeout = 1000 * 60 * 60 * 2 } = {}) {
    const started = Date.now();
    for (;;) {
      const job = await json(`/api/jobs/${jobId}`);
      if (onProgress) onProgress(job);
      if (job.status === "done") return job.result;
      if (job.status === "error") throw new ApiError(job.error || "အလုပ် မအောင်မြင်ပါ", 500, job);
      if (job.status === "cancelled") throw new ApiError("ပိတ်လိုက်ပါသည်", 499, job);
      if (Date.now() - started > timeout) throw new ApiError("အချိန် ကုန်သွားပါသည်", 504, job);
      await sleep(350);
    }
  }

  /* ------------------------------ downloads --------------------------- */
  function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 4000);
  }

  async function downloadResponse(response, fallbackName) {
    const blob = await response.blob();
    let filename = fallbackName;
    const disposition = response.headers.get("content-disposition") || "";
    const utf8 = /filename\*=UTF-8''([^;]+)/i.exec(disposition);
    const plain = /filename="?([^";]+)"?/i.exec(disposition);
    if (utf8) filename = decodeURIComponent(utf8[1]);
    else if (plain) filename = plain[1];
    downloadBlob(blob, filename);
    return filename;
  }

  /* ------------------------------- modals ----------------------------- */
  function openModal(id) {
    const modal = document.getElementById(id);
    if (modal) modal.classList.add("open");
  }
  function closeModal(id) {
    const modal = document.getElementById(id);
    if (modal) modal.classList.remove("open");
  }
  function bindModals() {
    document.querySelectorAll("[data-close-modal]").forEach((btn) => {
      btn.addEventListener("click", () => closeModal(btn.dataset.closeModal));
    });
    document.querySelectorAll(".modal").forEach((modal) => {
      modal.addEventListener("click", (event) => {
        if (event.target === modal && modal.dataset.static !== "true") {
          modal.classList.remove("open");
        }
      });
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        document.querySelectorAll(".modal.open").forEach((modal) => {
          if (modal.dataset.static !== "true") modal.classList.remove("open");
        });
      }
    });
  }

  function storageGet(key, fallback) {
    try {
      const raw = localStorage.getItem(key);
      return raw === null ? fallback : JSON.parse(raw);
    } catch (error) {
      return fallback;
    }
  }
  function storageSet(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch (error) {
      /* quota — ignore */
    }
  }

  async function capabilities(force = false) {
    return json(`/api/capabilities${force ? "?force=true" : ""}`);
  }

  document.addEventListener("DOMContentLoaded", () => {
    initTheme();
    bindModals();
    bindLoginForm();
  });

  return {
    ApiError,
    request,
    json,
    toast,
    formatTime,
    formatClock,
    parseClock,
    formatBytes,
    escapeHtml,
    uid,
    sleep,
    pollJob,
    downloadBlob,
    downloadResponse,
    openModal,
    closeModal,
    Auth,
    capabilities,
    currentTheme,
    toggleTheme,
    applyTheme,
    initTheme,
    storageGet,
    storageSet,
  };
})();
