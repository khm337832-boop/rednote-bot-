/* ============================================================
   layers.js — Konva canvas: Video + Blur + Logo + စာတန်းထိုး
   • Subtitle: Fabric/Konva Textbox ကဲ့သို့ browser text layout ကို သုံးသည်
     (မြန်မာစာကို ကိုယ်တိုင် စာလုံးဖြတ် မရေးပါ)
   • Blur: video frame ကို canvas filter ဖြင့် ချက်ချင်း preview
   • Snap guide, drag, resize, rotate, boundary check
   ============================================================ */
const CanvasView = (() => {
  const SNAP = 7;
  let container = null;
  let videoEl = null;
  let stage = null;
  let videoLayer = null;
  let blurLayer = null;
  let overlayLayer = null;
  let uiLayer = null;
  let videoNode = null;
  let transformer = null;
  let guides = { v: null, h: null };
  let stageW = 0;
  let stageH = 0;
  let currentTime = 0;
  let fontsReady = false;
  let rafId = null;
  let measureCtx = null;
  let subtitleNode = null;
  let subtitleSegmentId = null;
  let subtitleSignature = "";
  const logoNodes = new Map();
  const logoImages = new Map();
  const blurNodes = new Map();
  let ready = false;
  let syncing = false;

  const videoAspect = () => {
    const doc = Project.doc();
    const width = (doc && doc.video.width) || videoEl?.videoWidth || 16;
    const height = (doc && doc.video.height) || videoEl?.videoHeight || 9;
    return width / height || 16 / 9;
  };

  function ensureMeasure() {
    if (!measureCtx) {
      const canvas = document.createElement("canvas");
      measureCtx = canvas.getContext("2d");
    }
    return measureCtx;
  }

  function measureText(text, fontSize, bold) {
    const ctx = ensureMeasure();
    ctx.font = `${bold ? "700" : "400"} ${fontSize}px "Noto Sans Myanmar", "Pyidaungsu", sans-serif`;
    return ctx.measureText(text).width;
  }

  async function waitForFonts() {
    if (fontsReady) return;
    try {
      if (document.fonts && document.fonts.load) {
        await Promise.all([
          document.fonts.load('400 40px "Noto Sans Myanmar"'),
          document.fonts.load('700 40px "Noto Sans Myanmar"'),
        ]);
        await document.fonts.ready;
      }
    } catch (error) {
      /* fall back to system font */
    }
    fontsReady = true;
  }

  function init(options) {
    container = options.container;
    videoEl = options.video;
    stageW = Math.max(160, Math.round(container.clientWidth));
    stageH = Math.round(stageW / videoAspect());
    stage = new Konva.Stage({ container, width: stageW, height: stageH });
    videoLayer = new Konva.Layer({ listening: false });
    blurLayer = new Konva.Layer();
    overlayLayer = new Konva.Layer();
    uiLayer = new Konva.Layer({ listening: false });
    stage.add(videoLayer, blurLayer, overlayLayer, uiLayer);

    videoNode = new Konva.Image({ image: videoEl, width: stageW, height: stageH });
    videoLayer.add(videoNode);

    guides.v = new Konva.Line({
      points: [0, 0, 0, stageH],
      stroke: "#f472b6",
      strokeWidth: 1,
      dash: [5, 5],
      visible: false,
    });
    guides.h = new Konva.Line({
      points: [0, 0, stageW, 0],
      stroke: "#f472b6",
      strokeWidth: 1,
      dash: [5, 5],
      visible: false,
    });
    uiLayer.add(guides.v, guides.h);

    transformer = new Konva.Transformer({
      rotateEnabled: true,
      rotateAnchorOffset: 26,
      borderStroke: "#a855f7",
      borderStrokeWidth: 1.5,
      anchorStroke: "#a855f7",
      anchorFill: "#ffffff",
      anchorSize: 9,
      anchorCornerRadius: 3,
      ignoreStroke: true,
      flipEnabled: false,
      boundBoxFunc: (oldBox, newBox) => {
        if (newBox.width < 24 || newBox.height < 18) return oldBox;
        if (
          newBox.x < -2 ||
          newBox.y < -2 ||
          newBox.x + newBox.width > stageW + 2 ||
          newBox.y + newBox.height > stageH + 2
        ) {
          return oldBox;
        }
        return newBox;
      },
    });
    overlayLayer.add(transformer);

    stage.on("click tap", (event) => {
      if (event.target === stage || event.target === videoNode) {
        Project.select({ type: "none", id: null });
      }
    });

    videoLayer.batchDraw();
    ready = true;
    sync();
    return stage;
  }

  function resize({ render = true } = {}) {
    if (!ready || !container) return;
    const bounds = container.parentElement.getBoundingClientRect();
    const availableW = Math.max(160, Math.floor(bounds.width - 20));
    const availableH = Math.max(120, Math.floor(bounds.height - 20));
    const aspect = videoAspect();
    let width = availableW;
    let height = Math.round(width / aspect);
    if (height > availableH) {
      height = availableH;
      width = Math.round(height * aspect);
    }
    container.style.width = `${width}px`;
    container.style.height = `${height}px`;
    if (width === stageW && height === stageH) return;
    stageW = width;
    stageH = height;
    stage.width(stageW);
    stage.height(stageH);
    videoNode.width(stageW);
    videoNode.height(stageH);
    if (render && !syncing) sync();
  }

  /* ------------------------------ gaps -------------------------------- */
  function setGuides(vertical, horizontal) {
    guides.v.visible(vertical != null);
    if (vertical != null) guides.v.points([vertical, 0, vertical, stageH]);
    guides.h.visible(horizontal != null);
    if (horizontal != null) guides.h.points([0, horizontal, stageW, horizontal]);
    uiLayer.batchDraw();
  }

  function snapValue(value, targets) {
    for (const target of targets) {
      if (Math.abs(value - target) <= SNAP) return { value: target, hit: true, target };
    }
    return { value, hit: false, target: null };
  }

  function snapNode(node, box) {
    const targetsX = [0, stageW / 3, stageW / 2, (stageW * 2) / 3, stageW];
    const targetsY = [0, stageH / 3, stageH / 2, (stageH * 2) / 3, stageH];
    const centerX = box.x + box.width / 2;
    const centerY = box.y + box.height / 2;
    const left = snapValue(centerX, targetsX);
    const top = snapValue(centerY, targetsY);
    let dx = 0;
    let dy = 0;
    if (left.hit) dx = left.target - centerX;
    if (top.hit) dy = top.target - centerY;
    if (dx || dy) node.position({ x: node.x() + dx, y: node.y() + dy });
    setGuides(left.hit ? left.target : null, top.hit ? top.target : null);
  }

  function clearGuides() {
    setGuides(null, null);
  }

  /* ---------------------------- selection ----------------------------- */
  function nodeFor(selection) {
    if (!selection) return null;
    if (selection.type === "subtitle") return subtitleNode && subtitleSegmentId === selection.id ? subtitleNode : null;
    if (selection.type === "logo") return logoNodes.get(selection.id) || null;
    if (selection.type === "blur") return blurNodes.get(selection.id)?.shape || null;
    return null;
  }

  function applySelection() {
    const selection = Project.selection();
    const node = nodeFor(selection);
    transformer.nodes(node ? [node] : []);
    if (selection.type === "blur") {
      transformer.rotateEnabled(false);
    } else {
      transformer.rotateEnabled(true);
    }
    overlayLayer.batchDraw();
    uiLayer.batchDraw();
  }

  /* ------------------------------ subtitle ---------------------------- */
  function ensureSubtitle() {
    if (subtitleNode) return subtitleNode;
    subtitleNode = new Konva.Text({
      x: stageW / 2,
      y: stageH * 0.86,
      text: "",
      fontFamily: '"Noto Sans Myanmar", "Pyidaungsu", sans-serif',
      fontSize: 32,
      fill: "#ffffff",
      stroke: "#000000",
      strokeWidth: 2,
      lineHeight: 1.35,
      align: "center",
      wrap: "none",
      listening: true,
      draggable: true,
      perfectDrawEnabled: false,
    });
    overlayLayer.add(subtitleNode);

    subtitleNode.on("dragstart", () => Project.checkpoint());
    subtitleNode.on("dragmove", () => {
      const box = subtitleNode.getClientRect({ skipTransform: false });
      snapNode(subtitleNode, box);
      subtitleNode.x(Math.max(0, Math.min(stageW, subtitleNode.x())));
      subtitleNode.y(Math.max(0, Math.min(stageH, subtitleNode.y())));
      overlayLayer.batchDraw();
    });
    subtitleNode.on("dragend", () => {
      clearGuides();
      const segment = Project.selectedSegment();
      if (segment) {
        Project.setSegmentStyle(
          segment.id,
          { x: subtitleNode.x() / stageW, y: subtitleNode.y() / stageH },
          { history: false }
        );
      }
      applySelection();
    });
    subtitleNode.on("transformstart", () => Project.checkpoint());
    subtitleNode.on("transformend", () => {
      const segment = Project.selectedSegment();
      if (!segment) return;
      const scaleX = subtitleNode.scaleX();
      const scaleY = subtitleNode.scaleY();
      const newWidth = (subtitleNode.width() * scaleX) / stageW;
      const newFont = (subtitleNode.fontSize() * scaleY) / stageH;
      subtitleNode.scale({ x: 1, y: 1 });
      Project.setSegmentStyle(
        segment.id,
        {
          width: Math.max(0.05, Math.min(1, newWidth)),
          fontSize: Math.max(0.015, Math.min(0.3, newFont)),
          x: subtitleNode.x() / stageW,
          y: subtitleNode.y() / stageH,
          rotation: subtitleNode.rotation() || 0,
        },
        { history: false }
      );
      sync();
    });
    subtitleNode.on("click tap", (event) => {
      event.cancelBubble = true;
      if (subtitleSegmentId) Project.select({ type: "subtitle", id: subtitleSegmentId });
    });
    return subtitleNode;
  }

  function updateSubtitle(segment) {
    const node = ensureSubtitle();
    if (!segment) {
      node.visible(false);
      subtitleSegmentId = null;
      subtitleSignature = "";
      transformer.nodes([]);
      overlayLayer.batchDraw();
      return;
    }
    subtitleSegmentId = segment.id;
    const style = segment.style;
    const fontSize = Math.max(8, style.fontSize * stageH);
    const signature = [
      segment.id,
      segment.text,
      style.fontSize,
      style.width,
      style.x,
      style.y,
      style.rotation,
      style.color,
      style.outlineColor,
      style.outlineWidth,
      style.opacity,
      style.bold,
      Math.round(fontSize),
      Math.round(stageW),
      Math.round(stageH),
    ].join("|");
    node.visible(true);
    if (signature === subtitleSignature) {
      const selectedSeg =
        Project.selection().type === "subtitle" && Project.selection().id === segment.id;
      if (selectedSeg) transformer.nodes([node]);
      return;
    }
    subtitleSignature = signature;
    node.setAttrs({
      x: style.x * stageW,
      y: style.y * stageH,
      rotation: style.rotation || 0,
      fontSize,
      width: Math.max(40, style.width * stageW),
      fill: style.color,
      stroke: style.outlineColor,
      strokeWidth: Math.max(0, style.outlineWidth * fontSize),
      opacity: style.opacity,
      fontStyle: style.bold ? "bold" : "normal",
      text: "",
    });
    const lines = MyanmarText.wrap(segment.text, node.width(), (value) =>
      measureText(value, fontSize, style.bold)
    );
    node.text(lines.join("\n"));
    node.offsetX(node.width() / 2);
    const textHeight =
      typeof node.getTextHeight === "function" ? node.getTextHeight() : node.height();
    node.height(textHeight);
    node.offsetY(textHeight / 2);
    const selected =
      Project.selection().type === "subtitle" && Project.selection().id === segment.id;
    node.draggable(true);
    if (selected) transformer.nodes([node]);
    overlayLayer.batchDraw();
  }

  /* ------------------------------- blurs ------------------------------ */
  function blurRadiusPx(intensity) {
    const scale = Math.min(stageW, stageH) / 720;
    return Math.max(1, intensity * Math.max(0.35, scale) * 1.2);
  }

  function ensureBlur(box) {
    let entry = blurNodes.get(box.id);
    if (entry) return entry;
    const shape = new Konva.Shape({
      x: box.x * stageW,
      y: box.y * stageH,
      width: box.w * stageW,
      height: box.h * stageH,
      draggable: true,
      sceneFunc(ctx, node) {
        const width = node.width();
        const height = node.height();
        const native = ctx._context || ctx;
        const source = videoEl;
        if (source && source.videoWidth) {
          native.save();
          native.beginPath();
          native.rect(0, 0, width, height);
          native.clip();
          native.globalAlpha = 1;
          if ("filter" in native) {
            native.filter = `blur(${blurRadiusPx(node.getAttr("intensity") || 10)}px)`;
            native.drawImage(source, -node.x(), -node.y(), stageW, stageH);
            native.filter = "none";
          } else {
            native.globalAlpha = 0.85;
            native.drawImage(source, -node.x(), -node.y(), stageW, stageH);
          }
          native.restore();
        }
        ctx.fillStrokeShape(node);
      },
      hitFunc(ctx, node) {
        ctx.beginPath();
        ctx.rect(0, 0, node.width(), node.height());
        ctx.closePath();
        ctx.fillStrokeShape(node);
      },
      fill: "rgba(0,0,0,0)",
    });
    const outline = new Konva.Rect({
      x: box.x * stageW,
      y: box.y * stageH,
      width: box.w * stageW,
      height: box.h * stageH,
      stroke: "rgba(255,255,255,0.55)",
      strokeWidth: 1,
      dash: [6, 5],
      listening: false,
      visible: false,
    });
    blurLayer.add(shape);
    uiLayer.add(outline);

    shape.on("mousedown touchstart", (event) => {
      event.cancelBubble = true;
      Project.select({ type: "blur", id: box.id });
    });
    shape.on("dragstart", () => Project.checkpoint());
    shape.on("dragmove", () => {
      shape.x(Math.max(-shape.width() + 10, Math.min(stageW - 10, shape.x())));
      shape.y(Math.max(-shape.height() + 10, Math.min(stageH - 10, shape.y())));
      outline.position({ x: shape.x(), y: shape.y() });
      const item = Project.doc().blurs.find((entryItem) => entryItem.id === box.id);
      if (item) {
        Project.updateBlur(
          box.id,
          {
            x: Math.max(0, Math.min(1, shape.x() / stageW)),
            y: Math.max(0, Math.min(1, shape.y() / stageH)),
            w: shape.width() / stageW,
            h: shape.height() / stageH,
          },
          { history: false }
        );
      }
      blurLayer.batchDraw();
    });
    shape.on("dragend", () => {
      applySelection();
      blurLayer.batchDraw();
    });
    shape.on("transformend", () => {
      const scaleX = shape.scaleX();
      const scaleY = shape.scaleY();
      shape.scale({ x: 1, y: 1 });
      const width = shape.width() * scaleX;
      const height = shape.height() * scaleY;
      shape.width(width);
      shape.height(height);
      Project.updateBlur(
        box.id,
        {
          x: Math.max(0, Math.min(1, shape.x() / stageW)),
          y: Math.max(0, Math.min(1, shape.y() / stageH)),
          w: Math.max(0.01, width / stageW),
          h: Math.max(0.01, height / stageH),
        },
        { history: false }
      );
      applySelection();
    });

    entry = { shape, outline };
    blurNodes.set(box.id, entry);
    return entry;
  }

  function syncBlurs(blurs) {
    const ids = new Set(blurs.map((box) => box.id));
    blurNodes.forEach((entry, id) => {
      if (!ids.has(id)) {
        entry.shape.destroy();
        entry.outline.destroy();
        blurNodes.delete(id);
      }
    });
    const selected = Project.selection();
    blurs.forEach((box) => {
      const { shape, outline } = ensureBlur(box);
      shape.setAttrs({
        x: box.x * stageW,
        y: box.y * stageH,
        width: Math.max(8, box.w * stageW),
        height: Math.max(8, box.h * stageH),
        visible: box.visible,
        draggable: !box.locked,
        intensity: box.intensity,
      });
      outline.setAttrs({
        x: shape.x(),
        y: shape.y(),
        width: shape.width(),
        height: shape.height(),
        visible:
          selected.type === "blur" && selected.id === box.id && box.visible,
      });
    });
    blurLayer.batchDraw();
    uiLayer.batchDraw();
  }

  /* -------------------------------- logos ----------------------------- */
  function getLogoImage(src) {
    if (logoImages.has(src)) return logoImages.get(src);
    const image = new window.Image();
    image.crossOrigin = "anonymous";
    image.onload = () => {
      if (Project.doc()) syncLogos(Project.doc().logos);
    };
    image.onerror = () => {};
    image.src = src.startsWith("/") ? src : `/api/media/${src}`;
    logoImages.set(src, image);
    return image;
  }

  function ensureLogo(logo) {
    let node = logoNodes.get(logo.id);
    if (node) return node;
    node = new Konva.Image({
      image: getLogoImage(logo.src),
      draggable: true,
      listening: true,
    });
    overlayLayer.add(node);
    node.on("mousedown touchstart", (event) => {
      event.cancelBubble = true;
      Project.select({ type: "logo", id: logo.id });
    });
    node.on("dragstart", () => Project.checkpoint());
    node.on("dragmove", () => {
      snapNode(node, node.getClientRect({ skipTransform: false }));
      node.x(Math.max(0, Math.min(stageW, node.x())));
      node.y(Math.max(0, Math.min(stageH, node.y())));
      overlayLayer.batchDraw();
    });
    node.on("dragend", () => {
      clearGuides();
      const current = Project.doc().logos.find((item) => item.id === logo.id);
      if (current) {
        Project.updateLogo(
          logo.id,
          {
            x: node.x() / stageW - current.w / 2,
            y: node.y() / stageH - current.h / 2,
          },
          { history: false }
        );
      }
      applySelection();
    });
    node.on("transformstart", () => Project.checkpoint());
    node.on("transformend", () => {
      const current = Project.doc().logos.find((item) => item.id === logo.id);
      if (!current) return;
      const width = (node.width() * node.scaleX()) / stageW;
      const height = (node.height() * node.scaleY()) / stageH;
      node.scale({ x: 1, y: 1 });
      Project.updateLogo(
        logo.id,
        {
          w: Math.max(0.02, width),
          h: Math.max(0.02, height),
          x: node.x() / stageW - width / 2,
          y: node.y() / stageH - height / 2,
          rotation: node.rotation() || 0,
        },
        { history: false }
      );
      sync();
    });
    logoNodes.set(logo.id, node);
    return node;
  }

  function syncLogos(logos) {
    if (!Project.doc() || !Array.isArray(logos)) return;
    const ids = new Set(logos.map((logo) => logo.id));
    logoNodes.forEach((node, id) => {
      if (!ids.has(id)) {
        node.destroy();
        logoNodes.delete(id);
      }
    });
    logos.forEach((logo) => {
      const node = ensureLogo(logo);
      node.width(Math.max(6, logo.w * stageW));
      node.height(Math.max(6, logo.h * stageH));
      node.offsetX(node.width() / 2);
      node.offsetY(node.height() / 2);
      node.position({
        x: (logo.x + logo.w / 2) * stageW,
        y: (logo.y + logo.h / 2) * stageH,
      });
      node.rotation(logo.rotation || 0);
      node.opacity(logo.opacity);
      node.visible(logo.visible);
      node.draggable(!logo.locked);
    });
    // stacking order from layer_order (bottom → top)
    const doc = Project.doc();
    const order = doc.layer_order || [];
    logos.forEach((logo) => {
      const node = logoNodes.get(logo.id);
      if (!node) return;
      const index = order.indexOf(logo.id);
      node.zIndex(Math.max(0, index + 1));
    });
    // Export order: blur → logos → subtitles, so the caption stays on top.
    if (subtitleNode && subtitleNode.visible()) subtitleNode.moveToTop();
    if (transformer) transformer.moveToTop();
    overlayLayer.batchDraw();
  }

  /* -------------------------------- sync ------------------------------ */
  function sync() {
    if (!ready || !Project.doc() || syncing) return;
    syncing = true;
    try {
      const doc = Project.doc();
      resize({ render: false });
      updateSubtitle(Project.segmentAt(currentTime));
      syncBlurs(doc.blurs);
      syncLogos(doc.logos);
      applySelection();
      stage.batchDraw();
    } finally {
      syncing = false;
    }
  }

  function setTime(time) {
    currentTime = Number(time) || 0;
    if (!ready) return;
    const segment = Project.segmentAt(currentTime);
    const selected = Project.selection();
    // keep the selected caption visible while editing even between captions
    if (!segment && selected.type === "subtitle" && Project.selectedSegment()) {
      // while editing, keep the selected caption visible even between captions
      updateSubtitle(Project.selectedSegment());
    } else {
      updateSubtitle(segment);
    }
    stage.batchDraw();
  }

  function playLoop() {
    cancelAnimationFrame(rafId);
    const tick = () => {
      if (!ready) return;
      stage.batchDraw();
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
  }

  function stopLoop() {
    cancelAnimationFrame(rafId);
    rafId = null;
    stage.batchDraw();
  }

  function draw() {
    if (ready) stage.batchDraw();
  }

  function destroy() {
    stopLoop();
    if (stage) stage.destroy();
    ready = false;
    subtitleNode = null;
    subtitleSegmentId = null;
    logoNodes.clear();
    blurNodes.clear();
  }

  return {
    init,
    resize,
    sync,
    setTime,
    playLoop,
    stopLoop,
    draw,
    destroy,
    waitForFonts,
    applySelection,
    measureText,
    blurRadiusPx,
    stageSize: () => ({ width: stageW, height: stageH }),
  };
})();
