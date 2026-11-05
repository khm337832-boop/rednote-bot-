/* Headless test of the browser-side project model (no DOM needed).
   Run: node tests/js/project_model.test.js  (invoked from pytest) */
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const ROOT = path.resolve(__dirname, "../../frontend/js");
let failures = 0;

function check(name, condition, extra) {
  if (condition) {
    console.log(`  ✓ ${name}`);
  } else {
    failures += 1;
    console.error(`  ✗ ${name}${extra ? ` — ${extra}` : ""}`);
  }
}

const stored = {};
const sandbox = {
  console,
  window: {},
  document: { addEventListener() {} },
  localStorage: {
    getItem: (key) => (key in stored ? stored[key] : null),
    setItem: (key, value) => {
      stored[key] = String(value);
    },
  },
  requestAnimationFrame: (fn) => setTimeout(fn, 0),
  setTimeout,
  clearTimeout,
  setInterval,
  clearInterval,
  Math,
  JSON,
  Date,
  Number,
  String,
  Boolean,
  Object,
  Array,
  Map,
  Set,
  Promise,
  Error,
  Infinity,
  NaN,
  isNaN,
  parseInt,
  parseFloat,
};
sandbox.globalThis = sandbox;
sandbox.window = sandbox;

const requests = [];
sandbox.MSE = {
  uid: (prefix = "") => `${prefix}${Math.random().toString(16).slice(2, 10)}`,
  json: async (url, options) => {
    requests.push({ url, options });
    if (url === "/api/project/save") {
      return { project: { id: "savedproject01", updated_at: "2026-01-01T00:00:00Z", warnings: [] } };
    }
    if (url.startsWith("/api/project/")) {
      return { project: sandbox.__fixtureProject };
    }
    return {};
  },
  toast: () => {},
  storageGet: () => null,
  storageSet: () => {},
  formatTime: (v) => String(v),
};

vm.createContext(sandbox);
function load(file) {
  const source = fs.readFileSync(path.join(ROOT, file), "utf8");
  vm.runInContext(source, sandbox, { filename: file });
}
load("myanmar_text.js");
vm.runInContext("globalThis.MyanmarText = MyanmarText;", sandbox);
load("project.js");
vm.runInContext("globalThis.Project = Project;", sandbox);

const { Project, MyanmarText } = sandbox;

function fixture(segments) {
  return {
    id: "fixtureproject1",
    name: "စမ်းသပ်",
    video: { id: "video01", filename: "a.mp4", width: 1280, height: 720, duration: 20, has_audio: true },
    segments,
    logos: [],
    blurs: [],
    layer_order: ["subtitle"],
    style_defaults: {},
  };
}

const seg = (id, start, end, text) => ({
  id,
  start,
  end,
  text,
  style: { fontSize: 0.06, x: 0.5, y: 0.85, width: 0.8 },
});

console.log("MyanmarText");
check("syllables မင်္ဂလာပါ", JSON.stringify(MyanmarText.syllables("မင်္ဂလာပါ")) === JSON.stringify(["မင်္ဂ", "လာ", "ပါ"]));
check("syllables ဖြစ်တယ်", JSON.stringify(MyanmarText.syllables("ဖြစ်တယ်")) === JSON.stringify(["ဖြစ်", "တယ်"]));
check("normalize NFC + spaces", MyanmarText.normalize("  က\t\tခ  ") === "က ခ");
check("wrap သည် syllable မဖြတ်", (() => {
  const text = "မြန်မာစာတန်းထိုး ဖန်တီးနည်းကို အဆင့်ဆင့် ရှင်းပြပါမယ်";
  const lines = MyanmarText.wrap(text, 8, (value) => value.length);
  return lines.every((line) => {
    const bare = line.replace(/ /g, "");
    return MyanmarText.syllables(bare).join("") === bare;
  });
})());

console.log("Segment chain (CapCut)");
Project.setDocument(fixture([seg("a", 0, 2, "ပထမ"), seg("b", 2, 5, "ဒုတိယ"), seg("c", 5, 9, "တတိယ")]));
check("normalise chain", Project.doc().segments[1].start === 2 && Project.doc().segments[0].end === 2);

Project.setSegmentEnd("a", 3);
check("end ပြောင်းလျှင် နောက် start လိုက်ပြေး", Project.doc().segments[1].start === 3);
check("a.end == b.start", Project.doc().segments[0].end === 3);

Project.setSegmentStart("b", 4);
check("start ပြောင်းလျှင် ရှေ့ end လိုက်ပြေး", Project.doc().segments[0].end === 4);
check("b.start == 4", Project.doc().segments[1].start === 4);

Project.moveSegment("b", 1);
check("move သည် နှစ်ဖက် လိုက်ပြေး", Project.doc().segments[0].end === 5 && Project.doc().segments[2].start === 6);
check(
  "အားလုံး ကပ်လျက်",
  Project.doc().segments.every((segment, index, list) => index === list.length - 1 || segment.end === list[index + 1].start)
);

Project.setSegmentStart("a", -5);
check("အနုတ် အချိန် မဖြစ်", Project.doc().segments[0].start === 0);

Project.setSegmentEnd("a", 100);
check("ကပ်လျက် ကျော်မသွား", Project.doc().segments[0].end <= Project.doc().segments[1].start);
check("ကပ်လျက် ဆက်ရှိ", Project.doc().segments[0].end === Project.doc().segments[1].start);

console.log("Split / merge / delete");
Project.setDocument(fixture([seg("a", 0, 6, "မြန်မာစာတန်းထိုး ဖန်တီးနည်း ရှင်းပြပါမယ်")]));
const newId = Project.splitSegment("a", 3);
check("split သည် segment အသစ် ဖန်တီး", !!newId && Project.doc().segments.length === 2);
check("split အချိန် ကိုက်", Project.doc().segments[0].end === 3 && Project.doc().segments[1].start === 3);
check("split စာသား မပျက်", Project.doc().segments.map((s) => s.text).join("").replace(/ /g, "") === "မြန်မာစာတန်းထိုးဖန်တီးနည်းရှင်းပြပါမယ်");
Project.mergeWithNext("a");
check("merge သည် တစ်ခု ပြန်ဖြစ်", Project.doc().segments.length === 1 && Project.doc().segments[0].end === 6);

Project.deleteSegment("a");
check("ဖျက်ပြီး ဗလာ", Project.doc().segments.length === 0);

Project.setDocument(
  fixture([seg("a", 0, 2, "တစ်"), seg("b", 2, 4, "နှစ်"), seg("c", 4, 6, "သုံး")])
);
Project.deleteSegment("b");
check("အလယ် ဖျက်လျှင် ရှေ့က ဆက်ပေါ်", Project.doc().segments[0].end === 4);
check("ဖျက်ပြီး ၂ ခု", Project.doc().segments.length === 2);

Project.setDocument(fixture([seg("a", 0, 4, "ပထမ")]));
Project.addSegmentAt(2, { text: "အလယ်" });
check("playhead တွင် ထည့်လျှင် ခွဲပေး", Project.doc().segments.length === 2);
check("အသစ် စာသား", Project.doc().segments.length === 2 && Project.doc().segments[1].text.length > 0);

console.log("Undo / redo");
Project.setDocument(fixture([seg("a", 0, 2, "ပထမ"), seg("b", 2, 4, "ဒုတိယ")]));
const before = Project.doc().segments[0].end;
Project.setSegmentStyle("a", { color: "#FF0000" });
check("style ပြောင်း", Project.doc().segments[0].style.color === "#FF0000");
check("undo ရနိုင်", Project.canUndo());
Project.undo();
check("undo ပြန်ရ", Project.doc().segments[0].style.color !== "#FF0000" && Project.doc().segments[0].end === before);
Project.redo();
check("redo ပြန်ရ", Project.doc().segments[0].style.color === "#FF0000");

console.log("Logo / Blur");
Project.setDocument(fixture([seg("a", 0, 4, "စာတန်း")]));
Project.addLogo({ src: "logos/x.png", name: "logo.png", w: 0.2, naturalWidth: 200, naturalHeight: 100 });
check("logo ထည့်ပြီး", Project.doc().logos.length === 1);
const logoId = Project.doc().logos[0].id;
Project.updateLogo(logoId, { x: 0.4, opacity: 0.5 });
check("logo update", Project.doc().logos[0].opacity === 0.5);
Project.toggleLock("logo", logoId);
check("logo lock", Project.doc().logos[0].locked === true);
Project.reorder(logoId, "up");
Project.addBlur({ x: 0.1, y: 0.1, w: 0.3, h: 0.2, intensity: 12 });
check("blur ထည့်ပြီး", Project.doc().blurs.length === 1);
const blurId = Project.doc().blurs[0].id;
Project.updateBlur(blurId, { x: 0.95, w: 0.3 });
check("blur boundary အတွင်း", Project.doc().blurs[0].x + Project.doc().blurs[0].w <= 1.0001);
Project.deleteBlur(blurId);
check("blur ဖျက်ပြီး", Project.doc().blurs.length === 0);

console.log("Payload");
const payload = Project.toPayload();
check("payload has segments", Array.isArray(payload.segments) && payload.segments.length === 1);
check("payload no transient keys", !JSON.stringify(payload).includes("url"));

(async () => {
  await Project.save({ silent: true });
  const saveRequest = requests.filter((item) => item.url === "/api/project/save").pop();
  check("save ခေါ်သည်", !!saveRequest);
  check("save body တွင် name", saveRequest.options.body.name === "စမ်းသပ်");
  check("save ပြီး id ရ", Project.doc().id === "savedproject01");

  if (failures) {
    console.error(`\n${failures} စမ်းသပ်မှု မအောင်မြင်`);
    process.exit(1);
  }
  console.log("\nအားလုံး အောင်မြင်သည်");
})();
