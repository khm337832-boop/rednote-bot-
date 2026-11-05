# မြန်မာစာတန်းထိုး Video Editor (CapCut စတိုင်)

ဗီဒီယို တင်လိုက်တာနှင့် **မြန်မာစာတန်းထိုး** ကို အလိုအလျောက် ဖန်တီးပေးပြီး
CapCut လိုပဲ စာတန်း၊ Logo နှင့် Blur ကို Canvas ပေါ်တွင် ဆွဲ/ချဲ့/လှည့်/နေရာချ
နိုင်သည့် သီးသန့် (private) Web Application ဖြစ်ပါသည်။

* **စကားပြော → စာတန်း**: faster-whisper (large-v3၊ GPU မရှိလျှင် CPU + int8)
  သို့မဟုတ် openai-whisper (config မှ ရွေး) ဖြင့် မူရင်းဘာသာကို auto-detect လုပ်သည်။
  မြန်မာမဟုတ်သော speech ကို မူလ timing မပြောင်းဘဲ Gemini ဖြင့် မြန်မာသို့ ပြန်သည်။
  Gemini မရလျှင် အခြားဘာသာကို မြန်မာစာတန်းအဖြစ် မပြဘဲ error ပြသည်။
  ဘာသာပြန်/အချိန်ကို export မလုပ်မီ လူကိုယ်တိုင် စစ်ဆေးပြင်ဆင်ပါ။
* **CapCut စတိုင် timing**: စကားပြောတစ်ကြောင်းချင်းစီကို သီးသန့် segment ခွဲပြီး
  `segment[i].end === segment[i+1].start` ဖြစ်သည်အထိ စာတန်း **ဆက်ပေါ်နေသည်**
  (စကားကြား silence ရှိရင်တောင် မပျောက်)။
* **Glass UI**: gradient နောက်ခံ၊ rgba panel + backdrop blur + glow button၊
  Dark/Light mode၊ mobile friendly။
* **Blur**: နေရာရွေး၊ ဆွဲ၊ ချဲ့၊ အားး 1–20 — preview တွင် canvas filter၊
  export တွင် FFmpeg `boxblur`။
* **Logo**: PNG/JPG/SVG၊ အများအပြား၊ drag/resize/rotate/opacity၊ lock/hide၊ z-order။
* **Export**: MP4 (H.264 + AAC) တွင် စာတန်း၊ Blur၊ Logo အားလုံး burn-in၊
  SRT သီးသန့် ဒေါင်းလုဒ်၊ Progress bar။
* **Save/Load**: JSON (UTF-8)၊ ၃၀ စက္ကန့်တစ်ခါ auto-save၊ Dashboard တွင်
  ပရောဂျက် စာရင်း/ဖွင့်/ဖျက်။
* **မြန်မာစာ**: UTF-8 (BOM ပါ SRT)၊ NFC normalize၊ syllable boundary တွင် သာ
  ခွဲ၊ Noto Sans Myanmar ဖောင့်ကို Canvas နှင့် FFmpeg နှစ်ခုလုံး သုံး။

---

## ၁။ လိုအပ်ချက်များ

| အရာ | လိုအပ်ချက် |
| --- | --- |
| Python | 3.10 နှင့်အထက် |
| FFmpeg | **libass + libfreetype + libharfbuzz** ပါသော build |
| Browser | Chrome / Edge / Safari / Firefox နောက်ဆုံးပေါ် (Canvas filter + ES2020) |
| Disk | ဗီဒီယို + export အတွက် လုံလောက်သော နေရာ (ဗီဒီယို၏ ~2–3 ဆ) |
| GPU (optional) | NVIDIA CUDA ရှိလျှင် faster-whisper က float16 ဖြင့် အမြန်လုပ် |

### FFmpeg တွင် libass/harfbuzz ပါမပါ စစ်နည်း

```bash
ffmpeg -hide_banner -filters | grep -E " (ass|subtitles|boxblur|overlay) "
ffmpeg -hide_banner -version | grep -o "enable-lib\(ass\|harfbuzz\|freetype\)"
```

* `ass` နှင့် `subtitles` filter နှစ်ခုလုံး ပေါ်ရမည်။
* `--enable-libass --enable-libharfbuzz --enable-libfreetype` ပေါ်ရမည်။
* မပေါ်လျှင် libass မပါသော build ဖြစ်သည်။ install လုပ်ပါ —

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install -y ffmpeg fonts-noto-core
# macOS (Homebrew build တွင် libass/harfbuzz ပါသည်)
brew install ffmpeg
# Windows: gyan.dev "full" build ကို သုံးပါ (libass/harfbuzz ပါသည်)
```

> အကယ်၍ system FFmpeg တွင် libass မပါလျှင် app သည် `imageio-ffmpeg`
> (pip ဖြင့် အလိုအလျောက် ပါလာ) ၏ static build ကို အသုံးပြုရန် ကြိုးစားသည်။
> ထို build တွင် libass + harfbuzz ပါသည်။ ရွေးချယ်မှုကို
> `MSE_FFMPEG=/path/to/ffmpeg` ဖြင့် အတင်းသတ်မှတ်နိုင်သည်။
> အတိအကျ ဘယ် FFmpeg ကို သုံးနေသည်ကို Dashboard → “စနစ် အခြေအနေ” နှင့်
> `GET /api/capabilities` တွင် ကြည့်နိုင်သည်။

---

## ၂။ Installation

```bash
cd myanmar-subtitle-editor

python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -r backend/requirements.txt
```

`faster-whisper` ပါဝင်သည်။ **openai-whisper** ကို ရွေးလိုလျှင် (PyTorch ကြီးသည်) —

```bash
pip install openai-whisper==20250625
```

ပြီးလျှင် `backend/config.json` တွင် `"stt_backend": "openai-whisper"` ထည့်ပါ။

### Run

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

* Dashboard — <http://localhost:8000/>
* Editor — <http://localhost:8000/editor/{project_id}>
* API docs — <http://localhost:8000/api/docs>

ပထမဆုံး အကြိမ် transcribe လုပ်သောအခါ Whisper မော်ဒယ်ကို
HuggingFace မှ အလိုအလျောက် download လုပ်သည် (`large-v3` ≈ 3 GB)။
ထို့နောက် cache ဖြစ်သွားသည်။ မော်ဒယ်ကို ကြိုတင် download လုပ်လိုလျှင် —

```bash
python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', compute_type='int8')"
```

---

## ၃။ အသုံးပြုနည်း (အဆင့်ဆင့်)

1. **ဗီဒီယို တင်ရန်** — Dashboard တွင် drag & drop သို့မဟုတ် နှိပ်ပြီး ရွေးပါ။
   (`mp4`, `mkv`, `mov`, `m4v`, `webm` — မူလသတ်မှတ်ချက် အများဆုံး **500 MB**)။
   ဖိုင်ကို **chunk လိုက် stream** ရေးသဖြင့် browser/ဆာဗာ memory မပြည့်ပါ။
2. **စာတန်းထိုး ဖန်တီးရန်** — တင်ပြီးသည်နှင့် အလိုအလျောက် transcribe လုပ်သည်
   (background job + progress polling ဖြစ်သဖြင့် request timeout မဖြစ်)။
3. **တည်းဖြတ်ခန်း** —
   * **Canvas**: စာတန်း/Logo/Blur ကို ဆွဲ၊ handle ဖြင့် ချဲ့၊ လှည့်။ Frame အပြင်မထွက်
     (boundary check)။ လမ်းညွှန် snap guide (အလယ်/တစ်စိတ်ပိုင်း) ပါသည်။
   * **Timeline**: စာတန်း block များကို ကပ်လျက် ပြသည်။ block ကို ဆွဲ၍ ရွှေ့၊
     ဘေး handle ဖြင့် အချိန်ချဲ့/ကျုံ့ — **ကပ်လျက် segment ၏ အချိန်ပါ လိုက်ပြင်သည်**။
     ခွဲ (✂)၊ ပူး (⇔)၊ ဖျက်၊ အသစ်ထည့်။
   * **ညာ Properties**: စာသား၊ စတင်/ဆုံးချိန်၊ ဖောင့်အရွယ်၊ အရောင်၊ ဘောင်၊
     X/Y/Width/Height/Rotation/Opacity ကို နံပါတ်ဖြင့် အတိအကျ ထည့်နိုင်သည်။
   * **ဘယ် အလွှာများ**: select / hide / lock / z-order / ဖျက်။
4. **သိမ်းရန်** — `Ctrl+S` သို့မဟုတ် “သိမ်းရန်”။ ၃၀ စက္ကန့်တစ်ခါ အလိုအလျောက်
   သိမ်းသည်။ “သိမ်းဆည်းပြီးပါပြီ” toast ပြသည်။
5. **ဗီဒီယို သိမ်းရန် (Export)** — MP4 (H.264 + AAC) အဖြစ် burn-in လုပ်သည်။
   Progress bar နှင့် ဒေါင်းလုဒ် ခလုတ် ရသည်။
6. **SRT ရယူရန်** — UTF-8 (BOM) SRT ဖိုင်။ Windows player များတွင်လည်း
   မြန်မာစာ မှန်ကန်စွာ ပေါ်သည်။

### Keyboard shortcuts

| လုပ်ဆောင်ချက် | Key |
| --- | --- |
| ဖွင့်/ရပ် | `Space` |
| ၁ စက္ကန့် / ၅ စက္ကန့် ရွှေ့ | `←` `→` / `Shift` + `←` `→` |
| ရွေးထားသည်ကို ဖျက် | `Delete` |
| နောက်ပြန် / ရှေ့သို့ | `Ctrl+Z` / `Ctrl+Y` |
| သိမ်း | `Ctrl+S` |
| Playhead တွင် ခွဲ | `Ctrl+B` |
| နောက် စာတန်းနှင့် ပူး | `Ctrl+M` |
| ရွေးချယ်မှု ပယ် | `Esc` |

---

## ၄။ CapCut စာတန်း timing (အရေးအကြီးဆုံး)

```
ဥပမာ စကားပြောချိန်:  0.0–1.5 “မင်္ဂလာပါ” · 2.5–3.5 “ဒီနေ့” · 5.0–7.0 “ရာသီဥတု ကောင်းတယ်”
မှန်ကန်သော စာတန်းအချိန်: 0.0→2.5 · 2.5→5.0 · 5.0→ဗီဒီယိုအဆုံး (သို့) ၇.၀+၃ စက္ကန့်
```

* `segment[i].end = segment[i+1].start` (silence ရှိရင်လည်း စာတန်း ဆက်ပေါ်)။
* နောက်ဆုံး segment သည် ဗီဒီယိုအဆုံးအထိ သို့မဟုတ် စကားပြီးချိန်မှ
  `max_hold_seconds` (မူလ ၃ စက္ကန့်) အထိ။ `last_hold_mode` ဖြင့် ပြောင်းနိုင်သည်။
* ဗီဒီယိုအစတွင် ပထမစကား မစခင် စာတန်း မပေါ်။

**အကောင်အထည်ဖော်မှု (`backend/subtitle_generator.py`)**

1. `word_timestamps=True` ဖြင့် transcribe။
2. မြန်မာစာတွင် စကားလုံးကြား space မရှိသဖြင့် token အရေအတွက်ကို မသုံးဘဲ —
   * token များကို ပြန်ပေါင်း (`join_tokens` — မြန်မာစာတွင် space မထည့်)၊
   * token ကြား gap > `gap_threshold` (မူလ 0.5s) ဆိုလျှင် segment အသစ်၊
   * Whisper ၏ sentence segment အဆုံးနှင့် `။ ၊ . ! ?` ကို ထည့်စဉ်းစား။
3. ရှည်လွန်းသော စာသား/အချိန်ကို **syllable နယ်နိမိတ်တွင်သာ** ခွဲသည်။
   * grapheme/syllable ၃၅ လုံးကျော် (`max_chars`) သို့မဟုတ် ၆ စက္ကန့်ကျော်
     (`max_duration`) ဆိုလျှင် နှစ်ပိုင်း (လိုအပ်လျှင် ပိုမို) ခွဲ။
   * အချိန်ကို word timestamp အလိုက် အချိုးကျ မျှဝေသည်။
4. နောက်ဆုံး `enforce_hold_timing()` က above rule များကို တိတိကျကျ ချိန်သည်။

စမ်းသပ်မှု — `tests/test_segments.py::test_capcut_example_timing` သည် အထက်ပါ
ဥပမာကို တိတိကျကျ စစ်ဆေးသည်။

```bash
python -m pytest tests -q
```

---

## ၅။ မြန်မာစာ ကိုင်တွယ်မှု

* ဖိုင်အားလုံး **UTF-8**၊ HTML တွင် `<meta charset="UTF-8">`။ Python တွင်
  `encoding="utf-8"` အမြဲသုံး၊ SRT ကို `utf-8-sig` (BOM) ဖြင့် သိမ်း။
* Zawgyi မဟုတ်ဘဲ **Unicode** သာ။ Whisper ရလဒ်ကို **NFC normalize** လုပ်သည်
  (zero-width ဖယ်၊ space ညှိ)။ Zawgyi ဟု သံသယဖြစ်လျှင် `looks_like_zawgyi()`
  က သတိပေးသည်။
* **syllable ခွဲခြင်း** (`backend/myanmar_text.py`, `frontend/js/myanmar_text.js`) —
  orthographic cluster (ဗျည်း + တွဲဗျည်း + သရ + အသံ) ကို ပေါင်းပြီး
  `င်/န်/စ်` ကဲ့သို့ final ကို ရှေ့ syllable နှင့် ပူးသည်။
  `က္က` (stack) နှင့် `င်္က` (kinzi) ကိုလည်း မှန်ကန်စွာ ကိုင်တွယ်သည်။
  ထို့ကြောင့် စာကြောင်းခွဲခြင်း၊ segment ခွဲခြင်း၊ Canvas wrap အားလုံးတွင်
  syllable အလယ် မဖြတ်ပါ။
* **Render**: `drawtext` ကို **မသုံး**ပါ (မြန်မာ shaping မှားနိုင်သည်)။
  ASS ဖိုင်ထုတ်ပြီး libass ၏ `ass` filter ဖြင့် burn-in လုပ်သည်။
  ဖောင့်ကို `fontsdir` ဖြင့် တိုက်ရိုက် ညွှန်သဖြင့် system font မလိုအပ်။
* **Preview**: browser ၏ text layout (Konva Text + `document.fonts.ready`) ကို
  သုံးသည်။ ကိုယ်တိုင် စာလုံးဖြတ် မရေးပါ။
* ဖောင့် — `frontend/assets/fonts/` နှင့် `backend/fonts/` နှစ်ခုလုံးတွင်
  `NotoSansMyanmar-Regular.ttf` / `-Bold.ttf` (SIL OFL)။ Pyidaungsu ကို သုံးလိုလျှင်
  ဖိုင်ကို ထို folder နှစ်ခုသို့ ထည့်ပြီး `backend/config.py` ရှိ `FONT_FAMILY`,
  `FONT_REGULAR`, `FONT_BOLD` နှင့် `frontend/css/glass.css` ၏ `@font-face` ကို
  ပြောင်းပါ။

---

## ၆။ ဆက်တင် (config)

`backend/config.example.json` ကို `backend/config.json` အဖြစ် copy လုပ်ပြီး
ပြင်နိုင်သည်။ Environment variable ဖြင့်လည်း ပြောင်းနိုင်သည် (config ထက် ဦးစားပေး)။

| Key | Env | မူလ | အဓိပ္ပာယ် |
| --- | --- | --- | --- |
| `stt_backend` | `MSE_STT_BACKEND` | `faster-whisper` | `faster-whisper` / `openai-whisper` |
| `model_size` | `MSE_MODEL_SIZE` | `large-v3` | `large-v3`, `large-v2`, `medium`, `small`, `base`, `tiny` |
| `language` | `MSE_LANGUAGE` | `my` | Whisper ဘာသာစကား |
| `device` | `MSE_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `compute_type` | `MSE_COMPUTE_TYPE` | `auto` | `float16` (GPU), `int8`(CPU), `float32` |
| `beam_size` | `MSE_BEAM_SIZE` | `5` | မြင့်လျှင် တိကျသည်၊ နှေးသည် |
| `vad_filter` | `MSE_VAD_FILTER` | `true` | တိတ်ဆိတ်ချိန် ဖယ်ရန် |
| `gap_threshold` | `MSE_GAP_THRESHOLD` | `0.5` | token ကြား ဤထက်ကြာလျှင် segment အသစ် |
| `punctuation_break` | – | `true` | `။ ။ ၊ ! ?` တွင် ခွဲ |
| `max_chars` | `MSE_MAX_CHARS` | `35` | စာတန်းတစ်ခု၏ အများဆုံး grapheme |
| `max_duration` | `MSE_MAX_DURATION` | `6.0` | စကားပြော အများဆုံး စက္ကန့် |
| `min_duration` | `MSE_MIN_DURATION` | `0.3` | အနည်းဆုံး ပေါ်ချိန် |
| `max_hold_seconds` | `MSE_MAX_HOLD_SECONDS` | `3.0` | နောက်ဆုံး စာတန်း ဆက်ပေါ်ချိန် |
| `last_hold_mode` | `MSE_LAST_HOLD_MODE` | `cap` | `cap` (အများဆုံး ၃ စက္ကန့်) / `video_end` |
| `max_upload_mb` | `MSE_MAX_UPLOAD_MB` | `500` | တင်နိုင်သော အများဆုံး |
| `demo_mode` | `MSE_DEMO_MODE` | `auto` | `auto` (Whisper မရှိလျှင်) / `on` / `off` |
| `video_crf` | `MSE_VIDEO_CRF` | `20` | ပိုငယ် = အရည်အသွေးမြင့်၊ ဖိုင်ကြီး |
| `max_render_height` | `MSE_MAX_RENDER_HEIGHT` | `0` | export resolution ကန့်သတ် (0 = မူလ) |
| `access_password` | `EDITOR_PASSWORD` | (ဗလာ) | သီးသန့် စကားဝှက် |
| `ffmpeg_binary` | `MSE_FFMPEG` / `FFMPEG_BINARY` | (auto) | FFmpeg path |

### GPU / CPU ရွေးနည်း

* **GPU ရှိ** — `device: "cuda"`, `compute_type: "float16"` → large-v3 ဖြင့်
  real-time ထက် မြန်နိုင်သည် (GPU အလိုက်)။
* **GPU မရှိ** — `device: "cpu"`, `compute_type: "int8"` (မူလ auto က ဒီလိုရွေးသည်)။
  large-v3 သည် CPU တွင် နှေးသည် (၁ နာရီ ဗီဒီယိုအတွက် နာရီအတော်ကြာနိုင်သည်)။
  မြန်လိုလျှင် `medium` သို့မဟုတ် `small` သုံးပါ (တိကျမှု ကျသည်)။
* `vad_filter: true` က တိတ်ဆိတ်ချိန်ကို ဖယ်သဖြင့် ပိုမြန်သည်။
* မော်ဒယ်ကို memory တွင် cache လုပ်ထားသဖြင့် ဒုတိယအကြိမ် မြန်သည်။

---

## ၇။ API

| Method | Path | ဖော်ပြချက် |
| --- | --- | --- |
| GET | `/api/capabilities` | FFmpeg/libass/harfbuzz/Whisper/font စစ်ဆေးမှု + config |
| GET | `/healthz` | health check |
| POST | `/api/upload/start` | chunk upload စတင် `{filename,size}` |
| PUT | `/api/upload/{upload_id}?offset=N` | chunk (raw bytes) ရေးသွင်း |
| POST | `/api/upload/{upload_id}/complete` | စစ်ဆေးပြီး video အဖြစ် သိမ်း |
| POST | `/api/upload` | တစ်ခါတည်း streaming upload (multipart) |
| GET | `/api/video/{id}` | Range request ပံ့ပိုးသော video stream |
| GET | `/api/video/{id}/meta` | duration/resolution |
| GET | `/api/video/{id}/thumbnail` | ပုံငယ် (JPEG) |
| POST | `/api/logo` | Logo တင် (SVG ကို PNG အဖြစ် ပြောင်း) |
| GET | `/api/media/logos/{name}` | Logo ဖိုင် |
| POST | `/api/transcribe` | `{video_id, demo?, options?}` → job |
| GET | `/api/jobs/{id}` | job status/progress/stage/result |
| POST | `/api/jobs/{id}/cancel` | ရပ် |
| POST | `/api/project/save` | သိမ်း (id မပါလျှင် အသစ်) |
| GET | `/api/project/list` | ပရောဂျက် စာရင်း |
| GET | `/api/project/{id}` | ဖွင့် |
| DELETE | `/api/project/{id}` | ဖျက် |
| POST | `/api/export` | `{project}` → job (H.264/AAC burn-in) |
| GET | `/api/export/{job_id}/download` | ရလဒ် MP4 |
| POST | `/api/srt` | `{project}` → SRT (utf-8-sig) |
| GET | `/api/srt/{project_id}` | သိမ်းထားသော ပရောဂျက် SRT |
| POST | `/api/demo/video` | သရုပ်ပြ ဗီဒီယို ဖန်တီး (စမ်းသပ်ရန်) |
| GET/POST | `/api/auth/status`, `/api/auth/login`, `/api/auth/logout` | စကားဝှက် |

ပရောဂျက် JSON ဥပမာ —

```json
{
  "id": "9a8a2bfe85ca4ccb",
  "name": "သရုပ်ပြ ပရောဂျက်",
  "video": {"id": "demovideo01", "filename": "a.mp4", "width": 1280, "height": 720, "duration": 22.0},
  "segments": [
    {"id": "s1", "start": 0.0, "end": 2.5, "text": "မင်္ဂလာပါ",
     "style": {"fontSize": 0.055, "x": 0.5, "y": 0.86, "width": 0.82,
                "color": "#FFFFFF", "outlineColor": "#000000", "outlineWidth": 0.08,
                "opacity": 1, "bold": true, "rotation": 0}}
  ],
  "logos": [{"id": "l1", "src": "logos/abc.png", "x": 0.03, "y": 0.03, "w": 0.18, "h": 0.09,
             "rotation": 8, "opacity": 0.9, "visible": true, "locked": false}],
  "blurs": [{"id": "b1", "x": 0.6, "y": 0.62, "w": 0.25, "h": 0.2, "intensity": 14,
             "visible": true, "locked": false}],
  "layer_order": ["subtitle", "l1", "b1"],
  "style_defaults": {}
}
```

position/size အားလုံးကို **frame ၏ အချိုးအစား (0–1)** အဖြစ် သိမ်းသဖြင့်
resolution မည်သို့ပြောင်းလဲပြောင်း နေရာမှန်သည်။

---

## ၈။ ဖိုင်ဖွဲ့စည်းပုံ

```
myanmar-subtitle-editor/
├── backend/
│   ├── main.py                # FastAPI: upload / jobs / project / export / auth
│   ├── config.py              # config + FFmpeg/Whisper detection
│   ├── subtitle_generator.py  # whisper + CapCut segment logic
│   ├── myanmar_text.py        # NFC, syllable, wrap
│   ├── video_renderer.py      # ASS/libass burn-in, boxblur, logo, SRT
│   ├── project_manager.py     # JSON save/load (atomic)
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   ├── config.example.json
│   ├── fonts/                 # Noto Sans Myanmar (FFmpeg fontsdir)
│   ├── uploads/               # videos · logos · exports (runtime)
│   └── projects/              # <id>.json (runtime)
├── frontend/
│   ├── index.html             # Dashboard (Glass UI)
│   ├── editor.html            # Canvas + Timeline + Properties
│   ├── css/glass.css          # theme, glass, dashboard
│   ├── css/editor.css         # editor layout
│   ├── js/api.js              # fetch, toast, auth, job polling
│   ├── js/dashboard.js
│   ├── js/editor.js
│   ├── js/timeline.js
│   ├── js/layers.js           # Konva canvas: subtitle/logo/blur
│   ├── js/project.js          # model, undo/redo, autosave
│   ├── js/myanmar_text.js     # browser syllable/wrap (Python နှင့် တူ)
│   └── assets/fonts/ · assets/vendor/konva.min.js
├── tests/
│   ├── test_segments.py       # CapCut timing (spec ဥပမာ)
│   ├── test_myanmar_text.py
│   ├── test_renderer.py
│   ├── test_api.py
│   ├── test_frontend_model.py
│   └── js/project_model.test.js
├── Dockerfile
├── render.yaml
└── README.md
```

---

## ၉။ စမ်းသပ်မှုများ

```bash
pip install -r backend/requirements-dev.txt
python -m pytest tests -q
```

* `test_segments.py` — CapCut timing (spec ဥပမာ၊ silence hold၊ max hold၊
  syllable မဖြတ်ခြင်း၊ gap/punctuation/sentence break၊ merge)။
* `test_myanmar_text.py` — syllable၊ NFC၊ Zawgyi detect၊ wrap၊ split။
* `test_renderer.py` — ASS escaping/wrap၊ filter graph (crop→boxblur→overlay၊
  logo rotate/opacity)၊ SRT BOM၊ တကယ့် FFmpeg render (ဖြူစာလုံး ပေါ်မပေါ်
  pixel စစ်၊ blur variance ကျမကျ စစ်)။
* `test_api.py` — chunk upload၊ Range stream၊ transcribe demo job၊ project CRUD၊
  export job + download၊ SRT၊ စကားဝှက် ကာကွယ်မှု။
* `test_frontend_model.py` + `tests/js/project_model.test.js` — browser model
  (Node ဖြင့်)၊ chain invariant၊ undo/redo၊ logo/blur boundary။

---

## ၁၀။ Known limitations (သိထားရန်)

1. **Preview ≠ Export (အနည်းငယ်)** — 
   * Blur: preview သည် canvas `filter: blur(px)`၊ export သည် FFmpeg
     `boxblur` (ကွာခြားနိုင်သည်)။ formula ကို တူအောင် ချိန်ထားသည်
     (`intensity * min(w,h)/720 * 1.2`)။
   * စာသား line wrap: preview သည် browser metrics၊ export သည် Pillow metrics
     သုံးသဖြင့် စာလုံးအရေအတွက် အနည်းငယ် ကွာနိုင်သည်။ ဖောင့်တူညီသည်။
   * Canvas ဘောင် (stroke) သည် စာလုံး အနှစ်ခြား၊ libass ဘောင် (\bord) သည်
     အပြင်ဘက် ဖြစ်သဖြင့် အနည်းငယ် ကွာနိုင်သည်။
2. **Whisper တိကျမှု** — ဆူညံသော အသံ၊ ရောနှောဘာသာစကား၊ သီချင်းတွင် မှားနိုင်သည်။
   စာတန်းကို အမြဲ စစ်ဆေးပြီး ပြင်ပါ။ Word timestamp မပေးသော မော်ဒယ်တွင်
   အချိန်ကို segment အတွင်း အညီအမျှ ခွဲသည်။
3. **ဗီဒီယို အရှည်/အရွယ်** — 500 MB (config ပြောင်းနိုင်)။ အလွန်ရှည်သော
   ဗီဒီယိုတွင် transcribe/export အချိန် ကြာနိုင်သည်။ Export သည် CPU ကို
   အသုံးပြုသည် (GPU encode မပါ)။
4. **Silent video** — အသံမရှိသော ဗီဒီယိုတွင် စာတန်း အလိုအလျောက် မဖန်တီးနိုင်ပါ
   (invent မလုပ်ပါ)။ ကိုယ်တိုင် ထည့်နိုင်သည်။
5. **Jobs** — job အခြေအနေကို memory တွင် ထားသဖြင့် ဆာဗာ restart လျှင်
   ပျောက်သည်။ သိမ်းထားသော ပရောဂျက်များ/ဖိုင်များ မပျောက်ပါ။
6. **Storage** — တင်ထားသော ဗီဒီယို၊ export ဖိုင်များကို အလိုအလျောက်
   မဖျက်ပါ။ disk ပြည့်မလာစေရန် `backend/uploads/` ကို ကာလအလိုက်
   ရှင်းလင်းပါ။
7. **Multi-user** — တစ်ခုတည်းသော app ဖြစ်သည်။ ပရောဂျက် စာရင်းကို
   အားလုံး မြင်နိုင်သဖြင့် **EDITOR_PASSWORD** သတ်မှတ်ပြီး သီးသန့်
   server/HTTPS အောက်တွင် ထားပါ။
8. **Demo mode** — Whisper မထည့်ရသေးလျှင် “သရုပ်ပြ စာတန်း” ဖြင့် UI ကို
   စမ်းနိုင်သည်။ ထို စာသားများသည် **စမ်းသပ်ရန် သာ** ဖြစ်ပြီး ဗီဒီယို၏
   အသံနှင့် မသက်ဆိုင်ပါ။ (`demo_mode: "off"` ဖြင့် ပိတ်နိုင်သည်။)
9. **SVG Logo** — librsvg ပါသော FFmpeg/`cairosvg` ရှိမှ PNG အဖြစ် ပြောင်းနိုင်သည်။
   မရလျှင် PNG/JPG ဖြင့် တင်ပါ။
10. **Browser autoplay** — အသံပါ ဗီဒီယိုကို အလိုအလျောက် ဖွင့်ခွင့် မပြုသော
    browser များရှိသည်။ ▶ ကို နှိပ်ပါ။

---

## ၁၁။ သီးသန့် (private) ထားရှိမှု

```bash
export EDITOR_PASSWORD='လုံခြုံသော-စကားဝှက်'
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

စကားဝှက် သတ်မှတ်ထားလျှင် `/api/*` အားလုံး (auth endpoints မှလွဲ၍)
401 ပြန်သည်၊ Dashboard/Editor တွင် login overlay ပေါ်သည်။ cookie သည်
HttpOnly + SameSite=Lax (HTTPS တွင် Secure)။ စကားဝှက် မသတ်မှတ်ထားလျှင်
Dashboard တွင် သတိပေးချက် ပြသည်။

> Public server တွင် တင်လျှင် HTTPS reverse proxy (Caddy/Nginx/Cloudflare) ကို
> သုံးပါ။ ဤ app သည် session အခြေခံ auth သာဖြစ်ပြီး multi-tenant
> permission system မပါပါ။

---

## ၁၂။ Docker / Deploy

```bash
docker build -t myanmar-subtitle-editor .
docker run --rm -p 8000:8000 \
  -e EDITOR_PASSWORD='change-me' \
  -e HF_HOME=/data/hf \
  -v mse-data:/data \
  myanmar-subtitle-editor
```

* Image တွင် Debian FFmpeg (libass/harfbuzz ပါ) နှင့် ဖောင့် ပါသည်။
* `HF_HOME` volume ဖြင့် Whisper မော်ဒယ်ကို ပြန်မယူဘဲ သိမ်းထားနိုင်သည်။
* Render blueprint သည် free instance သာ သုံးပြီး persistent disk မပါပါ။ ထို့ကြောင့် restart/deploy ဖြစ်လျှင် video၊ project နှင့် model cache ပျောက်နိုင်သည် — အမြဲတမ်း save/load အဖြစ် အာမမခံပါ။
* Free instance ၏ RAM/disk အကန့်အသတ်ကြောင့် 500 MB video သို့မဟုတ် Whisper large-v3 အလုပ်လုပ်မည်ဟု အာမမခံနိုင်ပါ။ Blueprint တွင် small/int8 ကို သုံးထားသည်။
* `GEMINI_API_KEY` ကို environment တွင် သတ်မှတ်ထားမှ မြန်မာမဟုတ်သော speech ကို ဘာသာပြန်နိုင်သည် (free tier availability/quota အပေါ် မူတည်သည်)။ `EDITOR_PASSWORD` မရှိသော Render deployment တွင် private API များကို ပိတ်ထားသည်။ Secret ကို chat ထဲ မပို့ပါနှင့်။
* လက်ရှိ standalone editor တွင် video upload ကိုသာ ပံ့ပိုးပြီး arbitrary video link ကို တိုက်ရိုက် import မလုပ်နိုင်သေးပါ။ မူရင်း subtitle ကို အလိုအလျောက် detect/ဖုံးခြင်းလည်း မပြီးသေးပါ — blur layer ကို ကိုယ်တိုင် နေရာချနိုင်သည်။

---

## ၁၃။ ပြဿနာ ဖြေရှင်းခြင်း

| လက္ခဏာ | အကြောင်းရင်း / ဖြေရှင်းနည်း |
| --- | --- |
| “FFmpeg တွင် libass မပါပါ” | libass မပါသော build။ `apt install ffmpeg` (Debian) သို့မဟုတ် gyan.dev full build။ `MSE_FFMPEG` ဖြင့် ညွှန်။ |
| စာတန်း tofu (□) ဖြစ် | `backend/fonts/NotoSansMyanmar-*.ttf` ရှိမရှိ၊ ASS ၏ Fontname = “Noto Sans Myanmar” ဖြစ်မဖြစ် စစ်ပါ။ |
| “Whisper အသင့် မဖြစ်ပါ” | `pip install -r backend/requirements.txt`၊ ထို့နောက် server restart။ |
| မော်ဒယ် download မရ | HuggingFace သို့ ကွန်ရက် လိုအပ်။ `HF_HOME` cache ကို သိမ်းထားပါ။ |
| Transcribe အလွန်နှေး | `device: cpu` + `compute_type: int8`; `model_size: small`; `vad_filter: true`။ |
| Export မရ / crash | Disk နေရာ၊ `backend/uploads/_work` ရှင်းလင်းမှု၊ FFmpeg log (API response တွင် ပါသည်)။ |
| စာတန်း မပေါ် | Timeline တွင် segment ရှိမရှိ၊ current time က segment အတွင်း ရှိမရှိ၊ `visible` ဖြစ်မဖြစ် စစ်ပါ။ |
| Blur ကွာ | Preview (canvas) နှင့် export (boxblur) မတူညီသော filter ဖြစ်သည် — ရည်ရွယ်ချက် အတွက် ချိန်ထားသော်လည်း အနည်းငယ် ကွာနိုင်သည်။ |
| Upload ရပ်/ပျက် | chunk တစ်ခု မအောင်မြင်လျှင် ပြန်တင်ပါ။ proxy ၏ body size limit ကို စစ်ပါ (`client_max_body_size 0` ကဲ့သို့)။ |

---

## ၁၄။ လိုင်စင် / အသိအမှတ်ပြုချက်

* Noto Sans Myanmar — SIL Open Font License 1.1 (`backend/fonts/`၊ `frontend/assets/fonts/`)။
* Konva.js — MIT (`frontend/assets/vendor/konva.min.js`)။
* FFmpeg — LGPL/GPL (သီးသန့် install)။
* faster-whisper / openai-whisper — MIT။

ဤ app သည် ဗီဒီယိုနှင့် စာတန်းများကို သင့်ဆာဗာပေါ်တွင်သာ သိမ်းဆည်းသည်
(cloud API မလိုအပ်)၊ သို့သော် သီးသန့် server တွင် ထားရှိရန် တာဝန်သည်
အသုံးပြုသူ၏ တာဝန် ဖြစ်သည်။
