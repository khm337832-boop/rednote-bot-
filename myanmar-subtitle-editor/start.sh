#!/usr/bin/env bash
# မြန်မာစာတန်းထိုး Video Editor — တစ်ချက်နှိပ်ပြီး စတင်ရန်
#
#   ./start.sh              → http://localhost:8000
#   PORT=9000 ./start.sh    → မတူညီသော port
#   EDITOR_PASSWORD=xxx ./start.sh  → စကားဝှက် ကာကွယ်မှု
#
# ပထမဆုံး အကြိမ်တွင် venv ဖန်တီးပြီး requirements ကို install လုပ်ပါမည်
# (faster-whisper ပါဝင်သဖြင့် မိနစ်အနည်းငယ် ကြာနိုင်သည်၊ ~2 GB ခန့်)။
set -euo pipefail

cd "$(dirname "$0")"

PY=${PYTHON:-python3}
PORT=${PORT:-8000}

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "❌ Python 3.10+ မတွေ့ပါ။ https://www.python.org/downloads/ မှ install လုပ်ပါ။" >&2
  exit 1
fi

if [ ! -d .venv ]; then
  echo "▶ venv ဖန်တီးနေသည်..."
  "$PY" -m venv .venv
fi

# shellcheck disable=SC1091
. .venv/bin/activate 2>/dev/null || . .venv/Scripts/activate

if [ ! -f .venv/.installed ]; then
  echo "▶ requirements install လုပ်နေသည် (ပထမအကြိမ်သာ)..."
  python -m pip install --upgrade pip >/dev/null
  python -m pip install -r backend/requirements.txt
  touch .venv/.installed
fi

# system ffmpeg တွင် libass မပါလျှင် imageio-ffmpeg ၏ static build ကို သုံးမည်
if ! python - <<'PY' >/dev/null 2>&1
import os, subprocess, sys
ff = os.environ.get("FFMPEG_BINARY") or "ffmpeg"
try:
    out = subprocess.run([ff, "-hide_banner", "-filters"], capture_output=True, text=True).stdout
except Exception:
    sys.exit(1)
sys.exit(0 if (" ass " in out and " subtitles " in out) else 1)
PY
then
  FF=$(python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2>/dev/null || true)
  if [ -n "$FF" ]; then
    export FFMPEG_BINARY="$FF"
    echo "▶ system FFmpeg တွင် libass မပါ — bundled FFmpeg ကို သုံးမည်: $FF"
  else
    echo "⚠️ libass ပါသော FFmpeg မတွေ့ပါ — စာတန်း burn-in လုပ်မရနိုင်ပါ (README §1 ကို ကြည့်ပါ)။" >&2
  fi
fi

if [ -z "${EDITOR_PASSWORD:-}" ]; then
  echo "⚠️ EDITOR_PASSWORD မသတ်မှတ်ထားပါ — app ကို လူတိုင်း ဝင်နိုင်သည်။"
fi

echo "▶ စတင်နေသည် — http://localhost:${PORT}  (ရပ်ရန် Ctrl+C)"
exec python -m uvicorn backend.main:app --host 0.0.0.0 --port "${PORT}"
