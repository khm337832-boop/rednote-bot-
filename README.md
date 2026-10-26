# rednote-bot-

Telegram bot that turns a video link into a Burmese recap video
(Gemini script → Gemini TTS → ffmpeg render), controlled with both
**inline buttons and a persistent bottom keyboard** — no commands to memorize.

Send `/start` once in Telegram. The bot installs the persistent bottom
buttons and opens the inline main menu:

| Button | What it does |
|---|---|
| 🎬 Recap စတင်မယ် | Send a video link to start a recap |
| ⚙️ ဆက်တင်များ (20) | Open the 20-feature settings panel |
| 📊 အလုပ်အခြေအနေ | Show the running job |
| 🛑 လုပ်နေတာ ရပ်မယ် | Cancel the running job |
| 📋 ဆက်တင် အကျဉ်း / 📈 စာရင်းချုပ် | Settings summary / job stats |
| ❓ အသုံးပြုနည်း / 🔄 ဆက်တင် ပြန်ချိန် | Help / reset settings |

Commands still work too: `/start` `/menu` `/settings` `/status`
`/cancel` `/summary` `/help`.

The live progress message also keeps **Cancel / Status / Settings / Main**
inline buttons attached. While a job is running, Status and Settings open
in separate messages so progress updates do not overwrite them. Cancel
persists the request for worker jobs as well as cancelling direct jobs.

## Final-duration and first-person guarantees

The percentage setting means the **actual final video duration**, not merely
an estimated number of script characters. At the default 90%, a 10-minute
source targets a 9-minute final video. The accepted window is the target
± max(0.35 seconds, 2% of the source duration), allowing only normal
media/TTS granularity.

Character counts are only the first estimate. After Gemini TTS, the bot
measures the real narration seconds and this job's real chars/second. If the
narration is short, Gemini expands the full chronological script and the
changed scenes are synthesized again; if it is long, scenes are shortened
proportionally without deleting the ending. This repeats for up to four
calibration rounds. The rendered MP4 is then probed again, and delivery is
refused if its actual duration is outside the window.

Every narration scene must visibly speak in the protagonist's first person
with one consistent pronoun (`ကျွန်တော်` or `ကျွန်မ`) in the Burmese text
itself. A JSON `pronoun` field alone does not pass. Third-person scenes are
sent back for first-person repair, and the job fails rather than delivering
a third-person summary if repair does not succeed.

## The 20 features (⚙️ ဆက်တင်များ)

1. 🎯 Final ကြာချိန် — **actual final video duration** as % of source duration (90–100%; **90% minimum for every job**, default **90%**)
2. 🎙️ အသံ (Voice) — Charon / Fenrir / Puck / Kore / Aoede
3. 🔊 Deep အသံ — deep-voice filter on/off
4. ⏩ အမြန်နှုန်း — voice speed 0.9x–1.2x
5. 🔉 အသံအတိုးအကျယ် — volume 80–150%
6. 📐 ဗီဒီယို အရွယ် — 720 / 1280 / 1920 output
7. 💎 အရည်အသွေး — CRF 18 / 23 / 28
8. 🛡️ Shield (Crop+Flip) on/off
9. 🎨 အရောင် မြှင့် (color enhance) on/off
10. ⏸️ ခန်းကြား နားချိန် — pause between scenes 0–1.0s
11. ✂️ ဖြတ်ထုတ်ခွင့် — max cut ratio 0–15%
12. 📦 အပိုင်းခွဲ အချိန် — chunk size 3 / 5 / 10 min
13. 🗣️ Script အလျင် ချိန်ညှိ — initial TTS chars/sec estimate 5.5 / 6.5 / 7.5 (later corrected from measured TTS)
14. 👤 ပြောပြသူ ရှုထောင့် — protagonist first-person (ကျွန်တော်/ကျွန်မ) always; every scene is validated/repaired and the script retells the whole video in order, scene by scene, no summarizing
15. 🎭 ဇာတ်ကြောင်း လေသံ — natural / dramatic / emotional / calm tone
16. 📤 Telegram ခွဲပို့ — split files over 45MB on/off
17. 💬 Video Caption on/off
18. 🔔 ပြီး အကြောင်းကြား — completion notification on/off
19. ⚡ မြန်ဆန်မုဒ် — faster render preset on/off
20. 🚀 Link တန်း စ — auto-start on link, or ask for ✅ confirmation first

Settings are **per user**, saved to disk (`bot_settings.json` in the
work dir), and apply from the next video.

## Adding a new feature (button) later

The menu is generated from a registry — adding a feature takes
two small edits in `bot.py`, no menu code to touch:

1. Add a default in `DEFAULT_SETTINGS`:
   ```python
   "my_feature": True,
   ```
2. Add one entry in the `FEATURES` list:
   ```python
   {"key": "my_feature", "icon": "✨", "label": "My Feature",
    "kind": "toggle", "desc": "..."},
   ```
   `kind` is `"toggle"` (on/off, changes instantly) or `"cycle"`
   (opens a picker — add `"options": [(value, "label"), ...]`).

The button appears in ⚙️ ဆက်တင်များ automatically. Then read it
anywhere in the pipeline with `get_user_settings(chat_id)["my_feature"]`
(job code already receives the full settings snapshot as `job_settings` /
`st` in `build_chunk_recap`).

## Run

```bash
BOT_TOKEN=... GEMINI_API_KEY=... ADMIN_ID=... python bot.py
# FastAPI webhook on :10000  (POST /webhook)
```

Network-free regression tests:

```bash
python -m unittest discover -s tests -v
```

## Optional n8n pipeline

The existing direct bot pipeline remains the default. An optional n8n
orchestrated pipeline is available in [`n8n-workflow/`](n8n-workflow/):

```text
Telegram bot -> n8n Webhook -> Render worker API
  download -> Whisper transcribe -> Gemini rewrite -> Gemini TTS
  -> video-only alignment/render -> Telegram delivery -> cleanup
```

Set `PIPELINE_MODE=n8n` only after importing and activating the workflow and
configuring the required worker/Whisper secrets. See
[`n8n-workflow/README.md`](n8n-workflow/README.md) for setup and safety notes.
