# rednote-bot-

Telegram bot that accepts a directly uploaded Telegram video/document
video or a RedNote/video link and turns it into a Burmese recap video
(Gemini script → Gemini TTS → ffmpeg render), controlled with
**inline buttons, a persistent bottom keyboard, and Telegram's command
menu button** — no commands to memorize.

Version `v4.1-telegram-link-fix` downloads the complete source,
plans fixed **2-minute windows**, physically extracts and sends only the
first/current window with a **🎬 Recap လုပ်မယ်** button, and waits for the
user. Clicking that button writes the script and renders only that one
part; after it is delivered, the bot extracts and sends only the next
window. After every part is done, the user presses **🔗 ပြန်ဆက်မယ်** to
join and receive the final video. It installs the persistent bottom
buttons from `/start`, `/menu`,
**or the first ordinary message/video/link sent to each newly deployed
process**. `/start` and `/menu` always force-install them again, explain the
full current workflow, list all 20 currently selected features, and open
the inline main menu:

| Button | What it does |
|---|---|
| 🎬 Recap စတင်မယ် | Send a Telegram video file or video link to start a recap |
| ⚙️ ဆက်တင်များ (20) | Open the 20-feature settings panel |
| 📊 အလုပ်အခြေအနေ | Show the running job or the saved per-part session and its join/discard controls |
| 🛑 လုပ်နေတာ ရပ်မယ် | Cancel the running job |
| 📋 ဆက်တင် အကျဉ်း / 📈 စာရင်းချုပ် | Settings summary / job stats |
| ❓ အသုံးပြုနည်း / 🔄 ဆက်တင် ပြန်ချိန် | Help / reset settings |

Commands still work too: `/start` `/menu` `/settings` `/status`
`/cancel` `/summary` `/help`.

The live progress message also keeps **Cancel / Status / Settings / Main**
inline buttons attached. While a job is running, Status and Settings open
in separate messages so progress updates do not overwrite them. Cancel
persists the request for worker jobs as well as cancelling direct jobs.

## Sequential manual two-minute pipeline (default)

Set `RECAP_FLOW=manual` (the default in `render.yaml`). Whatever the
source duration is, Telegram intake follows this exact order:

1. Accept a Telegram video/document video or a video link — including a
   URL inside RedNote share text and a public Telegram post link such as
   `https://t.me/name/123` — and download the complete source. A
   Telegram-uploaded file is still subject to Telegram's Bot API **20MB
   download limit**; larger material must be sent as a link. A bare
   Telegram bot/group/channel profile link (`https://t.me/name`) does not
   identify a video, so the bot asks the user to forward/send the video
   itself or copy that video post's link.
2. Plan the 2-minute boundaries without physically splitting future
   windows. A 3-hour source becomes 90 planned parts; only the last part
   may be shorter. The complete downloaded source remains in the manual
   session for later on-demand extraction.
3. Extract and send only part 1 (or, later, only the current part) with a
   **🎬 Recap လုပ်မယ်** button. Its extraction progress shows MB written
   and elapsed time. No script for it—or any later part—has been written
   yet.
4. A click claims only that chronological part and only then writes its
   first-person script and renders it. Character names,
   protagonist/pronoun, prior-events summary, and last scene are updated
   after each finished part and carried into the next one.
5. After that recap is delivered, the bot extracts and sends only the
   next original window. If that advance is interrupted, **▶️ နောက်အပိုင်းပို့မယ်**
   retries just that next original. A future part cannot be recapped
   before earlier parts are done.
6. When every part is done, the user clicks **🔗 ပြန်ဆက်မယ်**. The bot joins
   the recap files in order, re-checks the whole-source duration target,
   splits only for Telegram size delivery, sends the final video, and
   deletes the session.

Sessions are stored below `WORK_ROOT/manual_sessions/` and expire after
`MANUAL_SESSION_TTL_HOURS` (24 by default). Render's disk is ephemeral;
the saved state survives an ordinary process restart on the same
instance but not replacement of the container/filesystem. Discarding a
session removes its retained files.

`RECAP_FLOW=auto` retains the v3.6 compatibility behaviour: automatic
per-part recap and automatic join, including the optional n8n handoff
for URL jobs. In manual mode, Telegram jobs intentionally stay in the
bot's direct script/render functions so each click is one bounded unit;
the worker API remains available to n8n independently.

## Final-duration and first-person guarantees

The percentage setting means the **actual final video duration**, not merely
an estimated number of script characters. At the default 90%, a 10-minute
source targets a 9-minute final video. The accepted window is the target
± max(0.35 seconds, 2% of the source duration), allowing only normal
media/TTS granularity.

In the manual flow the same rule is enforced **per 2-minute part** (for
example, each 120-second part targets 108 seconds at 90%), then enforced
again on the explicitly joined final against the full source duration.

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
12. 📦 မူရင်းပိုင်းချိန် — source is always planned into fixed 2-minute windows and extracted one at a time
13. 🗣️ Script အလျင် ချိန်ညှိ — initial TTS chars/sec estimate 5.5 / 6.5 / 7.5 (later corrected from measured TTS)
14. 👤 ပြောပြသူ ရှုထောင့် — protagonist first-person (ကျွန်တော်/ကျွန်မ) always; every scene is validated/repaired and the script retells the whole video in order, scene by scene, no summarizing
15. 🎭 ဇာတ်ကြောင်း လေသံ — natural / dramatic / emotional / calm tone
16. 📤 Telegram ခွဲပို့ — split files over 45MB on/off
17. 💬 Video Caption on/off
18. 🔔 ပြီး အကြောင်းကြား — completion notification on/off
19. ⚡ မြန်ဆန်မုဒ် — faster render preset on/off
20. 🚀 Video/Link တန်းစ — auto-start on an uploaded video or link, or ask for ✅ confirmation first

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

## Run / verify the live Render deployment

```bash
BOT_TOKEN=... GEMINI_API_KEY=... ADMIN_ID=... python bot.py
# FastAPI webhook on $PORT (Render default 10000)
```

On every startup the bot registers its commands, sets Telegram's chat menu
button to **commands**, and re-registers its webhook with
`allowed_updates=["message", "edited_message", "callback_query"]` so inline
button callbacks cannot remain disabled by an older webhook setting. The
webhook base is resolved in this order: explicit `WEBHOOK_BASE_URL` /
`PUBLIC_BASE_URL` (custom domains only — a stale value pointing at a
*different* `*.onrender.com` host is ignored), Render's actual
`RENDER_EXTERNAL_HOSTNAME`, then `RENDER_EXTERNAL_URL`. This guarantees a
freshly deployed instance always re-registers Telegram's webhook to
*itself*, never back to an older service. Setup is verified with `getMe` and
`getWebhookInfo` and retried with backoff.

A deployment is **not** verified until the live service reports the new
build itself. Open these URLs for the actual Render service:

```text
https://<service>.onrender.com/
https://<service>.onrender.com/health
https://<service>.onrender.com/healthz
```

`/healthz` is deliberately retained as an alias because the older live
service used that path; a Render service still configured to check it must
not fail new deployments with a 404 health check. For this release these
URLs must show `"version":"v4.1-telegram-link-fix"`,
`"recap_flow":"manual_sequential_per_part"`, `"manual_parts_enabled":true`,
`"source_segment_seconds":120`,
`"segment_pipeline":"download_plan_120s_extract_one_send_click_recap_then_next_explicit_join"`,
`"telegram_buttons":"inline_persistent_menu_and_per_part_recap"`,
`"telegram_link_handling":"profile_rejected_public_post_web_fallback"`, a current
`yt_dlp_version`, the service's real `public_host`, and the deployed
`render_git_repo_slug`, `render_git_commit`, and `render_git_branch`.
If `/health` returns `{"detail":"Not Found"}` or `/` shows another app, the
URL belongs to a different/older service or that service is not deploying
this repository's `main`; fixing that Render service's repository, branch,
root directory, or deploy status is required before Telegram can receive
the new buttons.

### Source download diagnostics and RedNote access

The downloader no longer sends yt-dlp stderr to `/dev/null`. Logs retain
only a bounded, redacted tail and the user receives the classified reason
(403/anti-bot, login/cookie required, geo/private/deleted, unsupported
extractor, network, or no playable format). Telegram profile, invite,
private-post, and public-post links are classified before yt-dlp; public
posts also get a Telegram web-preview direct-media fallback. The application uses the
installed `yt-dlp` Python module (with the `curl-cffi` extra), browser
headers, redirect resolution, RedNote canonical URL candidates, and
— for RedNote — direct media URLs embedded in the note page as fallbacks.

Most public links need no cookie. If RedNote blocks the server, set
**one** of these only in Render's Environment/Secret Files UI:

- `XHS_COOKIE`: a Netscape cookie file's contents, a `name=value; ...`
  cookie header, or a JSON cookie list for `xiaohongshu.com`.
- `YTDLP_COOKIES_FILE`: path to a mounted Netscape `cookies.txt`.

Never paste a bot token, cookie, API key, deploy hook, or other secret in
chat, GitHub issues, or source code. The bot sets `httpx`/`httpcore`
logging to WARNING and redacts bot-token-shaped values and sensitive URL
parameters to reduce accidental token exposure in application logs.

Network-free regression tests:

```bash
python -m unittest discover -s tests -v
```

## Optional n8n pipeline

The manual direct Telegram flow above is the default. The optional n8n
orchestrated pipeline remains available in
[`n8n-workflow/`](n8n-workflow/), and its worker API is unchanged:

```text
Telegram bot -> n8n Webhook -> Render worker API
  download -> Whisper transcribe -> Gemini rewrite -> Gemini TTS
  -> video-only alignment/render -> Telegram delivery -> cleanup
```

For Telegram URL jobs to use that older automatic handoff, set both
`RECAP_FLOW=auto` and `PIPELINE_MODE=n8n` after importing and activating
the workflow and configuring the required worker/Whisper secrets. With
the default `RECAP_FLOW=manual`, Telegram does not forward a per-part
click session to n8n. See
[`n8n-workflow/README.md`](n8n-workflow/README.md) for setup and safety notes.
