# rednote-bot-

Telegram bot that accepts a directly uploaded Telegram video/document
video or a RedNote/video link and turns it into a Burmese recap video
(Gemini script → Gemini TTS → ffmpeg render), controlled with
**inline buttons, a persistent bottom keyboard, and Telegram's command
menu button** — no commands to memorize.

Version `v5.0-auto-recap` is a **one-shot automatic flow**: the user
sends one video or link and the bot does everything else. It downloads
the complete source, processes it internally in fixed chronological
**2-minute windows** (a 1:30 source is 45 windows; a 3-hour source is
90), joins every recap in order, verifies the final duration, and sends
the finished final video — **no per-part clicking**. Transient failures
are retried automatically at every stage (source download, script, TTS,
per-window render, join, and final upload), and a job is only reported
as done after every final piece has actually been delivered. The final
video is split only when Telegram's upload size requires it. It
installs the persistent bottom buttons from `/start`, `/menu`,
**or the first ordinary message/video/link sent to each newly deployed
process**. `/start` and `/menu` always force-install them again, explain the
full current workflow, list all 20 currently selected features, and open
the inline main menu:

| Button | What it does |
|---|---|
| 🎬 Recap စတင်မယ် | Send a Telegram video file or video link to start a recap |
| ⚙️ ဆက်တင်များ (20) | Open the 20-feature settings panel |
| 📊 အလုပ်အခြေအနေ | Show the running job (legacy manual sessions also show their join/discard controls) |
| 🛑 လုပ်နေတာ ရပ်မယ် | Cancel the running job |
| 📋 ဆက်တင် အကျဉ်း / 📈 စာရင်းချုပ် | Settings summary / job stats |
| ❓ အသုံးပြုနည်း / 🔄 ဆက်တင် ပြန်ချိန် | Help / reset settings |

Commands still work too: `/start` `/menu` `/settings` `/status`
`/cancel` `/summary` `/help`.

The live progress message also keeps **Cancel / Status / Settings / Main**
inline buttons attached. While a job is running, Status and Settings open
in separate messages so progress updates do not overwrite them. Cancel
persists the request for worker jobs as well as cancelling direct jobs.

## One-shot automatic two-minute pipeline (default)

The default is `RECAP_FLOW=auto`. Whatever the source duration is — up
to about 1:30 and beyond — Telegram intake follows this exact order with
**one user send and no further clicks**:

1. Accept a Telegram video/document video or a video link — including a
   URL inside RedNote share text and a public Telegram post link such as
   `https://t.me/name/123` — and download the complete source. A
   Telegram-uploaded file is still subject to Telegram's Bot API **20MB
   download limit**; larger material must be sent as a link. A bare
   Telegram bot/group/channel profile link (`https://t.me/name`) does not
   identify a video, so the bot asks the user to forward/send the video
   itself or copy that video post's link. Transient download failures
   are retried (`AUTO_SOURCE_ATTEMPTS`, default 2); permanent causes
   (profile/invite/private links, unsupported source, 20MB limit,
   login/geo/deleted) fail fast with Burmese guidance.
2. Split the source internally into fixed 2-minute windows in
   chronological order (1:30 → 45 windows, 3:00 → 90; only the last may
   be shorter). The full original is then deleted to save disk.
3. Recap each window in order: first-person full-retelling script
   (character names, protagonist/pronoun, prior-events summary, and
   last scene carried forward after every window), measured TTS, and a
   render targeted at the selected percentage of that window. Each
   window gets bounded whole-segment retries
   (`AUTO_SEGMENT_ATTEMPTS`, default 3) on top of the script/TTS
   internal retries; story context only advances after a window
   succeeds. Progress shows window x/y, time range, and percent.
4. Join all window recaps in order (join retried once on failure),
   re-check the whole-source duration target, split only for Telegram
   size delivery, and send the final video. Every final piece is
   retried up to 3 times; if any piece cannot be delivered the job is
   reported as failed — never as a partially delivered success.

### Legacy manual per-part flow (explicit opt-in only)

The retired click-each-part flow (extract/send one original window,
user clicks **🎬 Recap လုပ်မယ်**, then **🔗 ပြန်ဆက်မယ်** at the end) only
activates with **both** `RECAP_FLOW=manual` **and**
`ENABLE_MANUAL_FLOW=1`, so a stale `RECAP_FLOW=manual` left in a hosting
dashboard cannot bring the old behaviour back. Old part buttons from
before this change still work through the recovery path below and never
block new automatic jobs.

In that legacy manual flow, sessions are stored below
`WORK_ROOT/manual_sessions/` and expire after
`MANUAL_SESSION_TTL_HOURS` (24 by default). Render's disk is ephemeral
and is wiped on every deploy/restart, so each original-part caption and
session status message also carries a compact `RNB1.` recovery token
(source reference, part plan, story context, and delivered recap
Telegram file_ids). If a session button is tapped after the disk was
wiped, the bot rebuilds the session from that token — re-downloading
the full source and already-delivered recaps — and then continues the
tapped action; part messages sent by older versions are recovered from
their caption plus the part video itself when possible. If only the
TTL passed but the files survived, tapping a session button revives
the session instead of expiring it. Original parts are always sent as
Bot-API-downloadable copies (≤19MB; the retained full-quality segment
is still used for rendering) so message-based recovery can never be
blocked by Telegram's 20MB download limit, and every Telegram update
runs behind a guard that answers the button and sends a retryable
message instead of silently doing nothing on an unexpected error.
`WORK_ROOT` may be set to a mounted persistent disk path to keep
session files on disk across deploys as well. Discarding a session
removes its retained files.

Telegram jobs always run in the bot's direct pipeline by default.
Forwarding Telegram URL jobs to n8n additionally requires the explicit
opt-in `N8N_TELEGRAM_HANDOFF=1` (see below), so a stale
`PIPELINE_MODE=n8n` cannot reroute or stall ordinary Telegram recaps;
the worker API remains available to n8n independently.

## Final-duration and first-person guarantees

The percentage setting means the **actual final video duration**, not merely
an estimated number of script characters. At the default 90%, a 10-minute
source targets a 9-minute final video. The accepted window is the target
± max(0.35 seconds, 2% of the source duration), allowing only normal
media/TTS granularity.

The same rule is enforced **per internal 2-minute window** (for
example, each 120-second window targets 108 seconds at 90%), then
enforced again on the automatically joined final against the full
source duration.

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

## Direct script-generation recovery

The click-triggered Gemini script writer does not depend on only the first
preferred model or on perfectly bare JSON. It tries the preferred models in
order, records safe finish/block/token diagnostics without logging secrets,
accepts fenced or lightly wrapped JSON and common scene/text key aliases,
salvages complete scenes from a truncated response, and—if a response has
no usable scenes—requests a complete replacement script with a shorter,
strict prompt instead of trying to “fill gaps” in an empty list. If every
attempt still returns no usable scenes, the part fails with that specific
Gemini reason rather than a misleading `0/<target> chars` duration error.
The measured 90%-duration and per-scene first-person gates remain unchanged.

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
URLs must show `"version":"v5.0-auto-recap"`,
`"recap_flow":"auto_concat"`, `"manual_parts_enabled":false`,
`"telegram_pipeline":"direct_auto"`, `"n8n_telegram_handoff_enabled":false`,
`"source_segment_seconds":120`,
`"segment_pipeline":"download_split_120s_recap_each_concat_send"`,
`"telegram_buttons":"inline_persistent_menu_auto_one_shot_manual_opt_in"`,
`"telegram_link_handling":"profile_rejected_public_post_web_fallback"`, a current
`yt_dlp_version`, the service's real `public_host`, and the deployed
`render_git_repo_slug`, `render_git_commit`, and `render_git_branch`.
They must also show
`"direct_script_generation":"multi_model_retry_robust_json_empty_scene_recovery"`
and the configured `direct_script_models` list, an `auto_retry` object
(`source_attempts`, `segment_attempts`, `final_upload_attempts_per_part`,
`"partial_final_delivery":"rejected_as_failed"`), plus
`"manual_session_recovery":"telegram_message_payload_v1_with_legacy_caption_fallback"`.
A healthy endpoint proves deployment only — full acceptance still
requires one real Telegram send that returns a finished final video.
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

The direct one-shot Telegram flow above is the default. The optional n8n
orchestrated pipeline remains available in
[`n8n-workflow/`](n8n-workflow/), and its worker API is unchanged:

```text
Telegram bot -> n8n Webhook -> Render worker API
  download -> Whisper transcribe -> Gemini rewrite -> Gemini TTS
  -> video-only alignment/render -> Telegram delivery -> cleanup
```

For Telegram URL jobs to use that older automatic handoff, set
`RECAP_FLOW=auto`, `PIPELINE_MODE=n8n`, **and** `N8N_TELEGRAM_HANDOFF=1`
after importing and activating the workflow and configuring the
required worker/Whisper secrets. Without that explicit opt-in, Telegram
jobs never forward to n8n even if `PIPELINE_MODE=n8n` is left over in
the dashboard. See
[`n8n-workflow/README.md`](n8n-workflow/README.md) for setup and safety notes.
