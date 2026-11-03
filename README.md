# rednote-bot-

Telegram bot that accepts a directly uploaded Telegram video/document
video or a RedNote/video link and turns it into a Burmese recap video
(Gemini script → Gemini TTS → ffmpeg render), controlled with
**inline buttons, a persistent bottom keyboard, and Telegram's command
menu button** — no commands to memorize.

Version `v5.22-script-first-no-overshoot` is a **one-shot automatic flow**: the user
sends one video or link and the bot does everything else. It downloads
the complete source, processes it internally in fixed chronological
**2-minute windows** (a 1:30 source is 45 windows; a 3-hour source is
90), joins every recap in order, verifies the final duration, and sends
the finished final video — **no per-part clicking**. Transient failures
are retried automatically at every stage (source download, script, TTS,
per-window render, join, and final upload), and a job is only reported
as done after every final piece has actually been delivered. The final
video is split only when Telegram's upload size requires it. Splitting
uses stream-copy at keyframe-snapped ~2-minute boundaries (accepted
when the parts cover the full source and none exceeds 1.5× the window;
only sparse-keyframe sources fall back to a bounded 720p/ultrafast
accurate extraction), so the split stage takes seconds instead of
re-encoding the whole source. While any job runs, the service also
self-pings its public `/healthz` URL every few minutes so Render's
free-plan 15-minute idle spin-down cannot kill a long job mid-render.
Every Gemini call (script, TTS, upload) is hard time-boxed
(`GEMINI_GENERATE_TIMEOUT_SECONDS`, default 240s per model attempt)
with elapsed-time heartbeat progress, so a single hanging request can
no longer freeze a job at "retrying script (3/3)"; if every model
still returns nothing usable the job fails fast with a clear message
instead of re-uploading and repeating the same calls. Overload-class
failures (429 / 503 "high demand") are waited out instead of burned
through in seconds: whole model rounds repeat with very short
backoff (6s → 10s, probing often so the job resumes as soon as
capacity returns, but giving up clearly instead of making the user
wait) inside a bounded budget
(`GEMINI_OVERLOAD_RETRY_BUDGET_SECONDS`, default 120s, max 12 rounds)
with visible countdown progress, a `gemini-3.5-flash-lite` capacity
fallback runs last in each round, and TTS batches likewise wait
longer between overload retries. Models that return 404 NOT_FOUND
(e.g. retired `gemini-2.5-flash-lite`) are remembered process-wide
and skipped from then on, and the reported error prefers the
overload cause over a trailing model's 404. Script generation races
models for the fastest answer: on calls that carry the video, the
first `GEMINI_HEDGE_INITIAL_PARALLEL` (2) models start together;
any model still unanswered after `GEMINI_HEDGE_DELAY_SECONDS` (15s)
gets the next model started in parallel, and the first successful
answer wins (text-only calls start single to save quota), so one
slow/queued request cannot stall the cascade. The script stage is also leaner:
the first prompt asks for ~15% more text than the target (models
undershoot counts; overshoot is trimmed locally for free), the
second full-script retry is skipped when the first call itself
failed on every model, first-person repairs are text-only (no video
re-processing), and two TTS batches run concurrently. The default TTS
voice is the original **Charon** with the Deep pitch-shift filter off
by default and a neutral reading style (no "deep/slow" instructions).
TTS requests use the current `voice_config.voice` shape (alternating
with the legacy `prebuilt_voice_config` shape per attempt, since
model generations accept different shapes), send plain text only
(current models recite verbatim -- instructions would be read
aloud), scan all response parts for the audio (WAV responses are
detected and decoded as WAV, legacy headerless PCM as raw s16le),
and fall back across `gemini-3.8-flash-tts`,
`gemini-3.8-flash-lite-tts`, `gemini-3.1-flash-tts-preview`,
`gemini-2.5-flash-tts`, `gemini-2.5-pro-tts` and
`gemini-2.5-flash-preview-tts`. Models that answer 404 are
remembered and skipped process-wide; models that answer a
plan-quota 429 ("exceeded your current quota") are remembered for
`GEMINI_QUOTA_SKIP_SECONDS` (30 min) and skipped without retry
storms. The latest TTS error per model is remembered, appended to
silent-scene failures, and exposed on `/healthz` as
`last_tts_error`, `last_tts_error_by_model` and
`quota_exhausted_models`. TTS providers run in `TTS_PROVIDER_ORDER`
(default `edge,gemini` -- completely free first, no quota queue):
Microsoft Edge's Burmese neural voice (`my-MM-ThihaNeural` —
always Thiha, one consistent narration voice) synthesizes each
*scene* separately in seconds with no
plan quota -- per-scene requests mean every clip is exactly its
own scene's audio (joined-batch slicing guesses were heard as
doubled/cut speech), and one flaky scene degrades to a short
silence instead of sinking the batch. Each synthesized clip is
validated (silent or grossly truncated results are retried, not
shipped), its encoder/service dead air is trimmed with
click-free micro-fades, and loudness is normalized once over
the assembled narration -- per-scene files joined raw with
per-clip loudnorm made the speech stop-start with jumping
loudness. Final part joins re-encode audio so AAC priming
cannot leave a gap at each seam. Scripts are prompted in
spoken Burmese register (စကားပြောဟန်, short sentences -- written
"book" Burmese made the narration sound read-aloud; key dialogue
lines are embedded as direct speech inside the first-person
retelling, closer to popular recap-channel storytelling), and Gemini
Charon is the fallback if Edge ever fails. (In production
the free-tier key's Gemini TTS plan quota was exhausted, which is
what put Edge first.) Set `TTS_PROVIDER_ORDER=gemini,edge` to put
Charon first again, or `EDGE_TTS_FALLBACK_ENABLED=0` to disable
Edge entirely. `/healthz` shows `tts_provider_order`,
`last_tts_provider`, `audio_pipeline`, and
`last_tts_silenced_scenes` (scenes replaced by silence in the
most recent synthesis -- nonzero after a job means narration
went missing somewhere).
The free Edge voice is the only working TTS in production (the free
Gemini TTS plan quota is exhausted), so its outages used to kill jobs.
A Microsoft-side blip made 3 of 5 scenes fail within seconds and the part
gave up: `Gemini TTS produced too many silent scenes: 3/5` — a confusing
message, because the failing engine was Edge and the Gemini quota error
was only stale. v5.16 makes that path resilient and honest:

- every scene gets **4 spaced attempts** (0.6s → 1.5s → 3s) instead of
  three instant retries;
- a scene that still fails is **re-requested sentence by sentence** and
  the pieces are rejoined, so a long failing request does not lose that
  scene's content;
- if the first scenes of a batch fail with **zero successes**, the batch
  stops early (Edge is down, not one request) and the next provider — or
  the retry pass — takes over in seconds instead of grinding through
  every scene;
- `synthesize_tts` then re-synthesizes **every leftover silent scene in a
  fresh pass** (2s later) before anything is accepted, and only genuinely
  unrecoverable scenes stay silent;
- one Edge request can no longer hang for minutes: Edge uses its own short
  timeout (`EDGE_TTS_TIMEOUT_SECONDS`, default 45s) instead of the 300s
  Gemini TTS timeout;
- the failure that reaches the user is now Burmese and names the real
  cause (`အသံသွင်းမှု မအောင်မြင်ပါ — scene 3/5 ခုမှာ အသံ မထွက်ပါ …`) with the
  Edge error in the suffix, and `/healthz` exposes
  `last_tts_retry_recovered_scenes` plus `tts_resilience` so flakiness is
  visible even when jobs succeed.

Script length is planned from the voice's **measured** speaking rate.
Earlier releases sized every script with a hard-coded 6.5 chars/second,
which matches the Gemini voices but not the free Microsoft Edge Burmese
neural voices that actually speak ~2.1-2.5 chars/second. Every script
therefore came out ~3x too long — a 2-minute window that should target
112 seconds of narration planned 702 characters and measured 341
seconds — and the duration calibration then deleted two thirds of it,
which was heard as broken, chopped speech ("အသံအမှန် 05:41 ရှည်နေလို့
ပစ်မှတ် 01:52 ဖြစ်အောင် ချိန်နေပါသည်"). Scripts are now sized with a
**learned rate**: the bot keeps a process-global chars/second per TTS
provider (`edge`, `gemini`), seeded at 12 for the Edge voice (production
measures `my-MM-ThihaNeural` at ~14-17 chars/second: a 313-char scene
took 18.65s) and 10 for Gemini. The first real measurement replaces the
seed outright; later measurements blend in (exponential moving average,
clamped to 1.2-24.0 chars/second) from `_narration_measurements`'
`effective_chars_per_second` after every measured window. The learned
rate drives every `target_chars` derivation — the direct per-window
script writer, its top-up prompt, and the n8n worker rewrite — and is
exposed on `/healthz` as `script_length_rate`. Trimming is now
**sentence-boundary only**: when measured narration is still too long,
whole trailing sentences are removed (scenes whose only sentence cannot
be split are dropped whole rather than chopped), so the TTS never reads
half a clause. The 90%-of-source final-duration gates and the
per-scene first-person gates are unchanged: the target is reached by
writing the right amount of text, not by shredding it.

It
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
3. **Script everything first**: write every window's first-person
   full-retelling script in order (character names,
   protagonist/pronoun, prior-events summary, and last scene carried
   forward after every window) *before any audio exists*, then check
   the joined script once as a whole (sentences repeated across
   parts are removed, keeping the first telling). Only then voice
   and edit each window from its finished script: speech-shaped
   polish (breath points for runaway sentences), a pre-TTS cap so a
   script cannot predict past its window, measured TTS with error
   checks, and a render targeted at the selected percentage of that
   window. Each phase of each window gets bounded whole-segment
   retries (`AUTO_SEGMENT_ATTEMPTS`, default 3) on top of the
   script/TTS internal retries. Progress shows window x/y, time
   range, and percent for both phases.
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

Character counts are only the first estimate — the pre-TTS char
gate is a coarse anti-summary floor (85% of the estimate), not the
duration guarantee — but that estimate is no longer a guess: scripts are
sized with the learned chars/second of the voice that will speak them
(free Edge `my-MM-ThihaNeural`: ~14-17 measured, cold-start seed 12), so
a 2-minute window at 90% plans about 1,300 characters — close to the
voice's real speed from the first draft. After TTS, the bot
measures the real narration seconds and this job's real chars/second. If the
narration is short, Gemini expands the full chronological script and the
changed scenes are synthesized again; if it is long *but still inside
the source's own length*, the finished speech is kept whole — spoken
narration is never trimmed back to the 90% target or force-fitted. The
one hard ceiling is the source itself (**the final must never be
longer than the original**): narration that runs past its window has
only that excess shaved at whole-sentence boundaries (target just
under the window) and the changed scenes are re-voiced. A result that
stays short is refused rather than shipped as an under-length recap,
and every delivery gate refuses a final that exceeds the source. Calibration targets are
computed from the *effective* rate (all script chars over all produced
speech seconds, including any quiet slices) so trims always converge --
the spoken-only rate inflates when some clips measure as silent and
once stalled calibration entirely. This repeats for up to four
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
13. 🗣️ Script အလျင် ချိန်ညှိ — script chars/sec. The shipped 6.5 default now means **auto**: the bot learns the real rate of the voice in use (free Edge `my-MM-ThihaNeural` ~14-17 measured) from measured TTS and sizes every script with it; choosing 5.5 / 7.5 is an explicit manual override
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

## v5.17 — when Gemini answers but the answer has no scenes

A production job died with
`GeminiScriptError … သုံးနိုင်သော scene မရပါ … Gemini retry response also
contained no usable Burmese scenes`. That message blamed the Gemini quota,
but both calls had **answered and parsed as JSON** — the payload simply
carried nothing the normalizer can use (empty `scenes`, text under an
unknown key, a truncated object, or a safety-shortened reply). Nothing
about it was debuggable: the diagnostics were discarded and only a generic
sentence reached the user.

v5.17 changes three things:

- **The budget can no longer be eaten silently.** Script calls ask for
  `SCRIPT_JSON_MAX_TOKENS` (16384, was 8192) output tokens, so a reasoning
  model cannot spend the whole allowance on hidden thinking and return an
  empty or half-written JSON object.
- **Two extra salvages before the job fails.** First the longest Burmese
  string inside the JSON answer already received is turned into timed
  scenes; then one more call asks for a **plain spoken retelling with no
  JSON envelope**, which is split on sentence boundaries and laid out over
  the window locally (`_scenes_from_plain_narration`). Only the model's own
  words are used and only Burmese sentences survive, so nothing is
  invented. A call that never answered at all (400/429/503) still fails
  fast instead of repeating the same wait.
- **The failure is honest and inspectable.** `last_script_error`,
  `last_script_raw_preview` (first 1200 characters of what Gemini actually
  returned), `last_script_error_by_model`, and `last_script_recovery`
  (`plain_text_retelling`, `prose_salvaged_from_json_answer`, or empty)
  are on `/health`, and the Telegram error carries the diagnostics plus
  the start of the reply — so the next report is actionable instead of
  guesswork. The message no longer claims the quota is exhausted when the
  models clearly answered.

## v5.18 — measured-duration calibration that can actually converge

A job died with

`Measured TTS duration is outside the final-duration target after
calibration: 124.2s actual, target 113.0s (acceptable 110.5-115.5s from
125.5s input); refusing to render a video with the wrong duration`

after **three** full attempts. The narration was only ~10% too long, and
the script *was* being trimmed — but the trim reasoned in **characters**
while the only boundary it allowed was a whole sentence. One Burmese
sentence is 10-15 seconds of speech, so each "improvement" jumped the
narration from ~124s to ~100s and back: the ±2% window (2.3s) was far
narrower than the only control the pipeline had. The gate then refused
the video instead of finishing it.

v5.18 fixes the control loop, not the gate:

- **Trim by predicted duration, not by characters.** The character budget
  is derived from *this job's measured chars/second*, so removing the
  right number of characters becomes removing the right number of
  seconds (`_trim_scenes_for_duration`).
- **Two-phase trimming.** Whole trailing sentences (or whole
  single-sentence scenes) go first, and the last few characters — the
  part that actually decides whether the ±3% window is met — are cut from
  the tail of the last scenes at a **word boundary**. A cut never lands
  inside a word, never leaves a dangling Burmese combining mark (which
  the TTS reads as a broken syllable), and always ends on `။`.
- **The window is ±3% of the source** (was ±2%). Word-level cutting moves
  in ~1-3s steps, so ±2% (2.3s on a 2-minute window) was tighter than the
  mechanism could physically hit; ±3% still pins the final video to the
  requested 90% within a second or two.
- **Nothing is ever stretched or padded.** Only text the model wrote is
  removed, in order, from the end.

The same calibration (and therefore the same failure) is visible on
`/health` as `last_duration_calibration` — target, window, and every
round's measured seconds/chars/status — and the terminal error now lists
those rounds instead of a bare "refusing to render".

## v5.19 — a job can no longer die silently

Production report: the progress message froze at "video ဖန်တီးနေပါသည်",
then 📊 အလုပ်အခြေအနေ answered **"⚪ လက်ရှိ လုပ်နေတာ မရှိပါ။"** — as if the
recap had never existed — and no error ever arrived.

Cause: jobs live **only in process memory** (`ACTIVE_JOBS`), and Render
restarts the process on every deploy/health recycle. A restart therefore
erased a running job instantly: the progress message stopped updating,
`/status` had nothing to report, and the chat never learned that its
video was gone. Nothing was even broken in the pipeline — the process had
simply been replaced mid-render.

v5.19 makes the job registry survive that:

- **Every job is mirrored to disk** (`active_jobs.json` in `WORK_ROOT`,
  written atomically and throttled): `job_id`, current stage, started /
  last-progress timestamps, plus the last outcome per chat. A progress
  update heartbeats the registry even when Telegram throttles the edit.
- **Startup recovery**: the next process reads the mirror, marks each
  unfinished job `interrupted`, records it in `interrupted_jobs_last_restart`
  on `/health`, and sends that chat a message — "🔁 Server ပြန် restart
  ဖြစ်လို့ ဒီ job ရပ်သွားပါတယ် / ⏱ ရပ်သွားတဲ့အဆင့်: … / video/link ကို
  ပြန်ပို့ပေးပါ" — instead of losing it.
- **Stall watchdog** (every 60s, `JOB_STALL_SECONDS` = 1200s): a job that
  stops reporting is cancelled, its slot released, and the chat told
  which stage it stalled at. A dead job can no longer block the chat
  forever.
- **The status screen is honest**: while a job runs it shows the live
  stage and elapsed time; when idle it shows the previous job's outcome
  (✅ done / 🛑 cancelled / 🔁 interrupted by restart / ❌ failed with the
  reason) instead of a bare "nothing is running".

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
URLs must show `"version":"v5.22-script-first-no-overshoot"`,
`"gemini_hedge_delay_seconds":15`, `"gemini_hedge_initial_parallel":2`,
`"gemini_generate_timeout_seconds":240`, `"gemini_tts_timeout_seconds":300`,
`"gemini_overload_retry_budget_seconds":120`, `"gemini_overload_max_rounds":12`,
`"default_voice":"Charon"`, `"default_deep_voice_filter":false`,
`"script_trim_policy":"sentence_then_word_tail_boundaries"`,
`"duration_calibration":"measured_rate_budget_sentences_then_word_tails"`,
`"last_duration_calibration":{}`,
`"job_registry":"disk_mirrored_active_jobs_with_stall_watchdog"`,
`"active_jobs":{}`, `"last_job_results":{}`,
`"interrupted_jobs_last_restart":[]`,
`"job_watchdog":{"enabled":true,"stall_seconds":1200,"interval_seconds":60}`,
`"tts_resilience":"per_scene_retries_then_sentence_pieces_then_fresh_pass"`,
`"last_tts_retry_recovered_scenes":0`, `"edge_tts_timeout_seconds":45`,
`"script_plain_text_fallback":"plain_text_retelling_split_locally_then_prose_salvage"`,
`"last_script_error":""`, `"last_script_raw_preview":""`,
`"last_script_recovery":""`, a `script_length_rate`
object (`"active_provider":"edge"`, `"learned_chars_per_second"` 12.0
on a fresh process, `"clamp":[1.2,24.0]`,
`"learning_source":"measured_tts_effective_chars_per_second"`),
`"recap_flow":"auto_concat"`, `"manual_parts_enabled":false`,
`"telegram_pipeline":"direct_auto"`, `"n8n_telegram_handoff_enabled":false`,
`"source_segment_seconds":120`,
`"source_split_acceptance":"stream_copy_keyframe_snapped_max_1p5x_window_then_720p_ultrafast_fallback"`,
`"render_keepalive":"self_ping_public_healthz_while_jobs_active"`,
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

After every deploy the bot tells its users it was updated: each chat
gets one short "✅ Bot ကို အသစ်ပြင်ပြီးပါပြီ" notice per version on its
first message after an update. For a *proactive* post-deploy DM (before
you open Telegram), set `OWNER_CHAT_ID` once in Render's Environment —
`/start` shows your Chat ID to copy. (Render wipes the disk on deploy,
so an env var is the only address book that survives one.)

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
