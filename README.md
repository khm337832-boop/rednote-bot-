# rednote-bot-

Telegram bot that accepts a directly uploaded Telegram video/document
video or a RedNote/video link and turns it into a Burmese recap video
(Gemini script → Edge Thiha TTS, Gemini Charon fallback → ffmpeg render), controlled with
**inline buttons, a persistent bottom keyboard, and Telegram's command
menu button** — no commands to memorize.

Version `v5.40-opening-introduction-repair` is a **one-shot automatic flow**: the user
sends one video or link and the bot does everything else. It downloads
the complete source, processes it internally in fixed chronological
**2-minute windows** (a 1:30 source is 45 windows; a 3-hour source is
90), joins every recap in order, verifies the final duration, and sends
the finished final video — **no per-part clicking**. Transient failures
are retried automatically at every stage (source download, script, TTS,
per-window render, join, and final upload), and a job is only reported
as done after every final piece has actually been delivered. The
announced source-window plan and the processed window count are kept
identical: stream-copy may use keyframe-snapped boundaries only when
it returns exactly the planned count, covers the full source, and no
window exceeds 1.5× the target; otherwise the bounded 720p/ultrafast
accurate extraction builds the planned windows. The joined final is
split only when Telegram's upload size requires it, and that separate
size-based delivery set is validated before the first upload (every
file fits, probes as video, and the file durations together still
equal the joined final), so an internal 11-window job can be sent as
9 larger Telegram files without losing two windows—and the captions
now label those as Telegram delivery files rather than source parts. While any job runs, the service also
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
failed on every model, first-person protagonist narrator repairs are text-only (no video
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
"book" Burmese made the narration sound read-aloud; the protagonist
narrates in first person with one locked self-pronoun, introduces
themself by full spoken name and role at the opening, and every other
speaker/listener is named for reported dialogue while any short direct
quote stays clearly quoted, matching lived-story recap-channel
storytelling rather than a report), and Gemini
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
provider (`edge`, `gemini`), seeded at 18 for the Edge voice (production
repeatedly measures `my-MM-ThihaNeural` at ~17.8-18.6 chars/second)
and 10 for Gemini. The first real measurement replaces the
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
per-scene first-person protagonist narrator gates remain in force: the target is reached by
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
3. **Script everything first**: write every window's first-person-protagonist
   full-retelling script in order *before any audio exists*. Each
   window first builds a `story_map`, and a cumulative **story
   bible** is carried forward — a Cast Lock Name Ladder with one full
   spoken name or memorable stable descriptive label per character,
   plus an optional source-grounded everyday `short_name`, exact
   `name_evidence_text`, an optional one-time `name_reveal_bridge`,
   role, relation, a Burmese first-mention `introduction`, an
   `identity_anchor` for returns, current `goal`/`motivation`, visual
   cue and aliases;
   protagonist identity; ordered beats with trigger, dialogue,
   reaction and result parts; current goal/conflict, what is at stake
   if the characters fail or refuse (`conflict_stakes`), the still-open
   `central_question`, setting, unresolved threads, important facts,
   summary, and last scene. The joined script is then checked as a
   whole (sentences repeated across parts are removed, keeping the
   first telling). If local clarity signals say a draft reads like
   disconnected captions (thin beats, weak transitions/causal links,
   duplicated scenes, missing first introductions, bare generic
   labels, planned dialogue/reactions/stakes that never reach the
   narration, or a final window with no unresolved-question close), a
   bounded text-only **story-editor** pass may rewrite that window without
   moving events across scene boundaries; it is accepted only if
   scene count/order, first-person protagonist narration, Cast Lock identities,
   length, and clarity hold up. Only then voice
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
   size delivery, validate that every delivery file fits/probes and
   that their durations sum to the joined final, and only then send
   the final video. Progress and captions distinguish internal
   windows (`x/y` windows joined) from size-based Telegram delivery
   files (`x/y` files sent). Uploads are streamed from disk and honour
   Telegram `retry_after`; every final piece is retried up to 3 times,
   and if any piece cannot be delivered the job is reported as failed
   — never as a partially delivered success.

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

## Final-duration and first-person protagonist Cast Lock guarantees

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
(free Edge `my-MM-ThihaNeural`: ~17.8-18.6 measured,
cold-start seed 18), so a 2-minute window at 90% plans about 1,950
characters — close to the voice's real speed from the first draft. After TTS, the bot
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

Edit sync is treated as a first-class concern, not an afterthought:
scripts are written in finer scenes (~8 seconds each) with the text
amount proportional to each scene's on-screen span. Each scene is
tied to its ordered story-map beat, starts from the previous
scene's result, and gives the cause before (or with) the reaction
instead of front-loading a later reveal onto earlier footage.
Model timestamps are repaired so no scene can end up with zero
footage, and at render time the scene boundaries are placed
*proportionally to the measured narration durations* (each scene
gets about as much footage as its speech needs, pulled toward the
model's timestamps within a few seconds of drift and snapped to
real detected shot cuts). When narration modestly outruns footage
the *footage is slowed down* (never the audio) instead of freezing
on a single frame. When the shortage is severe enough that one
slow pass would itself look frozen (narration/footage ratio over
1.5), that same scene is replayed from its own start for a second
pass — **two total showings, never three** — so both passes can
move at a near-natural speed. Before those passes are planned, the
preferred boundaries go through a minimum-span repair: if clustered
model timestamps would leave a scene with less unique footage than
two moving passes need, only the affected starts move enough to
reserve that span, so normal plans need no frozen-frame tail at
all. The segment renders
pass their length as an ffmpeg *input* option and the concatenated
video track is length-checked against the narration before muxing:
an output-side `-t` used to silently truncate every slowed (or
frozen) segment back to its input length, so the video track ended
early while container duration probes (which report the audio)
still passed — the root cause of two releases of "the video froze
but the audio kept going" reports.

The script also follows professional movie-recap structure: the
protagonist opens by introducing themself and fixing time, place, the
key relationship, and the immediate problem within the first
sentences; beats are told as previous result -> what I or the named
actor wants -> obstacle/spoken demand -> action -> the other person's
reaction or decision -> consequence/new problem (so schemes and
decisions are understood *while* they happen, not explained
afterwards); explicitly named place/time/flashback and
parallel-storyline transitions; every other character is introduced by
one locked Burmese spoken name or stable descriptive label plus
role/relation on first appearance; and scene endings land on the
visible result or new danger without inventing future events.

Narration text is written to be *spoken and followed by ear alone*:
everyday colloquial Burmese storytelling rather than book/report
prose — mostly one action plus its reason or result per sentence,
varied sentence lengths, natural connectors, and occasional short
impact sentences. The protagonist uses the locked “I” after the
opening self-introduction; every other actor is named whenever the
subject changes or a pronoun could confuse. Every other important
person receives a first-mention introduction (confirmed name + role +
relation/group/place, or one memorable descriptive label when no name
is supported). Important demands/questions/decisions are rendered as
speaker → listener answer/refusal/question → pressure/explanation →
decision/result, with short direct quotes only when the source meaning
is reliable and precise reported speech otherwise. Reactions carry
their visible or spoken reasons, the central conflict's
source-supported reason and stakes stay explicit, exact and
near-duplicate retellings are removed, and each scene must advance
who is involved, what changed, why it matters, and what problem
follows. Only the source's final window may close by restating its
genuinely unresolved decision/danger as a question; future events are
never predicted — no `…ပါသည်/…သည်` bookish endings or report
nominalizations, and no Latin/foreign script or digits (foreign words
are transliterated into Burmese sound and numbers are spelled out as
Burmese words — enforced deterministically before synthesis), so the
voice never stumbles over unreadable tokens and a listener who never
sees the picture can still retell who did/said what to whom, why, and
what happened next.

The protagonist is the narrator and tells the story in first person,
exactly like a lived-story recap. At the opening they introduce
themself once with their full spoken name (or locked stable label),
identity/role, the key relationship/other central person, the
time/place/current situation, and the inciting problem; later windows
resume from the previous result without mechanically repeating that
introduction. Their own actions, source-supported thoughts, and
reactions then use one source-supported locked pronoun —
`ကျွန်တော်` or `ကျွန်မ`, including its possessive variants — while the
wrong-gender pronoun, another character becoming an unquoted “I”, or
the protagonist repeatedly calling themself by name as a third-person
subject is rejected. Other characters may address the protagonist by
name/title in dialogue, and every other important person still follows
the Cast Lock Name Ladder. A protagonist-absent cutaway must open with
the named actor/place and a bridge back to the protagonist's story.
Quote-aware validation and bounded text-only repair run on direct,
worker, rewrite, top-up, calibration, render, and delivery paths; a
protagonist-present script with no established first-person voice, a
mixed/wrong pronoun, or a missing opening self-introduction is refused
rather than shipped.

Cast Lock name evidence is ranked so aliases stop drifting between
windows. Before a descriptive label is accepted, the whole window is
searched for names (readable burned-in subtitles/captions, on-screen
name cards, readable text, self-introductions, and characters clearly
addressing each other); a name revealed later in the same window is
used from the person's first mention unless the source deliberately
withholds it as a mystery. A readable burned-in subtitle, caption, or
on-screen name card is strongest; a clearly confirmed name in dialogue
comes next; weaker or conflicting later guesses become non-spoken
aliases instead of replacing the established identity. If no reliable
exact name exists, the label must still be memorable and unique —
place/group + role + a visible anchor, never a bare category such as
“unknown youth”, “woman”, or “old farmer” — and that one form is locked
for the whole video. Each card carries the canonical Burmese
`spoken_name`, the original `source_name`, role, relation, the exact
Burmese `introduction` to speak at first mention, current
`goal`/`motivation`, visual cue, aliases, `name_source`, and
`confidence`. Cards also retain the re-entry `identity_anchor`, an
optional source-grounded routine `short_name`, exact
`name_evidence_text`, and an optional one-time `name_reveal_bridge`
when an earlier label is later tied to a confirmed name. The
protagonist is the one deliberate exception to routine name
repetition: after the one full self-introduction, they normally call
themself by the locked pronoun, while other people may address them by
name/title in dialogue. Every other character uses the locked Name
Ladder — full spoken name in the first detailed introduction, then
only the approved short form (or the full name when none exists), then
the identity anchor on return; speaker and listener are named for
dialogue beats, and a returning or newly acting character is
re-identified instead of being left in an ambiguous `သူ`/`သူမ` chain.
Exact names still cannot be guaranteed when the source contains no
readable name evidence, but identity remains stable.

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
The measured 90%-duration and quote-aware per-scene first-person protagonist gates
remain hard requirements.

## The 19 features (⚙️ ဆက်တင်များ)

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
11. 📦 မူရင်းပိုင်းချိန် — source is always planned into fixed 2-minute windows and extracted one at a time
12. 🗣️ Script အလျင် ချိန်ညှိ — script chars/sec. The shipped 6.5 default now means **auto**: the bot learns the real rate of the voice in use (free Edge `my-MM-ThihaNeural` ~17.8-18.6 measured) from measured TTS and sizes every script with it; choosing 12 / 17 is an explicit manual override (the retired 5.5 / 7.5 UI values are migrated back to auto for real jobs)
13. 👤 ပြောပြသူ ရှုထောင့် — the protagonist is always the first-person narrator with a source-supported locked `ကျွန်တော်`/`ကျွန်မ`; the protagonist self-introduces once by full spoken name/role, then uses that pronoun for their own actions, while the Cast Lock Name Ladder fixes one full Burmese name/stable label plus any source-grounded routine short form for every other character, every scene is quote-aware validated/repaired, and the script retells the whole video in order, scene by scene, no summarizing
14. 🎭 ဇာတ်ကြောင်း လေသံ — natural / dramatic / emotional / calm tone
15. 📤 Telegram ခွဲပို့ — split files over 45MB on/off
16. 💬 Video Caption on/off
17. 🔔 ပြီး အကြောင်းကြား — completion notification on/off
18. ⚡ မြန်ဆန်မုဒ် — faster render preset on/off
19. 🚀 Video/Link တန်းစ — auto-start on an uploaded video or link, or ask for ✅ confirmation first

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

## v5.30 — Cast Lock integrity, continuity QA, and display geometry

- **Ambiguous aliases can no longer merge two people.** If a model
  assigns the same nickname/kinship alias to two different confirmed
  `spoken_name` cards, that alias is removed as identity evidence;
  independently confirmed people with conflicting visual cues also stay
  separate even when they share a source name. A descriptive label can
  still be upgraded to its subtitle/on-screen confirmed name.
- **Script QA uses the cumulative story bible.** The clarity gate now
  checks the current window against Cast Lock cards from earlier
  windows, flags scenes whose *opening sentence* leaves the actor as
  an unexplained `သူ/သူမ`, and routes Latin/CJK leftovers to the
  text-only story editor so names/terms are Burmese and TTS-safe before
  voicing. An editor candidate containing any such script is rejected.
- **Long-video memory prioritizes the active cast.** Context merging
  keeps the protagonist first, then the latest window's cast and
  unresolved threads, instead of letting early walk-on characters fill
  every capped slot on a 90-minute source. Prompts now receive all 16
  retained Cast Lock cards.
- **Display geometry is normalized.** Probing derives display size
  from SAR/DAR and a 90/270-degree display matrix, and rendering emits
  square pixels (`setsar=1`) at that display aspect instead of sizing
  from coded anamorphic pixels.
- **Operational honesty fixes.** The working `gemini-3.5-flash-lite`
  now precedes the production-404 `gemini-2.5-flash`; worker Gemini
  calls use the same wrapped/truncated-JSON recovery as direct calls;
  the unused “cut %” setting and model `cuts` request were removed;
  stale manual-rate values migrate to measured auto; and the n8n docs
  then described that release's third-person Cast Lock contract
  (superseded by v5.35's first-person protagonist contract).

## v5.31 — Who is who, conflict stakes, dialogue life, and ending hooks

- **Names are hunted before labels are accepted.** The script prompt
  must scan the whole window for subtitle/name-card/readable-text and
  clearly addressed dialogue names, then introduce every important
  person or group at first mention with name + role +
  relation/group/place. When no exact name is supported, it must use
  one memorable place/group + role + visible-anchor label instead of a
  bare category (“unknown youth”, “woman”, “old farmer”). Character
  cards now retain that exact introduction plus each person's current
  goal and motivation across windows.
- **The story map explains the conflict, not only the event.** Beats
  can carry `trigger`, `dialogue`, `reaction`, and `result`; the map
  also retains `conflict_stakes` (what refusal/failure would cost) and
  `central_question`. Direct, retry, plain-text, top-up, summary,
  editor, and worker prompts all require source-supported demand
  reasons/stakes and reaction reasons without inventing motives,
  consequences, dialogue, or inner thoughts the source does not show.
- **Important dialogue is dramatized as an exchange.** Demands,
  questions, warnings, negotiations, and decisions should flow
  speaker → listener response → reaction/decision, mostly as attributed
  reported speech with only short decisive source lines quoted. Event
  blocks must open with a causal “previous result → new need/decision →
  next action” bridge.
- **The local QA gate can send flat drafts to the editor.** It now
  detects missing first introductions, generic cast labels, planned
  dialogue with no exchange in the narration, thin reaction/motivation
  signals, mapped stakes that are never spoken, and — only when the
  caller marks the source's final window — an ending that ignores the
  still-unresolved question. The final-window flag is threaded through
  automatic, legacy combined, manual, and worker rewrite paths.

## v5.32 — Exact window counts and delivery integrity

- **Announced source windows now equal processed source windows.**
  The one-pass stream-copy split is accepted only when it covers the
  full source, no window exceeds 1.5× the target, and its file count
  exactly matches the announced 2-minute plan. A sparse-keyframe
  result such as “11 planned, 9 returned” now falls back to accurate
  planned extraction instead of silently processing longer windows.
- **Telegram delivery is validated before upload 1.** Internal
  windows are joined and then split only by the 45MB Telegram cap;
  those delivery files need not equal the internal window count.
  Before sending, every file must exist, fit, and probe as video,
  and their durations together must match the joined final. A bad
  split fails before any piece is uploaded instead of stopping after
  a partial prefix.
- **Sparse-keyframe final splits have an accurate fallback.** If a
  stream-copy split cannot divide an oversized delivery part (or its
  children do not cover the parent), planned-range extraction creates
  standalone children; stale segment files are cleared and numeric
  ordering is enforced.
- **Large uploads are streamed and flood-aware.** Telegram media is
  no longer copied whole into RAM for each attempt, and 429
  `retry_after` waits are honoured. Direct delivery progress/captions
  show both counts separately: internal windows joined and Telegram
  files sent.

## v5.33 — Natural names, glossary lock, and visible-text meanings

- **Detailed first introduction, natural everyday name after that.**
  Character cards keep one concise TTS-speakable `spoken_name`, the
  detailed first-use `introduction`, and a short `identity_anchor`
  (name + role/relationship) for re-entry after an absence or an actor
  switch. Prompts forbid repeating a long descriptive identity phrase
  as though it were a natural name and forbid alternating spellings or
  transliterations.
- **Story terms are locked and explained once.** Story maps and the
  cumulative context now retain `terms`: source term, one canonical
  Burmese `spoken_term`, the source-supported meaning/function in this
  story, an exact `first_explanation`, category/evidence, and aliases.
  Concepts such as System, rank/quality, Mortal, skills, levels,
  titles, organizations, places, and special items must be explained
  in plain Burmese at first use. If the source has not defined a term,
  prompts require describing only its visible function/effect instead
  of inventing a genre definition.
- **Important visible text is accounted for.** Story maps retain
  `visible_texts` for name cards, system/status panels, signs,
  documents, title cards, maps, and messages: timing, original text,
  kind, related character, and the Burmese meaning to narrate over the
  matching footage. Beats can also carry `visible_text` and
  `why_it_matters`; prompts require reconciling dialogue/audio,
  subtitles, and visible text, and forbid guessing unreadable wording.
- **The local QA gate routes comprehension gaps to the editor.** It
  detects returning people who are not re-identified, new glossary
  terms used without a first-use explanation, inconsistent term
  variants, important visible-text meanings missing from narration,
  repeated actor/group switches without a causal bridge, and mapped
  beat stakes that are never spoken.

## v5.34 — Name Ladder: natural, memorable names without identity guesses

- **Four-stage naming.** A person is first introduced with the full
  locked `spoken_name`, role, relationship/group, and—when useful—one
  visible memory cue. Routine references may then use an approved
  `short_name` only when the source itself uses that shorter address
  form, or when it is the compact unique core of a descriptive label.
  The code never invents a nickname, never shortens a foreign name
  arbitrarily, and a short form shared by two people is withdrawn
  rather than risking a merge. Long descriptive facts stay in the
  introduction/`identity_anchor` instead of being repeated as the
  subject of every sentence. On return after an absence, window
  boundary, or actor switch, narration re-identifies the person with
  the identity anchor.
- **Name evidence and reveal bridges are stored, not guessed.**
  Character cards retain `name_evidence_text` from the subtitle, name
  card, caption, or dialogue that supported a name. Important
  `name_card` visible-text entries are cross-linked to a card, can
  upgrade it to on-screen/confirmed evidence, and can create a minimal
  card when the person exists only in the visible-text ledger. When a
  descriptive label later becomes a confirmed real name, old forms
  remain non-spoken aliases and a one-time `name_reveal_bridge` tells
  the listener that the old label and new name are the same person.
- **Cast Lock protects approved full and short forms.** The local
  Cast Lock pass treats the full `spoken_name` and one approved
  `short_name` as the same identity, while still normalizing unknown
  aliases/variants only when they point to exactly one person. The
  Name Ladder fields are carried through cumulative context, all
  direct/editor/summary/worker/top-up card contracts, and manual
  recovery payloads.
- **Sentence-level identity QA.** The clarity gate now routes drafts
  to the bounded story editor when a short form appears before its
  full introduction, a long descriptive label is repeated like a
  natural name, a mapped current-window reveal bridge is missing, an
  ambiguous `သူ/သူမ` action follows two or more named people, or an
  important dialogue/reported-speech sentence lacks a nearby named
  speaker. These heuristics are conservative editor signals, not a
  reason to invent names or story facts.

## v5.40 — Opening introduction repair

Card introduction text is reused only when it actually passes the same quote-aware protagonist self-introduction check as the final gate and is TTS-safe. Otherwise a safe locked identity gets a deterministic introduction; foreign evidence is never spoken as a fallback. Final direct quality metrics use the same locked protagonist/pronoun as the repair pass, and opening diagnostics are exposed for investigation. This does not bypass the opening or professional gates.

## v5.39 — Foreign-name token lock and actor-targeted repair

- **Foreign card names are repaired at token level.** The first live
  v5.38 retry generated Chinese character cards themselves
  (`董礼茂`, `杨伟`, `乔碧落`, `九州学院长老`), leaving the deterministic
  ledger no Burmese target and making whole-scene repair copy the
  Chinese back. Direct and worker repair now asks the model only for
  a validated source-token → Burmese mapping, then applies it
  deterministically so scene content, POV, order, timestamps, and
  length cannot drift. Proper names/places/groups are transliterated;
  common terms, titles, skills, and phrases use a concise Burmese
  meaning. Every accepted target must contain Myanmar script and no
  TTS-unsafe letters.
- **Successful mappings become the new Cast/Glossary Lock.** The
  foreign source spelling is retained only as source/alias evidence;
  the Burmese result replaces unsafe card `spoken_name`/short forms,
  term `spoken_term`, protagonist identity, and context character
  lists. Later scenes/windows and quality metrics therefore use the
  speakable Burmese identity instead of re-learning the foreign one.
- **Simplified/Traditional variants share one lock.** Ledger and
  token replacements expand common CJK spelling variants (for
  example `杨伟`/`楊偉` and `九州学院长老`/`九州學院長老`) to the same
  Burmese target. Foreign `spoken_name`/`spoken_term` values in
  generation, editor, and repair prompts are explicitly marked as
  evidence—not locked spoken forms—to stop weaker fallback models
  preserving them verbatim.
- **Severe actor/dialogue blockers get their own repair target.**
  Bounded direct/worker repair now includes scenes behind severe
  unclear/unnamed scene openings, ambiguous pronoun sentences, and
  unattributed dialogue, rather than only POV/TTS failures. Unknown
  foreign tokens or unresolved professional blockers still fail the
  hard gate honestly; `/healthz` exposes `tts_token_repair` beside
  the sanitizer and remaining-token diagnostics.

## v5.38 — TTS-safe foreign-script repair

- **A few foreign letters no longer kill an otherwise valid script.**
  The first live v5.37 job failed only for 16 Latin/CJK/Korean
  characters after the whole-script story editor was rejected. The
  pass now first replaces ledger-known forms deterministically:
  character source names/aliases use their safe Burmese spoken or
  approved short form, glossary source terms use the locked
  `spoken_term`, and important on-screen source text uses its
  recorded Burmese meaning. A foreign-script card `spoken_name` is
  treated as evidence only, never as a TTS replacement target; when
  no speakable name form exists, one unique Burmese identity/visual
  label is the fallback.
- **Unknown tokens get targeted scene repair.** Direct and worker
  narrator-repair rounds now include TTS-unsafe scenes even when POV
  is already correct and even when the story-editor candidate was
  invalid. Repairs must preserve content and length, use locked
  Burmese name/term forms first, and transliterate/translate only
  when no locked form exists.
- **The hard gate remains, with evidence.** Unknown foreign script
  is not silently deleted or guessed. If bounded repairs still
  leave it, the failure and `/healthz` `last_script_quality` now name
  the exact remaining tokens and sanitize replacements.

## v5.37 — Professional script, edit-sync, and spoken-name gate

- **Footage must cover the window it narrates.** Model scene
  timestamps are validated against the whole source window. A
  compressed/late timestamp scale (the rejected v5.36 final began at
  62.7s of a 113.4s window and replayed all 11 late scenes) is
  renormalized in relative order across the full window before the
  editor, final validation, and alignment instead of being trusted.
  Absolute source-video timestamps in a later chunk are shifted back
  to chunk-relative time. Worker transcript timestamps remain ground
  truth and are never renormalized onto silent head/tail footage.
- **No consecutive retelling.** In addition to exact/near-duplicate
  cleanup, adjacent paraphrase echoes across a sentence or scene
  boundary and repeated clauses inside one sentence are removed when
  actor/quantity/negation/opposite-action checks prove they add no
  new event. An explicitly marked repeated action (“again/a second
  time”) is protected. Direct prompts also run a no-echo test across
  adjacent sentences and scene boundaries.
- **Names are spoken, not merely stored in cards.** A supported
  person's first short-form mention is promoted to the full locked
  spoken name and a card-supported Burmese introduction is inserted
  when the opening window does not already say who they are (a clear
  reciprocal relationship such as “I am Mina's brother” is not echoed
  back as “Mina is my sister”). Clear early `သူ/သူမ` actors are
  replaced with the one supported Name Ladder form; ambiguous cases
  and quoted speech are left untouched. Foreign-script name forms are
  never inserted into TTS text by the deterministic passes.
- **Severe defects no longer ship silently.** After all bounded
  editor/repair passes, direct and worker scripts fail honestly when
  local evidence still shows TTS-unsafe foreign script, repeated
  narration, a weak Cast Lock, many unclear/unnamed actors, missing
  introductions, generic cast labels, or an unreliable scene
  timeline. `/healthz` records a compact `last_script_quality`
  summary (score, failures, duplicates, actor clarity, introductions,
  TTS safety, timeline before/after, dedupe, name repairs, and editor
  status) so the next bad final can be diagnosed from evidence.

## v5.36 — Duration calibration converges after deploys

- **Edge Thiha cold starts use its measured pace.** The learned-rate
  process reset on every deploy previously fell back to 12 chars/sec,
  so the first v5.35 live job wrote 1,365 chars that measured only
  75.5 seconds against a 108.7-second target. Production repeatedly
  measures 17.8-18.6 chars/sec, so the Edge seed is now 18 and a 120s
  window at 90% plans about 1,944 chars from the first draft.
- **A small or empty calibration top-up no longer ends the job.** If
  measured narration is still short and the “new scenes only” answer
  fails, returns `scenes: []`, under-delivers, or makes no progress,
  calibration now expands the existing timed scenes in place against
  proportional per-scene character targets. It preserves scene order
  and timestamps, first-person protagonist POV, the Name Ladder, and
  source-only facts, then re-voices and re-measures.
- **The duration refusal stays honest.** A recap that remains below
  the ~90% floor after all bounded calibration rounds is still not
  padded, sped up, or shipped short; finals still never exceed source.

## v5.35 — First-person protagonist lived-story recap

- **The protagonist tells their own story.** This release supersedes
  the former outside-third-person rule. The opening self-introduces
  the protagonist by full spoken name/stable label, identity/role,
  key relationship, time/place/current situation, and inciting
  problem. Later parts resume from the prior result instead of
  repeating that introduction. The protagonist then uses one
  source-supported locked `ကျွန်တော်`/`ကျွန်မ`; wrong/mixed pronouns,
  another person's unquoted “I”, and protagonist-name subject drift
  are quote-aware validated and repaired or refused.
- **Everyone else keeps the Name Ladder.** Non-protagonists still get
  a detailed full-name introduction, a source-approved routine short
  form only, an identity anchor on return, retained name evidence,
  and a one-time reveal bridge. Other people may address the
  protagonist by name/title in dialogue without turning the
  protagonist back into a third-person subject.
- **Spoken storytelling, not a book/report.** Shared direct, worker,
  editor, repair, top-up, and calibration prompts now require the
  lived chain previous result → want → obstacle/demand → action →
  reaction/decision → consequence/new problem. Source-supported
  protagonist thoughts are allowed; invented inner monologue is not.
  Deterministic safeguards convert common written-register endings,
  flag bookish/report-like style for the editor, and require named
  dialogue exchanges rather than “they discussed it.”
- **No repeated retelling.** Exact and conservative near-duplicate
  sentence/scene cleanup runs inside windows as well as across the
  joined script, with actor, quantity, negation, and opposite-action
  safeguards. A protagonist-present script without an established
  first-person voice—or an opening without the self-introduction—is
  rejected before voicing/rendering/delivery.
- **Length and edit guarantees are unchanged.** This is not a short
  summary mode: the full ~90% retelling, final-never-longer-than-
  source rule, whole-sentence measured calibration, Edge Thiha first
  with Gemini Charon fallback, Turbo, script-first processing,
  input-side `-t`, no planned freeze, and replay-at-most-twice rules
  remain in force.

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
URLs must show `"version":"v5.40-opening-introduction-repair"`,
`"telegram_webhook_auth":"secret_token_header"`,
`"gemini_hedge_delay_seconds":15`, `"gemini_hedge_initial_parallel":2`,
`"gemini_generate_timeout_seconds":240`, `"gemini_tts_timeout_seconds":300`,
`"gemini_overload_retry_budget_seconds":120`, `"gemini_overload_max_rounds":12`,
`"default_voice":"Charon"`, `"default_deep_voice_filter":false`,
`"script_trim_policy":"sentence_then_word_tail_boundaries"`,
`"duration_calibration":"script_first_dedupe_expand_short_shave_over_source"`,
`"edit_sync":"proportional_boundaries_min_span_repair_cut_snapped_input_side_t_replay_max2_no_freeze"`,
`"story_continuity":"cumulative_story_bible_character_cards_ordered_beats"`,
`"story_editor":"clarity_gated_bounded_text_only_pass"`,
`"script_style":"pro_recap_first_person_protagonist_spoken_storytelling_name_ladder_near_duplicate_cleanup_conflict_stakes_reactions_causal_bridges_dialogue_exchanges_cast_lock"`,
`"narrator_enforcement":"first_person_protagonist_locked_pronoun_quote_aware_per_scene_validated"`,
`"cast_lock":"subtitle_on_screen_first_name_ladder_full_spoken_name_approved_short_form_or_stable_label_ambiguous_alias_rejected"`,
`"script_qa":"cumulative_name_ladder_introductions_sentence_identity_generic_labels_dialogue_reactions_stakes_ending_hook_tts_safe_editor_gate"`,
`"story_clarity_contract":"who_is_who_first_introductions_conflict_reason_stakes_reaction_reasons_why_next_bridges_dialogue_exchanges_unresolved_ending"`,
`"story_comprehension":"protagonist_self_introduction_locked_pronoun_natural_name_ladder_full_introductions_approved_short_forms_name_evidence_reveal_bridges_identity_anchors_glossary_first_explanations_visible_text_meanings_causal_bridges_stakes"`,
`"delivery_integrity":"exact_source_window_count_split_duration_coverage_preflight_streamed_upload_retry_after"`,
`"video_geometry":"display_sar_rotation_square_pixel_output"`,
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
`"source_split_acceptance":"stream_copy_exact_plan_count_max_1p5x_window_then_720p_ultrafast_fallback"`,
`"render_keepalive":"self_ping_public_healthz_while_jobs_active"`,
`"segment_pipeline":"download_split_120s_script_all_dedupe_then_voice_edit_concat_send"`,
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
requires one real Telegram send judged story-comprehensible by ear,
consistent character names/labels, narration matched to the footage,
no frozen stretches, any scene replay limited to two total, and a finished final no longer than
the original.
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
