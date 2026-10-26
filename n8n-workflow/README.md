# n8n Pipeline တပ်ဆင်ပုံ — Whisper First-Person Recap

ဒီ workflow က လက်ရှိ Render Telegram bot ကို အစားထိုးတာ မဟုတ်ပါ။

- **Telegram bot** က link လက်ခံ၊ settings ရွေး၊ status ပြပြီး final video ပြန်ပို့ခြင်းတို့ကို ဆက်လုပ်ပါသည်။
- **n8n** က အဆင့်များကို အစဉ်လိုက် စီမံပါသည်။
- **Render bot server** ကပဲ yt-dlp၊ ffmpeg၊ Gemini၊ Whisper နှင့် Telegram upload စတဲ့ လေးလံသည့် အလုပ်များကို ၎င်း၏ authenticated worker API မှတစ်ဆင့် လုပ်ပါသည်။

## Flow

```text
Telegram user က link ပို့
  -> Telegram bot က job/status ဖန်တီး
  -> n8n Webhook (onReceived, ချက်ချင်း respond)
  -> Worker job အတည်ပြု/ဖန်တီး
  -> download
  -> Whisper transcribe (timestamps ပါ)
  -> Gemini Burmese first-person rewrite (coverage စစ် + top-up)
  -> Gemini TTS (Telegram မှာ ရွေးထားသည့်အသံ)
  -> Render (narration ကို မညှိ၊ video ကိုပဲ ညှိ/ဖြတ်)
  -> Telegram deliver (ဖိုင်ကြီးလျှင် အပိုင်းခွဲ)
  -> n8n က worker job files ရှင်း
```

## ဖိုင်များ

| ဖိုင် | အဓိပ္ပာယ် |
|---|---|
| `rednote-n8n-workflow.json` | n8n ထဲသို့ တိုက်ရိုက် import လုပ်နိုင်သော workflow |
| `build_workflow.py` | အထက် JSON ကို ပြန်ထုတ်ပေးသော generator |
| `REQUIREMENTS.md` | မူရင်းလိုအပ်ချက်များနှင့် အတည်ပြုပြီးသား ဆုံးဖြတ်ချက်များ |
| `README.md` | ယခု တပ်ဆင်နည်းလမ်းညွှန် |

Workflow ကို ပြင်ပြီးနောက် JSON ပြန်ထုတ်ချင်လျှင်:

```bash
cd n8n-workflow
python3 build_workflow.py
```

## ၁) Secrets နှစ်ခု အရင်ဖန်တီးပါ

အောက်ပါ secret နှစ်ခုကို မတူသော၊ ရှည်ပြီး ကျပန်းသော တန်ဖိုးများဖြင့် ဖန်တီးပါ။

1. `WORKER_API_KEY` — n8n HTTP node များက Render worker ကို ခေါ်ရန် သုံးသည်။
2. `N8N_WEBHOOK_SECRET` — Telegram bot က n8n Webhook ကို ပို့ရာတွင် သုံးသည်။ မထည့်ဘဲလည်း ရသော်လည်း ထုတ်လုပ်မှုအသုံးအတွက် ထည့်ရန် အကြံပြုပါသည်။

Secret များကို GitHub/chat ထဲ မတင်ပါနှင့်။ Render နှင့် n8n environment ထဲမှာသာ ထည့်ပါ။

## ၂) Render env များ

Render service → **Environment** တွင် အရင်ရှိနေသော `BOT_TOKEN`, `GEMINI_API_KEY`, `ADMIN_ID` တို့ကို မပြောင်းပါနှင့်။ n8n mode အတွက် အောက်ပါတို့ ထပ်ထည့်ပါ။

```text
PIPELINE_MODE=direct
N8N_WEBHOOK_URL=
N8N_WEBHOOK_SECRET=<secret 2>
WORKER_API_KEY=<secret 1>

WHISPER_API_KEY=<Whisper/OpenAI-compatible API key>
# သို့မဟုတ် WHISPER_API_KEY အစား OPENAI_API_KEY ထည့်နိုင်သည်။

WHISPER_API_BASE=https://api.openai.com/v1
WHISPER_MODEL=whisper-1
WHISPER_CHUNK_SECONDS=600
```

အရေးကြီးချက်များ:

- အစတွင် `PIPELINE_MODE=direct` ထားပါ။
- `WORKER_API_KEY` မထည့်လျှင် worker API အားလုံး `503` ပြန်ပြီး ပိတ်ထားပါမည်။
- `WHISPER_API_KEY` မရှိလျှင်လည်း `OPENAI_API_KEY` ကို fallback အဖြစ် သုံးပါသည်။
- အခြား OpenAI-compatible transcription provider သုံးလျှင် `WHISPER_API_BASE` ကို ၎င်း၏ `/v1` base URL ဖြင့် ပြောင်းပါ။ Code က endpoint ကို `${WHISPER_API_BASE}/audio/transcriptions` အဖြစ် ခေါ်ပါသည်။
- `WHISPER_CHUNK_SECONDS=600` ဆိုလျှင် အသံကို ၁၀ မိနစ်အပိုင်းများအဖြစ် ခွဲပို့ပါမည်။ Provider ၏ file-size limit နိမ့်လျှင် ဤတန်ဖိုးကို လျှော့ပါ။

Env ပြောင်းပြီးနောက် Render deploy ပြီးလျှင် အောက်ပါ URL ကို ဖွင့်စစ်ပါ:

```text
https://your-render-service.onrender.com/
```

မျှော်လင့်ထားသော အခြေအနေ:

```json
{
  "status": "online",
  "pipeline_mode": "direct",
  "worker_api_configured": true,
  "whisper_configured": true,
  "n8n_webhook_configured": false
}
```

Secret တန်ဖိုးများကို API response တွင် မပြပါ။

## ၃) n8n env များ

n8n run နေသော environment တွင် အောက်ပါတို့ ထည့်ပါ။

```text
WORKER_BASE_URL=https://your-render-service.onrender.com
WORKER_API_KEY=<Render နှင့် အတူတူသော secret 1>
N8N_WEBHOOK_SECRET=<Render နှင့် အတူတူသော secret 2>
```

- `WORKER_BASE_URL` အဆုံးတွင် `/` ပါပါက workflow က အလိုအလျောက် ဖယ်ပေးပါသည်။
- Self-hosted n8n တွင် node expression များက environment variable ဖတ်ခွင့်မရလျှင် အောက်ပါပါ ထည့်ရနိုင်ပါသည်:

```text
N8N_BLOCK_ENV_ACCESS_IN_NODE=false
```

ပြီးနောက် n8n ကို restart လုပ်ပါ။

n8n Cloud သုံးလျှင်လည်း ဤ workflow က Execute Command မသုံးသောကြောင့် လေးလံသောအလုပ်များကို Render worker က လုပ်ပါသည်။ သို့သော် သင်၏ n8n plan/environment တွင် အထက် env များကို သတ်မှတ်နိုင်/မနိုင် အရင်စစ်ပါ။

## ၄) Workflow import လုပ်ခြင်း

1. n8n → **Workflows** → **Import from File** ကို ရွေးပါ။
2. `rednote-n8n-workflow.json` ကို import လုပ်ပါ။
3. Workflow name သည် **Rednote Bot — Whisper First Person Recap (Worker API)** ဖြစ်သင့်ပါသည်။
4. Sticky note နှင့် node connection များ ပေါ်လာ/မလာ စစ်ပါ။
5. အရင်ဆုံး workflow ကို **Inactive** အတိုင်းပဲ ထားပါ။

ဒီ workflow မှာ Telegram Trigger မရှိခြင်းသည် အမှားမဟုတ်ပါ။ တူညီသော Telegram bot token အတွက် webhook နှစ်ခု မပြိုင်စေရန် ရည်ရွယ်ချက်ရှိရှိ ချန်ထားခြင်းဖြစ်သည်။

## ၅) Production Webhook URL ရယူခြင်း

1. **Telegram Bot မှလာသော Job** Webhook node ကို ဖွင့်ပါ။
2. Path သည် အောက်ပါအတိုင်း ဖြစ်သင့်ပါသည်:

   ```text
   rednote-recap-job
   ```

3. Response Mode သည် **Respond Immediately / onReceived** ဖြစ်သင့်ပါသည်။
4. Workflow ကို **Activate** လုပ်ပါ။
5. **Production URL** ကို ကူးပါ။ Test URL ကို မသုံးပါနှင့်။ ဥပမာ:

   ```text
   https://your-n8n.example/webhook/rednote-recap-job
   ```

## ၆) Render ကို n8n နှင့် ချိတ်ပြီး အသက်သွင်းခြင်း

Workflow activate လုပ်ပြီးမှသာ Render တွင် အောက်ပါတို့ ထည့်ပါ:

```text
N8N_WEBHOOK_URL=https://your-n8n.example/webhook/rednote-recap-job
PIPELINE_MODE=n8n
```

ပြီးနောက် Render ကို redeploy/restart လုပ်ပါ။ Root URL တွင် အောက်ပါအတိုင်း ဖြစ်သင့်ပါသည်:

```json
{
  "pipeline_mode": "n8n",
  "worker_api_configured": true,
  "whisper_configured": true,
  "n8n_webhook_configured": true
}
```

Telegram တွင် test video link တစ်ခု ပို့ပြီး စစ်ပါ။ n8n Executions နှင့် Telegram status message နှစ်ခုလုံးတွင် download → deliver အထိ အဆင့်ဆင့် တိုးတက်နေသည်ကို မြင်ရပါမည်။

> n8n Webhook က 2xx မပြန်လျှင် bot က မူလ direct pipeline သို့ အလိုအလျောက် ပြန်ကျပါမည်။ သို့သော် n8n internal error/platform error များကို n8n execution logs တွင် ဆက်လက်စစ်ဆေးရပါမည်။

## Worker API အကျဉ်းချုပ်

Header:

```http
X-Worker-Key: <WORKER_API_KEY>
Content-Type: application/json
```

Endpoints:

| Method | URL | အလုပ် |
|---|---|---|
| `POST` | `/api/v1/jobs` | Job ဖန်တီး သို့မဟုတ် ရှိပြီးသားကို အတည်ပြု |
| `GET` | `/api/v1/jobs/{job_id}` | Job/stage status ဖတ် |
| `POST` | `/api/v1/jobs/{job_id}/stages/{stage}` | Stage တစ်ခု စတင်ပြီး `202` ပြန် |
| `POST` | `/api/v1/jobs/{job_id}/cancel` | Job ရပ် |
| `POST` | `/api/v1/jobs/{job_id}/fail` | Job ကို failed အဖြစ် အဆုံးသတ် |
| `DELETE` | `/api/v1/jobs/{job_id}` | Job folder ရှင်း |

Stage အမည်များ:

```text
download -> transcribe -> rewrite -> tts -> render -> deliver
```

Stage start သည် idempotent ဖြစ်နိုင်ရန် စီစဉ်ထားပါသည်။ Stage ပြီးပြီးသားဆိုလျှင် `200`၊ ယခုမှစလျှင် သို့မဟုတ် လုပ်နေဆဲဆိုလျှင် `202` ပြန်ပါသည်။ ယခင် stage မပြီးဘဲ နောက် stage စ ခေါ်လျှင် `409` ပြန်ပါသည်။

## Coverage နှင့် Video Alignment မှတ်ချက်များ

- Gemini က Whisper timestamp များကို ကိုယ်တိုင် မတီထွင်ရပါ။ Worker က Whisper segment များကို group လုပ်ပြီး group ID/start/end ကို ထိန်းထားပါသည်။
- Job/link တိုင်းအတွက် coverage အနည်းဆုံး **90%** ဖြစ်သည်။ Settings တွင် 90% အောက် သိမ်းထားသော တန်ဖိုးဟောင်း ရှိနေလျှင်လည်း worker က 90% သို့ အလိုအလျောက် မြှင့်သည်။ 95% သို့မဟုတ် 100% ရွေးထားလျှင် ၎င်းအတိုင်း သုံးသည်။
- Rewrite အောင်မြင်ရန် script ထဲတွင် ကိုယ်စားပြုထားသည့် Whisper speech duration သည် ရွေးထားသော coverage (အနည်းဆုံး 90%) အထက် ရှိရမည်။ Scene စာသားတစ်ကြောင်းတည်း ရှိနေရုံနှင့် covered မသတ်မှတ်ပါ။
- Burmese စာသားပမာဏ target ကို source video duration (မသိလျှင်သာ Whisper speech duration)၊ coverage နှင့် Settings ထဲက characters/second ဖြင့် တွက်ပြီး group တစ်ခုစီကို source timeline အလိုက် ခွဲဝေကာ မပြည့်သေးသော group များကို top-up လုပ်ပါသည်။
- Script စာလုံးပမာဏ၊ တကယ့် TTS narration ကြာချိန်နှင့် final render ကြာချိန် သုံးခုလုံးကို target နှင့် ထပ်စစ်ပါသည်။ ပမာဏမပြည့်သော recap ကို Telegram သို့ မပို့ဘဲ stage ကို failed ဖြစ်စေပါသည်။
- Final render တွင် narration ကို `atempo` သို့မဟုတ် audio stretch/pad များဖြင့် မညှိပါ။ Video ကိုပဲ scene အစနှင့် align လုပ်ပြီး ဖြတ်/ကြားခံ footage သုံး/မဖြစ်မနေလိုပါက final frame freeze လုပ်ပါသည်။
- မူရင်း BGM/audio ကို V1 တွင် မထားဘဲ Gemini TTS narration တစ်ခုတည်း ထည့်ပါသည်။
- Subtitle burn-in ကို V1 တွင် မလုပ်သေးပါ။

## ပိတ်ခြင်း / မူလပုံစံသို့ ပြန်သွားခြင်း

ပြဿနာတစ်ခုခုကြုံလျှင် Render တွင်:

```text
PIPELINE_MODE=direct
```

ပြောင်းပြီး redeploy/restart လုပ်ပါ။ ဒါဆိုရင် Telegram bot က n8n မသုံးဘဲ မူလ direct pipeline ကို ပြန်သုံးပါမည်။

Worker API ကိုပါ လုံးဝပိတ်ချင်လျှင် `WORKER_API_KEY` ကို Render env မှ ဖယ်ရှားပါ။

## ပြဿနာဖြေရှင်းချက် အမြန်

| လက္ခဏာ | စစ်ဆေးရန် |
|---|---|
| Worker endpoint က `503` ပြန် | Render တွင် `WORKER_API_KEY` မရှိသေးပါ |
| Worker endpoint က `401` ပြန် | Render နှင့် n8n ၏ `WORKER_API_KEY` မတူပါ |
| n8n Payload Code မှာ secret error | Render နှင့် n8n ၏ `N8N_WEBHOOK_SECRET` မတူပါ |
| n8n HTTP URL က `/api/v1/...` ပုံစံပဲ ဖြစ် | n8n တွင် `WORKER_BASE_URL` မရှိပါ သို့မဟုတ် env access ပိတ်ထားပါသည် |
| Transcribe fail | `WHISPER_API_KEY`/`OPENAI_API_KEY`, `WHISPER_API_BASE`, `WHISPER_MODEL` တို့ စစ်ပါ |
| Bot က n8n မသုံးဘဲ direct ဆက်လုပ် | `PIPELINE_MODE=n8n` နှင့် production `N8N_WEBHOOK_URL` နှစ်ခုလုံး ထည့်ထားလား၊ workflow activate ဖြစ်လား စစ်ပါ |
| Final video မတက် | Telegram status၊ n8n execution နှင့် Render logs မှ stage error ကို စစ်ပါ |

## ကန့်သတ်ချက်များ

- V1 သည် transcript အခြေခံဖြစ်သောကြောင့် စကားပြောသံလုံးဝမပါသော visual-only ဖြစ်ရပ်များကို Gemini က မမြင်နိုင်ပါ။
- အလွန်ကြီးမားသော video များကို size ကြောင့် တမင် မပယ်ချသော်လည်း Render disk, memory, download time, provider limit နှင့် Telegram upload limit များသည် လက်တွေ့ကန့်သတ်ချက်များ ဖြစ်နိုင်ပါသည်။
- Worker job files များကို workflow အောင်မြင်/ပျက်ပြီးလျှင် n8n က cleanup လုပ်ပါသည်။ Workflow ကြားဖြတ်ပျက်လျှင် job folder ကျန်နေနိုင်ပါသည်။
