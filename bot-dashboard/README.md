# Subtitle Studio dashboard

A separate, read-only status dashboard for the existing Telegram bot. It does not change the bot, its webhook, jobs, editor, or storage. It reads the bot's existing public `/healthz` endpoint server-side, every 30 seconds. Counts are aggregate, not individual job progress, and a healthy response cannot prove that an active job is making progress. If the upstream cannot be reached (or is waking up), the dashboard shows unavailable instead of assuming the bot is stopped.

## Run locally

Node.js 18 or later is required:

```sh
cd bot-dashboard
npm test
npm start
```

Open `http://localhost:3000`. The dashboard uses the live production URL by default; set `BOT_HEALTH_URL` on the dashboard service to use another trusted endpoint. The destination is configured on the server, not by visitors. It must be reachable from that service. Some sandbox environments block outbound access to Render, so their preview will show unavailable.

## Deploy without redeploying the bot

Create a **separate** Render web service using this repository and the Dockerfile at `bot-dashboard/Dockerfile` (Docker build context: `bot-dashboard`). The existing bot service and its webhook must be left unchanged. A separate service may also sleep on Render's free tier. The dashboard never triggers a bot restart; however, the bot host's own sleep/restarts and deploy behavior remain unchanged. Do not merge changes into the bot's auto-deploy branch merely to publish this dashboard, as that would redeploy the bot. Check free-tier availability and restrictions on Render.

The dashboard exposes only the aggregate fields already published by `/healthz`; it is not an authenticated administrative console. Do not add video, subtitle, user, or credential information to this public endpoint.
