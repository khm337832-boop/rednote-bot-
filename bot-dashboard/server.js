'use strict';
const http = require('node:http');
const fs = require('node:fs');
const path = require('node:path');

// Fixed upstream: requests cannot supply a destination, preventing an open proxy.
const HEALTH_URL = process.env.BOT_HEALTH_URL || 'https://rednote-bot-gzd0.onrender.com/healthz';
const page = fs.readFileSync(path.join(__dirname, 'index.html'));
function createServer(fetchHealth = async () => {
  const res = await fetch(HEALTH_URL, {signal: AbortSignal.timeout(12000), redirect: 'error'});
  if (!res.ok || !(res.headers.get('content-type') || '').includes('application/json')) throw Error('Unavailable');
  return res.json();
}) {
  return http.createServer(async (req, res) => {
    res.setHeader('Cache-Control', 'no-store');
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('Referrer-Policy', 'no-referrer');
    if (req.method !== 'GET') { res.writeHead(405); return res.end(); }
    if (req.url === '/' || req.url === '/index.html') {
      res.writeHead(200, {'Content-Type':'text/html; charset=utf-8'}); return res.end(page);
    }
    if (req.url === '/api/status') {
      try {
        const h = await fetchHealth();
        if (typeof h.ok !== 'boolean' || typeof h.telegram_webhook_ready !== 'boolean' || !Number.isInteger(h.active_jobs) || !Number.isInteger(h.paused_retry_jobs)) throw Error('Invalid health response');
        // Only expose aggregate health, never tokens, chat IDs, subtitles, or source URLs.
        const data = {ok:h.ok, telegram_webhook_ready:h.telegram_webhook_ready, active_jobs:h.active_jobs,
          paused_retry_jobs:h.paused_retry_jobs, version:String(h.version || ''), checked_at:new Date().toISOString()};
        res.writeHead(200, {'Content-Type':'application/json; charset=utf-8'}); return res.end(JSON.stringify(data));
      } catch (_) {
        res.writeHead(503, {'Content-Type':'application/json; charset=utf-8'}); return res.end(JSON.stringify({error:'Bot status unavailable'}));
      }
    }
    res.writeHead(404); res.end();
  });
}
if (require.main === module) createServer().listen(Number(process.env.PORT || 3000), '0.0.0.0', () => console.log('Dashboard listening'));
module.exports = {createServer};
