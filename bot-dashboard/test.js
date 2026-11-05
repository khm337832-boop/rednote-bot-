const {test} = require('node:test');
const assert = require('node:assert/strict');
const {createServer} = require('./server');

async function withServer(fetchHealth, fn) {
  const server = createServer(fetchHealth);
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  try { await fn(`http://127.0.0.1:${server.address().port}`); }
  finally { await new Promise(resolve => server.close(resolve)); }
}
test('shows aggregate live status without forwarding private fields', async () => {
  await withServer(async () => ({ok:true, telegram_webhook_ready:true, active_jobs:2,
    paused_retry_jobs:1, version:'v1', token:'secret', chat_id:123}), async base => {
    const r = await fetch(base+'/api/status'); const data = await r.json();
    assert.equal(r.status, 200); assert.equal(data.active_jobs, 2); assert.equal(data.paused_retry_jobs, 1);
    assert.equal(data.token, undefined); assert.equal(data.chat_id, undefined);
    assert.ok(data.checked_at); assert.equal(r.headers.get('cache-control'), 'no-store');
  });
});
test('unavailable and invalid upstream do not appear healthy', async () => {
  for (const get of [async () => {throw Error('down')}, async () => ({ok:true})]) {
    await withServer(get, async base => {
      const r = await fetch(base+'/api/status'); assert.equal(r.status, 503);
      assert.deepEqual(await r.json(), {error:'Bot status unavailable'});
    });
  }
});
test('serves dashboard and rejects unknown paths and methods', async () => {
  await withServer(async () => ({}), async base => {
    let r=await fetch(base+'/'); assert.equal(r.status,200); assert.match(await r.text(), /subtitle studio/);
    r=await fetch(base+'/api/status?url=http://localhost'); assert.equal(r.status,404);
    r=await fetch(base+'/api/status',{method:'POST'}); assert.equal(r.status,405);
  });
});
