// Pictures of the models, for looking at them closely while working on them.
//   node shots.mjs clay <glb> <out prefix> "az:el[:mode[:half[:cx,cy,cz]]]" ...
//       a model on its own, in plain grey, from the given directions (degrees; az 0 is the front,
//       90 from +x), without perspective; mode clay, flat, wire, normal, color or an attribute's
//       name (_VESSEL ...); half is half the height shown, in the model's units
//   node shots.mjs page <days> <out prefix> "anchor|x,y,z:az:el:dist[:fov]" | "view:<name>" ...
//       the model in Luna View as the app's Test tab opens it, looked at from the given direction
//       in the model's own frame, or in one of the page's own views (close, whole, face, hands, feet)
//   node shots.mjs smoke [days,days,...]
//       steps the page through the weeks and reports errors and warnings
// Environment: W, H (picture size), MODE (realistic or dreamy), EMB=0 (the fetus model in the early
// weeks), NOWOMB, NOCORD, NODOF, MODELDIR (models from another folder), LOOK (JSON over rig.look),
// MOVE=part:angle (a touched part held at that angle), WAIT (frames before each picture).
// Needs Playwright with Chromium; three.js from node_modules here if it is installed (npm i three),
// otherwise from the CDN the page itself uses.
import { createRequire } from 'module';
import http from 'http';
import fs from 'fs';
import path from 'path';
const TOOLS = path.dirname(new URL(import.meta.url).pathname);
let chromium;
try { ({ chromium } = await import('playwright')); } catch (err) { ({ chromium } = createRequire('/opt/node-tools/node_modules/')('playwright')); }
const ROOT = process.env.ROOT || path.resolve(TOOLS, '..');
const THREE_DIR = path.join(TOOLS, 'node_modules/three');
const LOCAL_THREE = fs.existsSync(path.join(THREE_DIR, 'build/three.module.js'));
const CDN = 'https://cdn.jsdelivr.net/npm/three@0.186.1/';
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.glb': 'model/gltf-binary', '.png': 'image/png', '.jpg': 'image/jpeg', '.svg': 'image/svg+xml', '.css': 'text/css', '.webp': 'image/webp' };
function patchIndex(html) {
  if (LOCAL_THREE) html = html.replace(/https:\/\/cdn\.jsdelivr\.net\/npm\/three@[\d.]+\/build\/three\.module\.js/, '/vendor/three/build/three.module.js')
    .replace(/https:\/\/cdn\.jsdelivr\.net\/npm\/three@[\d.]+\/examples\/jsm\//, '/vendor/three/examples/jsm/');
  if (!html.includes('composer.render(dt);')) throw new Error('no composer.render');
  html = html.replace('composer.render(dt);', 'if (window.__pre) window.__pre({ now, S, ga, dt });\n  composer.render(dt);');
  const tail = `\nwindow.__lv = { THREE, state, camera, controls, scene, embryo, gltfBaby, babyRoot, renderer, setDays, setView, applyTier, TIERS, wombWall, wombShell, placentaGroup, cord, particles, stars, sparks, moon, hemi, keyLight, backLight, headLight, embryoUniforms, babyUniforms, MODES, viewCam, embryoMove, sceneSize, gaOf, babyAnchor, get embryoActive() { return embryoActive; }, get composer() { return composer; }, get scenePass() { return scenePass; } };\n`;
  if (!html.includes('renderer.setAnimationLoop(frameLoop);')) throw new Error('no setAnimationLoop');
  return html.replace('renderer.setAnimationLoop(frameLoop);', 'renderer.setAnimationLoop(frameLoop);' + tail);
}
const server = http.createServer((req, res) => {
  const u = new URL(req.url, 'http://x');
  let p = decodeURIComponent(u.pathname);
  let file;
  if (p.startsWith('/vendor/three/') && !LOCAL_THREE) { res.writeHead(302, { Location: CDN + p.slice('/vendor/three/'.length) }); res.end(); return; }
  if (p.startsWith('/vendor/three/')) file = path.join(THREE_DIR, p.slice('/vendor/three/'.length));
  else if (p.startsWith('/tools/')) file = path.join(TOOLS, p.slice('/tools/'.length));
  else if (p.startsWith('/abs/')) file = p.slice('/abs'.length);
  else file = path.join(ROOT, p === '/' ? 'index.html' : p);
  if (process.env.MODELDIR && p.startsWith('/models/') && fs.existsSync(path.join(process.env.MODELDIR, path.basename(p)))) file = path.join(process.env.MODELDIR, path.basename(p));
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); res.end('nf'); return; }
    if (file === path.join(ROOT, 'index.html')) data = Buffer.from(patchIndex(data.toString('utf8')));
    res.writeHead(200, { 'Content-Type': TYPES[path.extname(file)] || 'application/octet-stream', 'Cache-Control': 'no-store' });
    res.end(data);
  });
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const port = server.address().port;
const [mode, a1, out, ...specs] = process.argv.slice(2);
const W = +(process.env.W || 700), H = +(process.env.H || 700);
const browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist', '--enable-webgl'] });
const ctx = await browser.newContext({ viewport: { width: W, height: H }, deviceScaleFactor: 1, reducedMotion: 'reduce' });
const page = await ctx.newPage();
page.on('console', (m) => { if (m.type() === 'error' || process.env.VERBOSE) console.log('[page]', m.type(), m.text().slice(0, 300)); });
page.on('pageerror', (e) => console.log('[pageerror]', e.message.slice(0, 300)));
try {
  if (mode === 'clay') {
    const glb = a1.startsWith('/') ? '/abs' + a1 : '/' + a1;
    await page.goto(`http://127.0.0.1:${port}/tools/clay.html?glb=${encodeURIComponent(glb)}&w=${W}&h=${H}`);
    await page.waitForFunction('window.ready === true', null, { timeout: 120000 });
    if (process.env.INFO) console.log(JSON.stringify(await page.evaluate('window.info()')));
    let i = 0;
    for (const s of specs) {
      const [az, el, m = 'clay', half = '0.62', c = '0,0,0'] = s.split(':');
      const [cx, cy, cz] = c.split(',').map(Number);
      const o = { az: +az, el: +el, half: +half, cx, cy, cz };
      if (m.startsWith('_')) o.attr = m; else o.mode = m;
      const url = await page.evaluate((o) => window.shot(o), o);
      fs.writeFileSync(`${out}-${++i}.png`, Buffer.from(url.split(',')[1], 'base64'));
    }
  } else if (mode === 'page') {
    const days = +a1;
    // as the app's Test tab opens it (realistic mode, full view); MODE=dreamy for the page's own default
    const mode = process.env.MODE || 'realistic';
    const emb = (process.env.EMB === '0' ? '?' : '?embryo=1&') + `app=1&view=full&lang=bg&mode=${mode}&days=${days}`;
    await page.goto(`http://127.0.0.1:${port}/index.html${emb}`);
    await page.addStyleTag({ content: 'body * { visibility: hidden !important } #scene { visibility: visible !important }' });
    await page.waitForFunction('window.__lv && window.__lunaReady', null, { timeout: 180000 });
    await page.evaluate(({ days, opt }) => {
      const L = window.__lv;
      L.applyTier(L.TIERS.length - 1);
      L.setDays(days);
      window.__spec = null;
      window.__frames = 0;
      window.__pre = ({ S }) => {
        const T = L.THREE;
        window.__frames += 1;
        if (opt.nowomb) for (const o of [L.wombWall, L.wombShell, L.placentaGroup, L.particles, L.stars, L.sparks]) o.visible = false;
        if (opt.nocord) { L.cord.visible = false; }
        const sp = window.__spec;
        if (opt.move && L.embryoActive) {
          // a touched part held at the furthest point of its swing
          const [name, angle] = opt.move.split(':');
          const j = L.embryo.rig.joints[name];
          L.embryoMove.uMoveAngle.value = +angle;
          L.embryoMove.uMovePart.value = j.part;
          L.embryoMove.uMovePivot.value.fromArray(j.pivot);
          L.embryoMove.uMoveAxis.value.fromArray(j.axis);
        }
        if (opt.nodof && L.scenePass.dof) L.scenePass.material.uniforms.uMaxBlur.value = 0;
        if (!sp || sp.anchor === 'page') return;
        const g = L.embryoActive ? L.embryo.group : L.gltfBaby.group;
        g.updateMatrixWorld(true);
        const tgt = new T.Vector3();
        if (sp.point) tgt.fromArray(sp.point).applyMatrix4(g.matrixWorld); else L.babyAnchor(sp.anchor, tgt);
        const a = sp.az * Math.PI / 180, e = sp.el * Math.PI / 180;
        const q = new T.Quaternion(); g.getWorldQuaternion(q);
        const dir = new T.Vector3(Math.sin(a) * Math.cos(e), Math.sin(e), Math.cos(a) * Math.cos(e)).applyQuaternion(q);
        const up = new T.Vector3(0, 1, 0).applyQuaternion(q);
        L.controls.target.copy(tgt);
        L.camera.position.copy(tgt).addScaledVector(dir, sp.dist * S);
        L.camera.up.copy(up);
        L.camera.fov = sp.fov; L.camera.clearViewOffset(); L.camera.aspect = opt.W / opt.H; L.camera.updateProjectionMatrix();
        L.camera.lookAt(tgt); L.camera.updateMatrixWorld();
        if (opt.nodof && L.scenePass.dof) L.scenePass.material.uniforms.uMaxBlur.value = 0;
      };
    }, { days, opt: { nowomb: !!process.env.NOWOMB, nocord: !!process.env.NOCORD, nodof: !!process.env.NODOF, move: process.env.MOVE || '', W, H } });
    const wantEmb = process.env.EMB !== '0' && days < 77;
    await page.waitForFunction((w) => (w ? window.__lv.embryoActive : window.__lv.gltfBaby.ready), wantEmb, { timeout: 180000 });
    if (process.env.LOOK) await page.evaluate((look) => {
      const L = window.__lv;
      const rig = L.embryo.rig;
      rig.look = Object.assign({}, rig.look || {}, look);
      if (look.indirect != null) L.embryoUniforms.uEmbIndirect.value = look.indirect;
      if (look.through) L.embryoUniforms.uThroughSigma.value.fromArray(look.through);
      if (look.pale != null) L.embryoUniforms.uPaleAmount.value = look.pale;
    }, JSON.parse(process.env.LOOK));
    let i = 0;
    for (const s of specs) {
      const [an, az, el, dist, fov = '24'] = s.split(':');
      if (an === 'view') {
        // one of the page's own views (close, whole, face, hands, feet), as the page frames it
        await page.evaluate((v) => { window.__spec = { anchor: 'page' }; window.__lv.setView(v); window.__frames = 0; }, az);
        await page.waitForTimeout(4500);
        await page.waitForFunction((n) => window.__frames >= n, 6, { timeout: 180000 });
        await page.screenshot({ path: `${out}-${++i}.png`, timeout: 180000 });
        continue;
      }
      const spec = { az: +az, el: +el, dist: +dist, fov: +fov };
      if (an.includes(',')) spec.point = an.split(',').map(Number); else spec.anchor = an;
      await page.evaluate((sp) => { window.__spec = sp; window.__frames = 0; }, spec);
      await page.waitForFunction((n) => window.__frames >= n, +(process.env.WAIT || 8), { timeout: 180000 });
      await page.screenshot({ path: `${out}-${++i}.png`, timeout: 180000 });
    }
  }
  else if (mode === 'smoke') {
    // the page as the app's Test tab opens it: step through the weeks and watch for errors
    const errors = [];
    page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') errors.push(`${m.type()}: ${m.text().slice(0, 200)}`); });
    page.on('pageerror', (e) => errors.push(`pageerror: ${e.message.slice(0, 200)}`));
    await page.goto(`http://127.0.0.1:${port}/index.html?embryo=1&app=1&view=full&lang=bg&mode=realistic&days=57`);
    await page.waitForFunction('window.__lv && window.__lunaReady', null, { timeout: 180000 });
    await page.evaluate(() => { window.__frames = 0; window.__pre = () => { window.__frames += 1; }; });
    for (const days of (a1 || '57,65,73,76,77,84,73,65,57').split(',').map(Number)) {
      await page.evaluate((d) => { window.__lv.setDays(d); window.__frames = 0; }, days);
      const emb = days < 77;
      await page.waitForFunction((w) => (w ? window.__lv.embryoActive && window.__lv.embryo.week === w : !window.__lv.embryoActive && window.__lv.gltfBaby.ready), emb ? (days < 63 ? 8 : days < 70 ? 9 : 10) : 0, { timeout: 180000 });
      await page.waitForFunction((n) => window.__frames >= n, 5, { timeout: 180000 });
      const st = await page.evaluate(() => ({ week: window.__lv.embryo.week, active: window.__lv.embryoActive, look: window.__lv.embryo.rig?.look || null, meshes: window.__lv.embryoActive ? window.__lv.embryo.models[window.__lv.embryo.week].meshes.length : 0 }));
      console.log(`days ${days}: embryo week ${st.week}, meshes ${st.meshes}, look ${st.look ? 'own' : 'page'}`);
      for (const v of ['close', 'face', 'hands', 'feet', 'whole']) await page.evaluate((v) => window.__lv.setView(v), v);
    }
    console.log(errors.length ? `ERRORS (${errors.length}):\n` + [...new Set(errors)].join('\n') : 'no errors or warnings');
  }
} finally {
  await browser.close();
  server.close();
}
