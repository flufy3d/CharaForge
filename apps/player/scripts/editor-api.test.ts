/*
 * 本地编辑接口测试：
 *   - ping / HEAD / 方法限制 / 非编辑路径放行
 *   - loopback + Origin 安全限制
 *   - 场景列表（深度、.bak.json 排除、examples）
 *   - 素材列表（PNG 尺寸、损坏图片跳过、缺失目录为空）
 *   - 上传 PNG（1×1）、命名清洗、碰撞后缀、overwrite、32 MB 上限
 *   - 保存场景（校验、pretty + 换行、备份、白名单/穿越拒绝）
 *   - 编辑命令（新增图片图层与关键帧、diff、out 落盘）
 *
 *   pnpm --filter @charaforge/player exec node --experimental-strip-types --disable-warning=ExperimentalWarning --test scripts/editor-api.test.ts
 */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { createServer, request, type Server } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { after, test } from 'node:test';
import { createEditorApi } from '../server/editorApi.ts';
import { encodePng } from './png.ts';

const cleanups: string[] = [];

function tinyScene(title = 'editor api test'): Record<string, unknown> {
  return {
    format: 'charaforge.scene',
    version: 1,
    meta: { title },
    timebase: { fps: 30 },
    duration: 60,
    composition: { width: 320, height: 180 },
    assets: { models: [] },
    layers: [],
  };
}

function writeScene(root: string, rel: string, scene: Record<string, unknown>): void {
  const path = join(root, ...rel.split('/'));
  mkdirSync(join(path, '..'), { recursive: true });
  writeFileSync(path, `${JSON.stringify(scene, null, 2)}\n`);
}

/** 只含 RIFF/VP8X 头的 WebP：列表只解析尺寸，不需要完整像素数据。 */
function webpHeader(width: number, height: number): Buffer {
  const buf = Buffer.alloc(30);
  buf.write('RIFF', 0, 'latin1');
  buf.writeUInt32LE(22, 4);
  buf.write('WEBP', 8, 'latin1');
  buf.write('VP8X', 12, 'latin1');
  buf.writeUInt32LE(10, 16);
  const data = 20;
  const w = width - 1;
  const h = height - 1;
  buf[data + 4] = w & 0xff;
  buf[data + 5] = (w >> 8) & 0xff;
  buf[data + 6] = (w >> 16) & 0xff;
  buf[data + 7] = h & 0xff;
  buf[data + 8] = (h >> 8) & 0xff;
  buf[data + 9] = (h >> 16) & 0xff;
  return buf;
}

function fixture(): string {
  const root = mkdtempSync(join(tmpdir(), 'charaforge-editor-'));
  cleanups.push(root);
  writeScene(root, 'scenes/demo.scene.json', tinyScene('demo'));
  writeScene(root, 'scenes/sub/one.scene.json', tinyScene('one'));
  writeScene(root, 'scenes/sub/a/two.scene.json', tinyScene('two'));
  writeScene(root, 'scenes/sub/a/b/three.scene.json', tinyScene('three'));
  writeFileSync(join(root, 'scenes', 'backup.bak.json'), '{}\n');
  writeScene(root, 'packages/scene/examples/example.scene.json', tinyScene('example'));
  writeFileSync(join(root, 'packages', 'scene', 'examples', 'example.bak.json'), '{}\n');
  writeFileSync(join(root, 'secret.json'), `${JSON.stringify(tinyScene('secret'), null, 2)}\n`);

  mkdirSync(join(root, 'assets', 'editor'), { recursive: true });
  writeFileSync(join(root, 'assets', 'editor', 'static.png'), encodePng(2, 3, new Uint8Array(2 * 3 * 4), false));
  writeFileSync(join(root, 'assets', 'editor', 'wide.webp'), webpHeader(64, 32));
  writeFileSync(join(root, 'assets', 'editor', 'broken.png'), 'not a png');
  // 只写文件头，列表不需要解码整个音频
  writeFileSync(join(root, 'assets', 'editor', 'clip.wav'), Buffer.from('RIFF\x24\x00\x00\x00WAVEfmt ', 'latin1'));
  return root;
}

async function start(root: string): Promise<{ base: string; server: Server }> {
  const api = createEditorApi(root);
  const server = createServer((req, res) => {
    if (!api(req, res)) {
      res.statusCode = 404;
      res.end('next');
    }
  });
  await new Promise<void>((ok) => server.listen(0, '127.0.0.1', ok));
  const addr = server.address();
  if (!addr || typeof addr === 'string') throw new Error('服务没有监听端口');
  return { base: `http://127.0.0.1:${addr.port}`, server };
}

async function stop(server: Server): Promise<void> {
  server.closeAllConnections();
  await new Promise<void>((ok) => server.close(() => ok()));
}

interface ApiResponse {
  status: number;
  body: Record<string, any>;
  text: string;
  headers: Headers;
}

async function fetchJson(url: string, init?: RequestInit): Promise<ApiResponse> {
  const res = await fetch(url, init);
  const text = await res.text();
  let body: Record<string, any> = {};
  if (text) body = JSON.parse(text) as Record<string, any>;
  return { status: res.status, body, text, headers: res.headers };
}

function postJson(url: string, body: unknown): Promise<ApiResponse> {
  return fetchJson(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

function upload(base: string, query: string, body: Buffer): Promise<ApiResponse> {
  return fetchJson(`${base}/__charaforge/editor/upload?${query}`, {
    method: 'POST',
    body: new Uint8Array(body),
  });
}

function oversizedUpload(base: string): Promise<number> {
  return new Promise((resolve, reject) => {
    const url = new URL(`${base}/__charaforge/editor/upload?kind=image&name=huge.png`);
    // 只声明超大 Content-Length，不真的分配 32 MB；服务端应在读取前直接 413。
    const req = request(
      {
        hostname: url.hostname,
        port: url.port,
        path: `${url.pathname}${url.search}`,
        method: 'POST',
        headers: { 'Content-Length': String(32 * 1024 * 1024 + 1) },
      },
      (res) => {
        res.resume();
        res.on('end', () => resolve(res.statusCode ?? 0));
      },
    );
    req.on('error', reject);
    req.end();
  });
}

function mockResponse(): { res: any; text(): string } {
  let payload = '';
  const res: any = {
    statusCode: 0,
    headers: {} as Record<string, unknown>,
    writableEnded: false,
    destroyed: false,
    setHeader(key: string, value: unknown) {
      this.headers[key.toLowerCase()] = value;
    },
    end(chunk?: unknown) {
      if (chunk !== undefined) payload += String(chunk);
      this.writableEnded = true;
    },
  };
  return { res, text: () => payload };
}

after(() => {
  for (const dir of cleanups) rmSync(dir, { recursive: true, force: true });
});

test('ping 返回版本、HEAD 无 body、其他方法 405', async () => {
  const root = fixture();
  const { base, server } = await start(root);
  try {
    const ping = await fetchJson(`${base}/__charaforge/editor/ping`);
    assert.equal(ping.status, 200);
    assert.deepEqual(ping.body, { ok: true, version: 1 });

    const head = await fetch(`${base}/__charaforge/editor/ping`, { method: 'HEAD' });
    assert.equal(head.status, 200);
    assert.equal(await head.text(), '');
    assert.ok(Number(head.headers.get('content-length')) > 0);

    const post = await fetchJson(`${base}/__charaforge/editor/ping`, { method: 'POST' });
    assert.equal(post.status, 405);
    assert.equal(post.body.ok, false);
    assert.equal(typeof post.body.error, 'string');
  } finally {
    await stop(server);
  }
});

test('非编辑路径返回 false 交给后续中间件；未知编辑路由 404', async () => {
  const root = fixture();
  const { base, server } = await start(root);
  try {
    const other = await fetch(`${base}/__charaforge/list-assets?dir=x`);
    assert.equal(other.status, 404);
    assert.equal(await other.text(), 'next');

    const unknown = await fetchJson(`${base}/__charaforge/editor/nope`);
    assert.equal(unknown.status, 404);
    assert.equal(unknown.body.ok, false);
  } finally {
    await stop(server);
  }
});

test('非 loopback / 非本机 Origin 被 403 拒绝', () => {
  const root = fixture();
  const api = createEditorApi(root);

  const remote = mockResponse();
  const remoteReq = {
    url: '/__charaforge/editor/ping',
    method: 'GET',
    headers: {},
    socket: { remoteAddress: '10.1.2.3' },
  } as any;
  assert.equal(api(remoteReq, remote.res), true);
  assert.equal(remote.res.statusCode, 403);

  const origin = mockResponse();
  const originReq = {
    url: '/__charaforge/editor/ping',
    method: 'GET',
    headers: { origin: 'http://evil.example' },
    socket: { remoteAddress: '127.0.0.1' },
  } as any;
  assert.equal(api(originReq, origin.res), true);
  assert.equal(origin.res.statusCode, 403);

  const ok = mockResponse();
  const okReq = {
    url: '/__charaforge/editor/ping',
    method: 'GET',
    headers: { origin: 'http://localhost:5173' },
    socket: { remoteAddress: '::1' },
  } as any;
  assert.equal(api(okReq, ok.res), true);
  assert.equal(ok.res.statusCode, 200);
});

test('场景列表：scenes 深度 <= 3 + examples，排除 .bak.json', async () => {
  const root = fixture();
  const { base, server } = await start(root);
  try {
    const r = await fetchJson(`${base}/__charaforge/editor/scenes`);
    assert.equal(r.status, 200);
    const paths = r.body.scenes.map((s: { path: string }) => s.path);
    assert.deepEqual(paths, [
      'packages/scene/examples/example.scene.json',
      'scenes/demo.scene.json',
      'scenes/sub/a/two.scene.json',
      'scenes/sub/one.scene.json',
    ]);
    for (const s of r.body.scenes) {
      assert.ok(s.size > 0);
      assert.ok(!Number.isNaN(Date.parse(s.mtime)));
    }
    assert.ok(!paths.includes('scenes/backup.bak.json'));
    assert.ok(!paths.includes('scenes/sub/a/b/three.scene.json'));
  } finally {
    await stop(server);
  }
});

test('读取场景：返回校验后的 scene，路径非法被拒绝', async () => {
  const root = fixture();
  const { base, server } = await start(root);
  try {
    const r = await fetchJson(`${base}/__charaforge/editor/scene?path=scenes/demo.scene.json`);
    assert.equal(r.status, 200);
    assert.equal(r.body.ok, true);
    assert.equal(r.body.path, 'scenes/demo.scene.json');
    assert.equal(r.body.scene.meta.title, 'demo');
    const bad = await fetchJson(`${base}/__charaforge/editor/scene?path=../outside.json`);
    assert.equal(bad.status, 400);
    const missing = await fetchJson(`${base}/__charaforge/editor/scene?path=scenes/nope.scene.json`);
    assert.equal(missing.status, 404);
  } finally {
    await stop(server);
  }
});

test('素材列表：图片尺寸、坏图跳过、音频不带尺寸、缺失目录为空', async () => {
  const root = fixture();
  const { base, server } = await start(root);
  try {
    const images = await fetchJson(`${base}/__charaforge/editor/assets?kind=image`);
    assert.equal(images.status, 200);
    assert.deepEqual(images.body.assets.map((a: { name: string }) => a.name), ['static.png', 'wide.webp']);
    const png = images.body.assets[0];
    assert.equal(png.uri, 'editor/static.png');
    assert.equal(png.width, 2);
    assert.equal(png.height, 3);
    assert.ok(png.size > 0);
    const webp = images.body.assets[1];
    assert.equal(webp.width, 64);
    assert.equal(webp.height, 32);
    assert.match(png.id, /^[A-Za-z0-9_.:-]+$/);
    assert.ok(!Number.isNaN(Date.parse(png.mtime)));

    const audio = await fetchJson(`${base}/__charaforge/editor/assets?kind=audio`);
    assert.equal(audio.status, 200);
    assert.deepEqual(audio.body.assets.map((a: { name: string }) => a.name), ['clip.wav']);
    assert.equal(audio.body.assets[0].width, undefined);
    assert.equal(audio.body.assets[0].uri, 'editor/clip.wav');

    const badKind = await fetchJson(`${base}/__charaforge/editor/assets?kind=video`);
    assert.equal(badKind.status, 400);

    const emptyRoot = mkdtempSync(join(tmpdir(), 'charaforge-editor-empty-'));
    cleanups.push(emptyRoot);
    const empty = await start(emptyRoot);
    try {
      const r = await fetchJson(`${empty.base}/__charaforge/editor/assets?kind=image`);
      assert.equal(r.status, 200);
      assert.deepEqual(r.body.assets, []);
    } finally {
      await stop(empty.server);
    }
  } finally {
    await stop(server);
  }
});

test('上传 PNG：1×1 尺寸、名称清洗、碰撞后缀、overwrite、sha256', async () => {
  const root = fixture();
  const { base, server } = await start(root);
  try {
    const png = encodePng(1, 1, new Uint8Array([255, 0, 0, 255]), false);
    const first = await upload(base, 'kind=image&name=My%20Hat.png', png);
    assert.equal(first.status, 200);
    assert.equal(first.body.ok, true);
    assert.equal(first.body.asset.uri, 'editor/My-Hat.png');
    assert.equal(first.body.asset.width, 1);
    assert.equal(first.body.asset.height, 1);
    assert.equal(first.body.asset.size, png.length);
    assert.equal(first.body.asset.sha256, createHash('sha256').update(png).digest('hex'));
    assert.ok(existsSync(join(root, 'assets', 'editor', 'My-Hat.png')));

    const second = await upload(base, 'kind=image&name=My%20Hat.png', png);
    assert.equal(second.status, 200);
    assert.equal(second.body.asset.uri, 'editor/My-Hat-2.png');

    const overwrite = await upload(base, 'kind=image&name=My%20Hat.png&overwrite=1', png);
    assert.equal(overwrite.status, 200);
    assert.equal(overwrite.body.asset.uri, 'editor/My-Hat.png');

    // 扩展名按识别结果，不听用户文件名
    const wrongExt = await upload(base, 'kind=image&name=photo.jpg', png);
    assert.equal(wrongExt.status, 200);
    assert.equal(wrongExt.body.asset.uri, 'editor/photo.png');

    const invalid = await upload(base, 'kind=image&name=x.png', Buffer.from('not an image'));
    assert.equal(invalid.status, 400);

    const listed = await fetchJson(`${base}/__charaforge/editor/assets?kind=image`);
    assert.ok(listed.body.assets.some((a: { name: string }) => a.name === 'My-Hat-2.png'));
  } finally {
    await stop(server);
  }
});

test('上传超过 32 MB 返回 413 且不写文件', async () => {
  const root = fixture();
  const { base, server } = await start(root);
  try {
    const status = await oversizedUpload(base);
    assert.equal(status, 413);
    assert.ok(!existsSync(join(root, 'assets', 'editor', 'huge.png')));
  } finally {
    await stop(server);
  }
});

test('保存场景：校验、pretty + 换行、备份、白名单与穿越拒绝', async () => {
  const root = fixture();
  const { base, server } = await start(root);
  try {
    const invalid = await postJson(`${base}/__charaforge/editor/save-scene`, {
      path: 'scenes/new.scene.json',
      scene: { ...tinyScene(), duration: 0 },
    });
    assert.equal(invalid.status, 400);
    assert.equal(invalid.body.ok, false);
    assert.ok(!existsSync(join(root, 'scenes', 'new.scene.json')));

    const created = await postJson(`${base}/__charaforge/editor/save-scene`, {
      path: 'scenes/created/nested.scene.json',
      scene: tinyScene('created'),
    });
    assert.equal(created.status, 200);
    assert.equal(created.body.path, 'scenes/created/nested.scene.json');
    assert.ok(created.body.bytes > 0);
    const written = readFileSync(join(root, 'scenes', 'created', 'nested.scene.json'), 'utf8');
    assert.ok(written.endsWith('\n'));
    assert.equal(JSON.parse(written).meta.title, 'created');

    const example = await postJson(`${base}/__charaforge/editor/save-scene`, {
      path: 'packages/scene/examples/new.scene.json',
      scene: tinyScene('example new'),
    });
    assert.equal(example.status, 200);
    assert.ok(existsSync(join(root, 'packages', 'scene', 'examples', 'new.scene.json')));

    const backup = await postJson(`${base}/__charaforge/editor/save-scene`, {
      path: 'scenes/demo.scene.json',
      scene: tinyScene('updated'),
      backup: true,
    });
    assert.equal(backup.status, 200);
    assert.equal(backup.body.backup, 'scenes/demo.scene.json.bak.json');
    assert.equal(JSON.parse(readFileSync(join(root, 'scenes', 'demo.scene.json'), 'utf8')).meta.title, 'updated');
    assert.equal(JSON.parse(readFileSync(join(root, 'scenes', 'demo.scene.json.bak.json'), 'utf8')).meta.title, 'demo');

    const badPaths: [string, number][] = [
      ['scenes/../secret.json', 400],
      ['scenes/../../secret.json', 400],
      ['/scenes/x.json', 400],
      ['C:/scenes/x.json', 400],
      ['scenes\\x.json', 400],
      ['scenes/x.JSON', 200], // 大小写不敏感，仍写入 scenes/x.JSON（位置合法）
      ['assets/evil.json', 403],
      ['packages/scene/examples/../../../../secret.json', 400],
      ['scenes/x.txt', 400],
    ];
    for (const [p, status] of badPaths) {
      const r = await postJson(`${base}/__charaforge/editor/save-scene`, { path: p, scene: tinyScene() });
      assert.equal(r.status, status, `path=${p}`);
      if (status !== 200) assert.equal(r.body.ok, false);
    }
    assert.equal(JSON.parse(readFileSync(join(root, 'secret.json'), 'utf8')).meta.title, 'secret');
  } finally {
    await stop(server);
  }
});

test('场景 symlink 逃出仓库时拒绝写入，且列表不跟随', async (t) => {
  const root = fixture();
  const outside = mkdtempSync(join(tmpdir(), 'charaforge-editor-outside-'));
  cleanups.push(outside);
  writeFileSync(join(outside, 'outside.json'), `${JSON.stringify(tinyScene('outside'), null, 2)}\n`);
  const link = join(root, 'scenes', 'escape.json');
  try {
    symlinkSync(join(outside, 'outside.json'), link, 'file');
  } catch {
    t.skip('当前 Windows 账户无法创建 symlink');
    return;
  }
  const { base, server } = await start(root);
  try {
    const r = await postJson(`${base}/__charaforge/editor/save-scene`, {
      path: 'scenes/escape.json',
      scene: tinyScene('escaped'),
    });
    assert.equal(r.status, 403);
    assert.equal(JSON.parse(readFileSync(join(outside, 'outside.json'), 'utf8')).meta.title, 'outside');

    const list = await fetchJson(`${base}/__charaforge/editor/scenes`);
    assert.ok(!list.body.scenes.some((s: { path: string }) => s.path === 'scenes/escape.json'));
  } finally {
    await stop(server);
  }
});

test('编辑命令：新增图片图层/关键帧、diff、preview、out 落盘', async () => {
  const root = fixture();
  const { base, server } = await start(root);
  try {
    const scene = tinyScene('command');
    const commands = [
      {
        op: 'addImageLayer',
        asset: { id: 'hat', uri: 'editor/My-Hat.png', width: 1, height: 1 },
        layer: {
          id: 'hat.layer',
          kind: 'image',
          image: 'hat',
          in: 0,
          out: 60,
          transform: { x: 160, y: 90, scale: 1 },
        },
      },
      {
        op: 'setKeyframes',
        target: { kind: 'layer', id: 'hat.layer', path: 'opacity' },
        keys: [
          { t: 0, v: 0 },
          { t: 30, v: 1 },
        ],
        ease: 'linear',
      },
    ];
    const r = await postJson(`${base}/__charaforge/editor/command`, {
      sceneObject: scene,
      commands,
      out: 'scenes/command.out.json',
      frame: 30,
    });
    assert.equal(r.status, 200);
    assert.equal(r.body.ok, true);
    assert.equal(r.body.changed, true);
    assert.ok(Array.isArray(r.body.diff));
    assert.ok(r.body.diff.length >= 3);
    const paths = r.body.diff.map((c: { path: string }) => c.path);
    assert.ok(paths.includes('/assets/images/0'));
    assert.ok(paths.includes('/layers/0'));
    assert.ok(paths.some((p: string) => p.startsWith('/layers/0/opacity')));
    assert.ok(r.body.effects.includes('render'));
    assert.equal(r.body.preview.frame, 30);
    assert.equal(r.body.preview.activeLayers.length, 1);
    assert.equal(r.body.preview.activeLayers[0].id, 'hat.layer');
    assert.equal(r.body.preview.activeLayers[0].opacity, 1);
    assert.equal(r.body.scene.layers.length, 1);
    assert.equal(r.body.scene.layers[0].kind, 'image');
    assert.deepEqual(r.body.scene.layers[0].opacity.keys, [
      { t: 0, v: 0, ease: 'linear' },
      { t: 30, v: 1, ease: 'linear' },
    ]);
    assert.equal(r.body.out, 'scenes/command.out.json');
    assert.ok(existsSync(join(root, 'scenes', 'command.out.json')));
    assert.equal(JSON.parse(readFileSync(join(root, 'scenes', 'command.out.json'), 'utf8')).layers.length, 1);

    const viaPath = await postJson(`${base}/__charaforge/editor/command`, {
      scene: 'scenes/demo.scene.json',
      commands: { op: 'setScene', patch: { title: 'via path' } },
      dryRun: true,
    });
    assert.equal(viaPath.status, 200);
    assert.equal(viaPath.body.changed, true);
    assert.equal(viaPath.body.scene.meta.title, 'via path');
    assert.equal(viaPath.body.out, null);
    assert.equal(JSON.parse(readFileSync(join(root, 'scenes/demo.scene.json'), 'utf8')).meta.title, 'demo');

    const traversed = await postJson(`${base}/__charaforge/editor/command`, {
      scene: 'scenes/../secret.json',
      commands: { op: 'noop' },
    });
    assert.equal(traversed.status, 400);
  } finally {
    await stop(server);
  }
});
