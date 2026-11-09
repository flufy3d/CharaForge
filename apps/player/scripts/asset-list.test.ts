/*
 * 只读素材列表接口测试：
 *   - 目录/文件列表返回形状与排序
 *   - 路径穿越、反斜杠、绝对路径、盘符、非允许目录、空 dir 拒绝
 *   - 目录内 symlink 不列出、目录本身是 symlink（指向 assets/models/ 外）拒绝
 *   - 真实仓库 assets/ 存在时，Animation 8 个 motion3、Exp3 23 个 exp3
 *
 *   pnpm --filter @charaforge/player test:viewer
 */
import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, readdirSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { createServer, type Server } from 'node:http';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { after, test } from 'node:test';
import { createAssetListHandler, listAssetFiles, normalizeAssetDir } from '../server/assetList.ts';

const REPO = resolve(import.meta.dirname, '..', '..', '..');
const cleanups: string[] = [];

function fixture(): string {
  const root = mkdtempSync(join(tmpdir(), 'charaforge-list-'));
  cleanups.push(root);
  mkdirSync(join(root, 'assets', 'models', 'demo', 'Animation'), { recursive: true });
  mkdirSync(join(root, 'assets', 'models', 'demo', 'Exp3'), { recursive: true });
  mkdirSync(join(root, 'scenes'), { recursive: true });
  writeFileSync(join(root, 'assets', 'models', 'demo', 'Animation', 'a.motion3.json'), '{}');
  writeFileSync(join(root, 'assets', 'models', 'demo', 'Animation', 'b.motion3.json'), 'x'.repeat(10));
  writeFileSync(join(root, 'assets', 'models', 'demo', 'Exp3', '脸红.exp3.json'), '{}');
  writeFileSync(join(root, 'scenes', 'secret.json'), 'SECRET');
  return root;
}

async function start(root: string): Promise<{ base: string; server: Server }> {
  const handler = createAssetListHandler(root);
  const server = createServer((req, res) => {
    if (!handler(req, res)) {
      res.statusCode = 404;
      res.end('next');
    }
  });
  await new Promise<void>((ok) => server.listen(0, '127.0.0.1', ok));
  const addr = server.address();
  if (!addr || typeof addr === 'string') throw new Error('服务没有监听端口');
  return { base: `http://127.0.0.1:${addr.port}`, server };
}

async function get(base: string, dir: string): Promise<{ status: number; body: { dir?: string; files?: { name: string; size: number; mtime: string }[]; error?: string } }> {
  const res = await fetch(`${base}/__charaforge/list-assets?dir=${encodeURIComponent(dir)}`);
  return { status: res.status, body: (await res.json()) as { dir?: string; files?: { name: string; size: number; mtime: string }[]; error?: string } };
}

after(() => {
  for (const d of cleanups) rmSync(d, { recursive: true, force: true });
});

test('normalizeAssetDir：只允许 assets/models/ 下的 Animation / Exp3', () => {
  assert.equal(normalizeAssetDir('assets/models/demo/Animation'), 'assets/models/demo/Animation');
  assert.equal(normalizeAssetDir('assets/models/a/b/Exp3'), 'assets/models/a/b/Exp3');
  for (const bad of [
    '',
    '/assets/models/demo/Animation',
    'assets\\models\\demo\\Animation',
    'C:/assets/models/demo/Animation',
    'assets/models/demo/Animation/../Exp3',
    'assets/models/../../scenes',
    'assets/models/demo/Animation/',
    'assets/models/demo/./Animation',
    'assets/models/demo',
    'assets/models/demo/Animation2',
    'scenes',
  ]) {
    assert.throws(() => normalizeAssetDir(bad), /./, `应拒绝：${bad}`);
  }
});

test('列表接口返回排序后的文件与 ISO mtime', async () => {
  const root = fixture();
  const { base, server } = await start(root);
  try {
    const r = await get(base, 'assets/models/demo/Animation');
    assert.equal(r.status, 200);
    assert.equal(r.body.dir, 'assets/models/demo/Animation');
    assert.deepEqual(r.body.files?.map((f) => f.name), ['a.motion3.json', 'b.motion3.json']);
    assert.equal(r.body.files?.[0]!.size, 2);
    assert.equal(r.body.files?.[1]!.size, 10);
    assert.ok(!Number.isNaN(Date.parse(r.body.files![0]!.mtime)));
  } finally {
    await new Promise<void>((ok) => server.close(() => ok()));
  }
});

test('中文文件名正常返回', async () => {
  const root = fixture();
  const { base, server } = await start(root);
  try {
    const r = await get(base, 'assets/models/demo/Exp3');
    assert.equal(r.status, 200);
    assert.deepEqual(r.body.files?.map((f) => f.name), ['脸红.exp3.json']);
  } finally {
    await new Promise<void>((ok) => server.close(() => ok()));
  }
});

test('路径穿越 / 非法目录被拒绝', async () => {
  const root = fixture();
  const { base, server } = await start(root);
  try {
    const bad = [
      'assets/models/demo/Animation/../Exp3',
      'assets/models/../../scenes',
      '/assets/models/demo/Animation',
      'assets\\models\\demo\\Animation',
      'C:/assets/models/demo/Animation',
      'assets/models/demo',
      'assets/models/demo/Animation2',
      'assets/models/demo/Animation/',
    ];
    for (const dir of bad) {
      const r = await get(base, dir);
      assert.ok(r.status === 400 || r.status === 403, `${dir} 应被拒绝，实际 ${r.status}`);
      assert.equal(typeof r.body.error, 'string');
    }
    // 双重编码的 .. 也按解码后的 .. 拒绝
    const res = await fetch(`${base}/__charaforge/list-assets?dir=assets/models/demo/Animation/%2e%2e/Exp3`);
    assert.equal(res.status, 400);
  } finally {
    await new Promise<void>((ok) => server.close(() => ok()));
  }
});

test('非 GET 返回 405', async () => {
  const root = fixture();
  const { base, server } = await start(root);
  try {
    const res = await fetch(`${base}/__charaforge/list-assets?dir=assets/models/demo/Animation`, { method: 'POST' });
    assert.equal(res.status, 405);
  } finally {
    await new Promise<void>((ok) => server.close(() => ok()));
  }
});

test('目录内的 symlink 不列出', async (t) => {
  const root = fixture();
  const link = join(root, 'assets', 'models', 'demo', 'Animation', 'escape.motion3.json');
  try {
    symlinkSync(join(root, 'scenes', 'secret.json'), link, 'file');
  } catch {
    t.skip('当前 Windows 账户无法创建 symlink');
    return;
  }
  const { base, server } = await start(root);
  try {
    const r = await get(base, 'assets/models/demo/Animation');
    assert.equal(r.status, 200);
    assert.ok(!r.body.files?.some((f) => f.name === 'escape.motion3.json'));
  } finally {
    await new Promise<void>((ok) => server.close(() => ok()));
  }
});

test('目录本身是 symlink 且指向 assets/models/ 外时拒绝', async (t) => {
  const root = fixture();
  const exp3 = join(root, 'assets', 'models', 'demo', 'Exp3');
  rmSync(exp3, { recursive: true, force: true });
  try {
    symlinkSync(join(root, 'scenes'), exp3, 'dir');
  } catch {
    t.skip('当前 Windows 账户无法创建 symlink');
    return;
  }
  const { base, server } = await start(root);
  try {
    const r = await get(base, 'assets/models/demo/Exp3');
    assert.equal(r.status, 403);
    assert.match(r.body.error ?? '', /越出/);
  } finally {
    await new Promise<void>((ok) => server.close(() => ok()));
  }
});

test('真实仓库：Animation 8 个 motion3、Exp3 23 个 exp3', (t) => {
  const animation = join(REPO, 'assets', 'models', 'deepseek-chan', 'DeepSeek', 'Animation');
  const exp3 = join(REPO, 'assets', 'models', 'deepseek-chan', 'DeepSeek', 'Exp3');
  let names: string[];
  try {
    names = readdirSync(animation);
  } catch {
    t.skip('assets/ 未安装（先运行 pnpm setup:assets）');
    return;
  }
  assert.equal(names.filter((n) => n.endsWith('.motion3.json')).length, 8);
  assert.equal(readdirSync(exp3).filter((n) => n.endsWith('.exp3.json')).length, 23);
  const list = listAssetFiles(REPO, 'assets/models/deepseek-chan/DeepSeek/Animation');
  assert.equal(list.files.filter((f) => f.name.endsWith('.motion3.json')).length, 8);
});
