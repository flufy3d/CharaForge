/*
 * 播放器 / harness 的 Vite 配置。
 *
 * 开发服务器通过中间件只读地暴露仓库里的几个目录（白名单），URL 路径 = 仓库相对路径：
 *   /assets/...                  模型、视频、音频（setup:assets 安装）
 *   /vendor/cubism/core/...      Cubism Core（页面用 <script> 引入）
 *   /scenes/...                  场景与镜头表
 *   /packages/scene/examples/... 示例场景
 *   /__charaforge/list-assets   动作/表情查看器的只读目录列表（仅 assets/models/.../Animation 或 Exp3）
 *   /__charaforge/editor/...    本地编辑接口（只允许 loopback；见 server/editorApi.ts）
 * 支持 Range 请求（<video>/<audio> 精确 seek 需要）。
 * 路径段可以写成 `_f.<base64url(UTF-8)>`（ASCII 安全的路由，给中文文件名用，见 decodePath），解码后照常做白名单与防穿越检查。
 *
 * pnpm build 的产物（apps/player/dist/）只含页面和打包后的代码（含 Cubism Framework 编译结果），
 * 不含 Cubism Core、模型、视频、音频、场景：这些仍按上面的 URL 从仓库目录读取。
 * 所以 dist/ 必须由带同一白名单的服务器提供：pnpm preview（vite preview，复用本中间件）。
 */
import { createReadStream, existsSync, statSync } from 'node:fs';
import { dirname, extname, join, normalize, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import type { Connect, Plugin } from 'vite';
import { defineConfig } from 'vite';
import { createAssetListHandler } from './server/assetList.ts';
import { createEditorApi } from './server/editorApi.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
export const REPO = resolve(HERE, '..', '..');
const MOUNTS = ['assets/', 'vendor/cubism/core/', 'scenes/', 'packages/scene/examples/', 'packages/scene/schema/'];

const TYPES: Record<string, string> = {
  '.json': 'application/json; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.wav': 'audio/wav',
  '.mp4': 'video/mp4',
  '.webm': 'video/webm',
  '.moc3': 'application/octet-stream',
  '.txt': 'text/plain; charset=utf-8',
};

/** 与 packages/runtime/src/asciiUrl.ts 的 ASCII_SEGMENT_PREFIX 一致 */
const ASCII_SEGMENT_PREFIX = '_f.';

/**
 * URL 路径 → 仓库相对路径。逐段百分号解码；`_f.<base64url>` 段解码回 UTF-8 原名（ASCII 安全的路由：
 * 文件名是中文时，播放器把这一段编码成 base64url，避免反向代理把路径解码成非 ASCII 后转发出错，见 asciiUrl.ts）。
 * 解码出的段里不允许出现 / \ NUL，也不允许是 . 或 ..（不能借编码绕过下面的白名单与防穿越检查）；非法时抛错。
 */
function decodePath(pathname: string): string {
  return pathname
    .split('/')
    .map((seg) => {
      let d = decodeURIComponent(seg);
      if (d.startsWith(ASCII_SEGMENT_PREFIX)) {
        const b64 = d.slice(ASCII_SEGMENT_PREFIX.length);
        if (!/^[A-Za-z0-9_-]+$/.test(b64)) throw new Error('bad segment');
        d = Buffer.from(b64, 'base64url').toString('utf8');
        if (!d || /[/\\\0]/.test(d) || d === '.' || d === '..') throw new Error('bad segment');
      }
      return d;
    })
    .join('/');
}

export function repoFiles(): Plugin {
  const assetList = createAssetListHandler(REPO);
  // 本地编辑接口：只接受 loopback，处理 /__charaforge/editor/...；非该前缀返回 false。
  const editorApi = createEditorApi(REPO);
  const handler: Connect.NextHandleFunction = (req, res, next) => {
    // 编辑接口必须在静态白名单之前；它自己会拒绝非本机请求。
    if (editorApi(req, res)) return;
    // 只读素材目录接口（动作/表情查看器用）：必须在 MOUNTS 白名单之前处理。
    if (assetList(req, res)) return;
    const url = new URL(req.url ?? '/', 'http://x');
    let rel: string;
    try {
      rel = decodePath(url.pathname).replace(/^\/+/, '');
    } catch {
      res.statusCode = 400;
      return res.end('bad path');
    }
    if (!MOUNTS.some((m) => rel.startsWith(m))) return next();
    const norm = normalize(rel);
    // MOUNTS 用 / 书写；Windows 上 normalize 会返回 \，比较前先统一。
    const mountRel = norm.split(sep).join('/');
    // 规范化之后必须仍在白名单目录内（拒绝 assets/../ 之类的穿越）
    if (norm.split(sep).includes('..') || !MOUNTS.some((m) => mountRel.startsWith(m))) {
      res.statusCode = 403;
      return res.end();
    }
    const file = join(REPO, norm);
    if (!existsSync(file) || !statSync(file).isFile()) {
      res.statusCode = 404;
      return res.end(`not found: ${rel}`);
    }
    const size = statSync(file).size;
    res.setHeader('Content-Type', TYPES[extname(file).toLowerCase()] ?? 'application/octet-stream');
    res.setHeader('Accept-Ranges', 'bytes');
    res.setHeader('Cache-Control', 'no-cache');
    const range = req.headers.range && /bytes=(\d*)-(\d*)/.exec(req.headers.range);
    if (range) {
      const start = range[1] ? Number(range[1]) : size - Number(range[2]);
      const end = range[1] && range[2] ? Math.min(Number(range[2]), size - 1) : size - 1;
      if (start >= size || start > end) {
        res.statusCode = 416;
        res.setHeader('Content-Range', `bytes */${size}`);
        return res.end();
      }
      res.statusCode = 206;
      res.setHeader('Content-Range', `bytes ${start}-${end}/${size}`);
      res.setHeader('Content-Length', String(end - start + 1));
      if (req.method === 'HEAD') return res.end();
      return void createReadStream(file, { start, end }).pipe(res);
    }
    res.setHeader('Content-Length', String(size));
    if (req.method === 'HEAD') return res.end();
    createReadStream(file).pipe(res);
  };
  return {
    name: 'charaforge-repo-files',
    configureServer(server) {
      server.middlewares.use(handler);
    },
    configurePreviewServer(server) {
      server.middlewares.use(handler);
    },
  };
}

export default defineConfig({
  root: HERE,
  // 相对 base：build 产物里的脚本、样式、Worker 都用相对路径，页面挂在任意路径前缀下（反向代理）也能用。
  // 运行时请求（assets/、scenes/、vendor/…）由页面代码相对 document.baseURI 解析（apps/player/src/util.ts 的 pageUrl）。
  base: './',
  plugins: [repoFiles()],
  resolve: {
    alias: [{ find: /^@cubism\/framework\/(.*)$/, replacement: `${join(REPO, '.cache', 'cubism-framework')}/$1` }],
  },
  optimizeDeps: {
    entries: ['index.html', 'player.html', 'harness.html', 'motions.html', 'editor.html', 'export.html'],
    include: ['pixi.js', 'react', 'react-dom/client', 'react/jsx-runtime', 'react/jsx-dev-runtime'],
  },
  server: {
    port: 5173,
    strictPort: false,
    fs: { allow: [REPO] },
  },
  // pnpm preview：与 dev 同一个白名单中间件（configurePreviewServer），dist/ 只有页面与打包后的代码
  preview: {
    port: 4173,
    strictPort: false,
  },
  build: {
    target: 'es2022',
    // 打包产物放 _app/：缺省的 assets/ 会和白名单里的 /assets/（模型、视频、音频）抢同一个 URL 前缀
    assetsDir: '_app',
    rollupOptions: {
      input: {
        // index = 默认编辑器；player 保留独立播放器/参考视频对照页。
        index: join(HERE, 'index.html'),
        player: join(HERE, 'player.html'),
        harness: join(HERE, 'harness.html'),
        motions: join(HERE, 'motions.html'),
        editor: join(HERE, 'editor.html'),
        export: join(HERE, 'export.html'),
      },
    },
  },
});
