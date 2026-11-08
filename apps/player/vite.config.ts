/*
 * 播放器 / harness 的 Vite 配置。
 *
 * 开发服务器通过中间件只读地暴露仓库里的几个目录（白名单），URL 路径 = 仓库相对路径：
 *   /assets/...                  模型、视频、音频（setup:assets 安装）
 *   /vendor/cubism/core/...      Cubism Core（页面用 <script> 引入）
 *   /scenes/...                  场景与镜头表
 *   /packages/scene/examples/... 示例场景
 * 支持 Range 请求（<video>/<audio> 精确 seek 需要）。
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

export function repoFiles(): Plugin {
  const handler: Connect.NextHandleFunction = (req, res, next) => {
    const url = new URL(req.url ?? '/', 'http://x');
    let rel: string;
    try {
      rel = decodeURIComponent(url.pathname).replace(/^\/+/, '');
    } catch {
      return next();
    }
    if (!MOUNTS.some((m) => rel.startsWith(m))) return next();
    const norm = normalize(rel);
    // 规范化之后必须仍在白名单目录内（拒绝 assets/../ 之类的穿越）
    if (norm.split(sep).includes('..') || !MOUNTS.some((m) => norm.startsWith(m))) {
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
  plugins: [repoFiles()],
  resolve: {
    alias: [{ find: /^@cubism\/framework\/(.*)$/, replacement: `${join(REPO, '.cache', 'cubism-framework')}/$1` }],
  },
  optimizeDeps: {
    entries: ['index.html', 'harness.html'],
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
      input: { index: join(HERE, 'index.html'), harness: join(HERE, 'harness.html') },
    },
  },
});
