/*
 * 只读素材目录接口（dev / preview 共用，由 vite.config.ts 的 repoFiles() 挂载）。
 *
 *   GET /__charaforge/list-assets?dir=assets/models/deepseek-chan/DeepSeek/Animation
 *   → { "dir": "...", "files": [{ "name": "IDLE.motion3.json", "size": 35673, "mtime": "2026-..." }] }
 *
 * 只允许 assets/models/ 下、末级目录为 Animation 或 Exp3 的目录。目录内容只列出文件名/大小/修改时间，
 * 不读取、不执行文件内容。路径逐段校验，并在 realpath 之后再次确认没有越出 assets/models/；
 * 目录内的 symlink 文件直接跳过，防止借 symlink 暴露仓库外文件。
 *
 * 这个接口是为了解决“model3.json 没有声明 Motions，但动作文件在 Animation/ 下”的问题；
 * 不生成 index 文件，assets/ 不进 git 时也不会过期。
 */
import { readdirSync, realpathSync, statSync } from 'node:fs';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { join, posix, resolve, sep } from 'node:path';

export const ASSET_LIST_PATH = '/__charaforge/list-assets';

const MODELS_DIR = 'assets/models';
const ALLOWED_LEAVES = new Set(['Animation', 'Exp3']);

export interface AssetFileInfo {
  name: string;
  size: number;
  mtime: string;
}

export interface AssetListResult {
  dir: string;
  files: AssetFileInfo[];
}

export class AssetListError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'AssetListError';
    this.status = status;
  }
}

/**
 * 校验查询参数里的仓库相对目录。只做字符串层面的检查，不访问文件系统：
 * 拒绝绝对路径、盘符、反斜杠、NUL、空段、`.` / `..`，并要求位于 assets/models/ 下且末级是 Animation / Exp3。
 * 返回值是规范化后的仓库相对路径（始终用 `/`）。
 */
export function normalizeAssetDir(dir: string): string {
  if (!dir) throw new AssetListError(400, '缺少 dir 参数');
  if (dir.includes('\0')) throw new AssetListError(400, 'dir 不能包含 NUL');
  if (dir.includes('\\')) throw new AssetListError(400, 'dir 必须使用 / 分隔');
  if (dir.startsWith('/')) throw new AssetListError(400, 'dir 必须是仓库相对路径');
  if (/^[a-zA-Z]:/.test(dir)) throw new AssetListError(400, 'dir 不能是绝对路径');
  const segments = dir.split('/');
  if (segments.some((s) => s === '' || s === '.' || s === '..')) {
    throw new AssetListError(400, 'dir 不能包含空段、. 或 ..');
  }
  const norm = posix.normalize(dir);
  if (norm !== dir) throw new AssetListError(400, 'dir 不是规范路径');
  if (!norm.startsWith(`${MODELS_DIR}/`)) throw new AssetListError(403, `只允许 ${MODELS_DIR}/ 下的目录`);
  const leaf = norm.slice(norm.lastIndexOf('/') + 1);
  if (!ALLOWED_LEAVES.has(leaf)) throw new AssetListError(403, '只允许 Animation/ 和 Exp3/ 目录');
  return norm;
}

/** 列出允许目录下的普通文件（跳过 symlink 与子目录）。repoRoot 是仓库根目录。 */
export function listAssetFiles(repoRoot: string, dir: string): AssetListResult {
  const norm = normalizeAssetDir(dir);
  let modelsReal: string;
  try {
    modelsReal = realpathSync.native(resolve(repoRoot, MODELS_DIR));
  } catch {
    throw new AssetListError(404, `${MODELS_DIR}/ 不存在（请先运行 pnpm setup:assets）`);
  }
  let targetReal: string;
  try {
    targetReal = realpathSync.native(resolve(repoRoot, norm));
  } catch {
    throw new AssetListError(404, `目录不存在：${norm}`);
  }
  const prefix = modelsReal.endsWith(sep) ? modelsReal : modelsReal + sep;
  if (targetReal !== modelsReal && !targetReal.startsWith(prefix)) {
    throw new AssetListError(403, `目录越出 ${MODELS_DIR}/：${norm}`);
  }
  let stat;
  try {
    stat = statSync(targetReal);
  } catch {
    throw new AssetListError(404, `目录不存在：${norm}`);
  }
  if (!stat.isDirectory()) throw new AssetListError(404, `不是目录：${norm}`);

  const files: AssetFileInfo[] = [];
  for (const entry of readdirSync(targetReal, { withFileTypes: true })) {
    // 目录内的 symlink 一律不列出：即使它指向 assets/models/ 内部，也没有必要让只读列表跟着跳转。
    if (!entry.isFile() || entry.isSymbolicLink()) continue;
    const path = join(targetReal, entry.name);
    let st;
    try {
      st = statSync(path);
    } catch {
      continue;
    }
    if (!st.isFile()) continue;
    files.push({ name: entry.name, size: st.size, mtime: st.mtime.toISOString() });
  }
  files.sort((a, b) => a.name.localeCompare(b.name, 'en'));
  return { dir: norm, files };
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.statusCode = status;
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store');
  res.setHeader('Content-Length', String(Buffer.byteLength(text)));
  res.end(text);
}

/**
 * 返回一个 Connect 风格的处理器：命中接口时处理并返回 true，否则返回 false 交给后续中间件。
 * 只读，只接受 GET；错误也以 JSON 返回，便于页面和测试读取。
 */
export function createAssetListHandler(repoRoot: string): (req: IncomingMessage, res: ServerResponse) => boolean {
  return (req, res) => {
    let url: URL;
    try {
      url = new URL(req.url ?? '/', 'http://x');
    } catch {
      return false;
    }
    if (url.pathname !== ASSET_LIST_PATH) return false;
    if (req.method !== 'GET') {
      sendJson(res, 405, { error: 'method not allowed（只支持 GET）' });
      return true;
    }
    try {
      const dir = url.searchParams.get('dir') ?? '';
      sendJson(res, 200, listAssetFiles(repoRoot, dir));
    } catch (e) {
      if (e instanceof AssetListError) sendJson(res, e.status, { error: e.message });
      else sendJson(res, 500, { error: String((e as Error)?.message ?? e) });
    }
    return true;
  };
}
