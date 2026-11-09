/*
 * 本地编辑 HTTP 接口（dev / preview 共用，由 vite.config.ts 的 repoFiles() 挂载）。
 *
 * 只给本机的人类、脚本和 AI 使用：仅接受 loopback 客户端，不做跨域开放。
 * 路由（全部以 /__charaforge/editor 开头）：
 *   GET  /ping                        → { ok, version }
 *   GET  /scenes                      → 场景文件列表（scenes/ 深度 ≤ 3 + packages/scene/examples/）
 *   GET  /scene?path=...              → 读取并校验一个场景（只读）
 *   GET  /assets?kind=image|audio     → assets/editor/ 下的素材列表（图片含尺寸）
 *   POST /upload?kind=image|audio&name=...   原始字节上传，按文件头判定类型
 *   POST /save-scene                 → 校验并写入场景 JSON（可选 .bak.json 备份）
 *   POST /command                    → 应用编辑命令并返回新场景/diff/preview
 *   POST /export                     → 501 占位（真正的导出在浏览器/渲染流水线）
 *
 * 安全约定：
 *   - 只接受 127.0.0.1 / ::1 / ::ffff:127.0.0.1（测试里 socket 可能没有 remoteAddress）；
 *     带 Origin 时也只允许 localhost / 127.0.0.1 / ::1，避免任意网页借浏览器改仓库文件。
 *   - 所有路径都是仓库相对路径：拒绝绝对路径、盘符、反斜杠、NUL、空段、`.` / `..`、
 *     Windows 设备名，并在 realpath 之后再次确认没有借 symlink 越出仓库。
 *   - 上传和 JSON 请求体都有硬上限；内容只写文件，从不执行。
 *   - 错误一律返回 { ok:false, error }；未知异常只记服务端日志，不把堆栈发给客户端。
 */
import { createHash } from 'node:crypto';
import {
  closeSync,
  copyFileSync,
  existsSync,
  lstatSync,
  mkdirSync,
  openSync,
  readSync,
  readdirSync,
  readFileSync,
  realpathSync,
  statSync,
  writeFileSync,
} from 'node:fs';
import type { IncomingMessage, ServerResponse } from 'node:http';
import { dirname, extname, isAbsolute, join, posix, relative, resolve, sep } from 'node:path';
import type { Connect } from 'vite';
import { applyEditCommands, parseScene, previewAt } from '@charaforge/scene';
import type { EditCommand, EditResult, Scene } from '@charaforge/scene';

export const EDITOR_API_PREFIX = '/__charaforge/editor';

/** 场景 JSON 请求体上限：全片场景约 4 MB，留足余量。 */
const JSON_BODY_LIMIT = 64 * 1024 * 1024;
/** 图片/音频上传上限。 */
const UPLOAD_BODY_LIMIT = 32 * 1024 * 1024;

const SCENE_DIRS = ['scenes/', 'packages/scene/examples/'] as const;
const EDITOR_ASSETS_DIR = 'assets/editor';
const MAX_SCENE_DEPTH = 3;

/** 与 packages/scene 的 ImageAsset 限制一致。 */
const MAX_IMAGE_SIDE = 4096;
const MAX_IMAGE_PIXELS = 16_777_216;

const IMAGE_EXTENSIONS = new Set(['.png', '.webp']);
const AUDIO_EXTENSIONS = new Set(['.wav', '.mp3', '.ogg', '.m4a', '.mp4']);

const WINDOWS_RESERVED_NAME = /^(con|prn|aux|nul|com[1-9]|lpt[1-9])(\..*)?$/i;

export interface SceneFileInfo {
  path: string;
  size: number;
  mtime: string;
}

export interface EditorAssetInfo {
  id: string;
  uri: string;
  width?: number;
  height?: number;
  size: number;
  mtime: string;
  name: string;
}

export class EditorApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'EditorApiError';
    this.status = status;
  }
}

/** loopback 判定。socket 不存在/remoteAddress 缺失（mock、测试）按本机处理。 */
export function isLoopbackAddress(address: string | null | undefined): boolean {
  return (
    address === undefined ||
    address === null ||
    address === '' ||
    address === '127.0.0.1' ||
    address === '::1' ||
    address === '::ffff:127.0.0.1'
  );
}

function isAllowedOrigin(origin: string | undefined): boolean {
  if (!origin) return true;
  try {
    const host = new URL(origin).hostname.toLowerCase();
    return host === 'localhost' || host === '127.0.0.1' || host === '::1' || host === '[::1]';
  } catch {
    return false;
  }
}

function errText(e: unknown): string {
  return String((e as Error)?.message ?? e);
}

function isInside(root: string, target: string): boolean {
  const rel = relative(root, target);
  return rel === '' || (rel !== '..' && !rel.startsWith(`..${sep}`) && !isAbsolute(rel));
}

/**
 * 仓库相对路径的字符串级校验。返回值仍用 `/` 分隔，且保证已经规范化。
 * 除绝对路径/盘符/反斜杠/NUL/空段/`.`/`..` 外，也拒绝 Windows 设备名和尾随点/空格，
 * 避免同一个路径在 Windows 上被解释成别处。
 */
export function normalizeRepoRelPath(raw: unknown, what = 'path'): string {
  if (typeof raw !== 'string' || raw.length === 0) throw new EditorApiError(400, `${what} 必须是非空字符串`);
  if (raw.length > 512) throw new EditorApiError(400, `${what} 过长`);
  if (/[\0-\x1f<>"|?*\\:]/.test(raw)) throw new EditorApiError(400, `${what} 包含非法字符`);
  if (raw.startsWith('/')) throw new EditorApiError(400, `${what} 必须是仓库相对路径`);
  if (posix.normalize(raw) !== raw) throw new EditorApiError(400, `${what} 不是规范路径`);
  const segments = raw.split('/');
  if (segments.some((s) => s === '' || s === '.' || s === '..')) {
    throw new EditorApiError(400, `${what} 不能包含空段、. 或 ..`);
  }
  for (const s of segments) {
    if (s.endsWith('.') || s.endsWith(' ')) throw new EditorApiError(400, `${what} 的段不能以点或空格结尾`);
    if (WINDOWS_RESERVED_NAME.test(s)) throw new EditorApiError(400, `${what} 包含 Windows 保留名`);
  }
  return raw;
}

/** 场景路径白名单：scenes/ 或 packages/scene/examples/ 下的 .json。 */
export function normalizeScenePath(raw: unknown, what = 'path'): string {
  const rel = normalizeRepoRelPath(raw, what);
  if (!rel.toLowerCase().endsWith('.json')) throw new EditorApiError(400, `${what} 必须以 .json 结尾`);
  if (!SCENE_DIRS.some((d) => rel.startsWith(d))) {
    throw new EditorApiError(403, `${what} 只允许 ${SCENE_DIRS.join(' 或 ')} 下`);
  }
  return rel;
}

/** repo 的真实路径（解析 repo 自身可能是 symlink 的情况）。 */
function repoRealPath(repo: string): string {
  try {
    return realpathSync.native(resolve(repo));
  } catch {
    return resolve(repo);
  }
}

/**
 * 确认 abs 及其最近的存在祖先在 repoReal 内。覆盖两种情况：
 *   - abs 已存在：realpath 会解析整条路径上的 symlink；
 *   - abs 不存在：向上找到最近的存在目录，realpath 之后仍必须在仓库内。
 * 悬空 symlink 会被视为越界（写文件时可能把内容建到仓库外）。
 */
function assertRealInside(repoReal: string, abs: string, what: string): void {
  let probe = abs;
  for (;;) {
    try {
      const st = lstatSync(probe);
      if (st.isSymbolicLink()) {
        let real: string;
        try {
          real = realpathSync.native(probe);
        } catch {
          throw new EditorApiError(403, `${what} 是悬空符号链接`);
        }
        if (!isInside(repoReal, real)) throw new EditorApiError(403, `${what} 的符号链接越出仓库`);
      }
      if (!isInside(repoReal, realpathSync.native(probe))) {
        throw new EditorApiError(403, `${what} 越出仓库`);
      }
      return;
    } catch (e) {
      if (e instanceof EditorApiError) throw e;
      const code = (e as NodeJS.ErrnoException).code;
      if (code !== 'ENOENT' && code !== 'ENOTDIR') throw new EditorApiError(403, `${what} 无法访问`);
      const parent = dirname(probe);
      if (parent === probe) throw new EditorApiError(403, `${what} 越出仓库`);
      probe = parent;
    }
  }
}

/** 读取已有文件前的检查：路径在白名单内、实际文件存在且没有越出仓库。 */
function resolveExistingFile(repoReal: string, rel: string, what: string): string {
  const abs = resolve(repoReal, rel);
  assertRealInside(repoReal, abs, what);
  let st;
  try {
    st = statSync(abs);
  } catch {
    throw new EditorApiError(404, `${what} 不存在：${rel}`);
  }
  if (!st.isFile()) throw new EditorApiError(404, `${what} 不是文件：${rel}`);
  return abs;
}

/**
 * 写文件前的检查：拒绝目录/悬空 symlink，必要时创建父目录，并在 mkdir 后再核对一次 realpath。
 * 返回绝对路径。
 */
function prepareWriteFile(repoReal: string, rel: string, what: string): string {
  const abs = resolve(repoReal, rel);
  assertRealInside(repoReal, abs, what);
  try {
    const st = lstatSync(abs);
    if (st.isDirectory()) throw new EditorApiError(400, `${what} 是目录：${rel}`);
  } catch (e) {
    if (e instanceof EditorApiError) throw e;
    if ((e as NodeJS.ErrnoException).code !== 'ENOENT') throw new EditorApiError(403, `${what} 无法访问`);
  }
  const dir = dirname(abs);
  try {
    mkdirSync(dir, { recursive: true });
    if (!isInside(repoReal, realpathSync.native(dir))) {
      throw new EditorApiError(403, `${what} 的父目录越出仓库`);
    }
  } catch (e) {
    if (e instanceof EditorApiError) throw e;
    throw new EditorApiError(500, `无法创建 ${what} 的父目录`);
  }
  return abs;
}

function validateScene(v: unknown): Scene {
  try {
    return parseScene(v);
  } catch (e) {
    throw new EditorApiError(400, errText(e));
  }
}

function readSceneFile(repoReal: string, rel: string): Scene {
  const abs = resolveExistingFile(repoReal, rel, '场景');
  let parsed: unknown;
  try {
    parsed = JSON.parse(readFileSync(abs, 'utf8'));
  } catch (e) {
    throw new EditorApiError(400, `场景 JSON 解析失败：${errText(e)}`);
  }
  return validateScene(parsed);
}

/**
 * 场景写盘格式：小型示例保持 2 空格可读；全片场景（图层多/参数轨多）用紧凑 JSON，
 * 避免 4 MB 的场景被展开成 14 MB。
 */
function sceneJsonText(scene: Scene): string {
  const params = scene.layers.reduce((n, layer) => n + (layer.kind === 'model' ? layer.params?.length ?? 0 : 0), 0);
  const compact = scene.layers.length > 20 || params > 2000;
  return `${JSON.stringify(scene, null, compact ? 0 : 2)}\n`;
}

// ---------- HTTP 基础设施 ----------

function sendJson(
  req: IncomingMessage,
  res: ServerResponse,
  status: number,
  body: unknown,
  headers?: Record<string, string>,
): void {
  if (res.writableEnded || res.destroyed) return;
  const text = JSON.stringify(body);
  try {
    res.statusCode = status;
    res.setHeader('Content-Type', 'application/json; charset=utf-8');
    res.setHeader('Cache-Control', 'no-store');
    res.setHeader('Content-Length', String(Buffer.byteLength(text)));
    for (const [k, v] of Object.entries(headers ?? {})) res.setHeader(k, v);
    if (req.method === 'HEAD') res.end();
    else res.end(text);
  } catch {
    // 客户端已经断开（request aborted）：没有可写的地方，忽略即可。
  }
}

function sendError(req: IncomingMessage, res: ServerResponse, status: number, message: string, headers?: Record<string, string>): void {
  sendJson(req, res, status, { ok: false, error: message }, headers);
}

function methodNotAllowed(req: IncomingMessage, res: ServerResponse, allow: string): void {
  sendError(req, res, 405, `method not allowed（只支持 ${allow}）`, { Allow: allow });
}

/**
 * 读取请求体。超限时立即停止收集并返回 413；剩余数据会被丢弃（req.resume），
 * 避免客户端继续上传时连接卡死。客户端中断按 400 处理。
 */
function readBody(req: IncomingMessage, limit: number): Promise<Buffer> {
  const declared = Number(req.headers['content-length']);
  if (Number.isFinite(declared) && declared > limit) {
    req.resume();
    return Promise.reject(new EditorApiError(413, `请求体过大（上限 ${limit} 字节）`));
  }
  return new Promise<Buffer>((resolve, reject) => {
    const chunks: Buffer[] = [];
    let total = 0;
    let settled = false;

    const cleanup = () => {
      req.off('data', onData);
      req.off('end', onEnd);
      req.off('error', onError);
      req.off('aborted', onAborted);
      req.off('close', onClose);
    };
    const settle = (fn: () => void) => {
      if (settled) return;
      settled = true;
      cleanup();
      fn();
    };
    const onData = (chunk: Buffer) => {
      total += chunk.length;
      if (total > limit) {
        req.pause();
        settle(() => reject(new EditorApiError(413, `请求体过大（上限 ${limit} 字节）`)));
        req.resume();
        return;
      }
      chunks.push(chunk);
    };
    const onEnd = () => settle(() => resolve(Buffer.concat(chunks)));
    const onError = () => settle(() => reject(new EditorApiError(400, '请求体读取失败')));
    const onAborted = () => settle(() => reject(new EditorApiError(400, '请求被中断')));
    const onClose = () => {
      if (!req.complete) settle(() => reject(new EditorApiError(400, '请求被中断')));
    };

    req.on('data', onData);
    req.on('end', onEnd);
    req.on('error', onError);
    req.on('aborted', onAborted);
    req.on('close', onClose);
  });
}

async function readJsonBody(req: IncomingMessage): Promise<Record<string, unknown>> {
  const buf = await readBody(req, JSON_BODY_LIMIT);
  let value: unknown;
  try {
    value = JSON.parse(buf.toString('utf8'));
  } catch {
    throw new EditorApiError(400, '请求体不是合法 JSON');
  }
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    throw new EditorApiError(400, '请求体必须是 JSON 对象');
  }
  return value as Record<string, unknown>;
}

// ---------- 尺寸 / 类型识别 ----------

interface ImageHeader {
  mime: 'image/png' | 'image/webp';
  ext: '.png' | '.webp';
  width: number;
  height: number;
}

function ascii(bytes: Buffer, start: number, length: number): string {
  return bytes.subarray(start, start + length).toString('latin1');
}

/** 解析 PNG IHDR / WebP VP8、VP8L、VP8X 的像素尺寸；不合格返回 null。 */
export function parseImageHeader(bytes: Buffer): ImageHeader | null {
  // PNG：8 字节签名 + IHDR 块（长度 13，类型 IHDR，宽高各 4 字节大端）
  if (
    bytes.length >= 24 &&
    bytes[0] === 0x89 &&
    ascii(bytes, 1, 3) === 'PNG' &&
    bytes[4] === 0x0d &&
    bytes[5] === 0x0a &&
    bytes[6] === 0x1a &&
    bytes[7] === 0x0a &&
    ascii(bytes, 12, 4) === 'IHDR' &&
    bytes.readUInt32BE(8) === 13
  ) {
    const width = bytes.readUInt32BE(16);
    const height = bytes.readUInt32BE(20);
    if (width > 0 && height > 0) return { mime: 'image/png', ext: '.png', width, height };
    return null;
  }

  // WebP：RIFF....WEBP + 第一个 chunk（VP8 / VP8L / VP8X）
  if (bytes.length >= 30 && ascii(bytes, 0, 4) === 'RIFF' && ascii(bytes, 8, 4) === 'WEBP') {
    const fourcc = ascii(bytes, 12, 4);
    const chunkSize = bytes.readUInt32LE(16);
    const data = 20;
    // 列表只读文件头 64 字节，不能要求整个 chunk 到齐；只检查解析尺寸所需的头部。
    if (fourcc === 'VP8 ') {
      // 3 字节 frame tag，随后 0x9d 0x01 0x2a；宽高各 14 位
      if (chunkSize < 10 || bytes.length < data + 10) return null;
      if (bytes[data + 3] !== 0x9d || bytes[data + 4] !== 0x01 || bytes[data + 5] !== 0x2a) return null;
      const width = bytes.readUInt16LE(data + 6) & 0x3fff;
      const height = bytes.readUInt16LE(data + 8) & 0x3fff;
      if (width > 0 && height > 0) return { mime: 'image/webp', ext: '.webp', width, height };
      return null;
    }
    if (fourcc === 'VP8L') {
      if (chunkSize < 5 || bytes.length < data + 5 || bytes[data] !== 0x2f) return null;
      const b0 = bytes[data + 1]!;
      const b1 = bytes[data + 2]!;
      const b2 = bytes[data + 3]!;
      const b3 = bytes[data + 4]!;
      const width = 1 + (((b1 & 0x3f) << 8) | b0);
      const height = 1 + (((b3 & 0x0f) << 10) | (b2 << 2) | ((b1 & 0xc0) >> 6));
      if (width > 0 && height > 0) return { mime: 'image/webp', ext: '.webp', width, height };
      return null;
    }
    if (fourcc === 'VP8X') {
      if (chunkSize < 10 || bytes.length < data + 10) return null;
      const width = 1 + (bytes[data + 4]! | (bytes[data + 5]! << 8) | (bytes[data + 6]! << 16));
      const height = 1 + (bytes[data + 7]! | (bytes[data + 8]! << 8) | (bytes[data + 9]! << 16));
      if (width > 0 && height > 0) return { mime: 'image/webp', ext: '.webp', width, height };
      return null;
    }
  }
  return null;
}

interface AudioHeader {
  ext: '.wav' | '.mp3' | '.ogg' | '.m4a';
  mime: string;
}

/** 按文件头识别上传音频；不看用户给的扩展名。 */
export function parseAudioHeader(bytes: Buffer): AudioHeader | null {
  if (bytes.length >= 12 && ascii(bytes, 0, 4) === 'RIFF' && ascii(bytes, 8, 4) === 'WAVE') {
    return { ext: '.wav', mime: 'audio/wav' };
  }
  if (bytes.length >= 3 && ascii(bytes, 0, 3) === 'ID3') return { ext: '.mp3', mime: 'audio/mpeg' };
  if (bytes.length >= 4) {
    // MPEG 音频 frame sync：11 位全 1，且版本/layer 不是保留值
    const version = (bytes[1]! >> 3) & 0x03;
    const layer = (bytes[1]! >> 1) & 0x03;
    if (bytes[0] === 0xff && (bytes[1]! & 0xe0) === 0xe0 && version !== 1 && layer !== 0) {
      return { ext: '.mp3', mime: 'audio/mpeg' };
    }
  }
  if (bytes.length >= 4 && ascii(bytes, 0, 4) === 'OggS') return { ext: '.ogg', mime: 'audio/ogg' };
  if (bytes.length >= 12 && ascii(bytes, 4, 4) === 'ftyp') return { ext: '.m4a', mime: 'audio/mp4' };
  return null;
}

/** 读文件头 64 字节（列表时避免整文件读入）。 */
function readFileHead(path: string, count = 64): Buffer {
  const fd = openSync(path, 'r');
  try {
    const buf = Buffer.alloc(count);
    const got = readSync(fd, buf, 0, count, 0);
    return buf.subarray(0, got);
  } finally {
    closeSync(fd);
  }
}

// ---------- 素材 ID / 上传文件名 ----------

/** 场景资源 ID 只允许 [A-Za-z0-9_.:-]；文件名不合法时追加短哈希避免碰撞。 */
function assetIdFor(fileName: string): string {
  const cleaned = fileName.replace(/[^A-Za-z0-9_.:-]+/g, '.');
  if (cleaned && cleaned === fileName) return cleaned;
  const hash = createHash('sha256').update(fileName).digest('hex').slice(0, 6);
  return `${cleaned || 'asset'}.${hash}`;
}

function sanitizeUploadStem(raw: string): string {
  let name = raw.replace(/\\/g, '/');
  name = name.slice(name.lastIndexOf('/') + 1);
  name = name.replace(/[\0-\x1f\u007f]/g, '');
  name = name.replace(/\.[^.]*$/, '');
  name = name.replace(/[^A-Za-z0-9._-]+/g, '-');
  name = name.replace(/-+/g, '-').replace(/^[.\-_]+|[.\-_]+$/g, '');
  if (name.length > 96) name = name.slice(0, 96).replace(/[.\-_]+$/, '');
  return name || 'asset';
}

function sha256(bytes: Buffer): string {
  return createHash('sha256').update(bytes).digest('hex');
}

// ---------- 列表 ----------

function sceneEntry(repoReal: string, rel: string): SceneFileInfo | null {
  // 列表不跟随 symlink；直接跳过（与 assetList 的策略一致）。
  const abs = join(repoReal, ...rel.split('/'));
  let st;
  try {
    st = statSync(abs);
  } catch {
    return null;
  }
  if (!st.isFile()) return null;
  return { path: rel, size: st.size, mtime: st.mtime.toISOString() };
}

function walkScenes(repoReal: string, dir: string, relDir: string, depth: number, out: SceneFileInfo[]): void {
  if (depth > MAX_SCENE_DEPTH) return;
  let entries;
  try {
    entries = readdirSync(dir, { withFileTypes: true });
  } catch {
    return;
  }
  for (const entry of entries) {
    if (entry.isSymbolicLink()) continue;
    const rel = `${relDir}/${entry.name}`;
    if (entry.isDirectory()) {
      if (depth < MAX_SCENE_DEPTH) walkScenes(repoReal, join(dir, entry.name), rel, depth + 1, out);
      continue;
    }
    if (!entry.isFile()) continue;
    if (!entry.name.toLowerCase().endsWith('.json') || entry.name.toLowerCase().endsWith('.bak.json')) continue;
    const info = sceneEntry(repoReal, rel);
    if (info) out.push(info);
  }
}

/** scenes/（深度 ≤ 3）+ packages/scene/examples/*.json，不解析内容。 */
export function listScenes(repoReal: string): SceneFileInfo[] {
  const out: SceneFileInfo[] = [];
  const scenesRoot = join(repoReal, 'scenes');
  try {
    if (isInside(repoReal, realpathSync.native(scenesRoot))) {
      walkScenes(repoReal, scenesRoot, 'scenes', 1, out);
    }
  } catch {
    // scenes/ 不存在：返回空列表
  }
  const examplesRoot = join(repoReal, 'packages', 'scene', 'examples');
  try {
    if (isInside(repoReal, realpathSync.native(examplesRoot))) {
      for (const entry of readdirSync(examplesRoot, { withFileTypes: true })) {
        if (!entry.isFile() || entry.isSymbolicLink()) continue;
        if (!entry.name.toLowerCase().endsWith('.json') || entry.name.toLowerCase().endsWith('.bak.json')) continue;
        const info = sceneEntry(repoReal, `packages/scene/examples/${entry.name}`);
        if (info) out.push(info);
      }
    }
  } catch {
    // examples 目录不存在：跳过
  }
  return out.sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0));
}

/** assets/editor/ 下指定类别的素材；图片解析文件头尺寸，坏文件跳过；目录不存在返回空。 */
export function listEditorAssets(repoReal: string, kind: 'image' | 'audio'): EditorAssetInfo[] {
  const dir = join(repoReal, EDITOR_ASSETS_DIR);
  let entries;
  try {
    if (!isInside(repoReal, realpathSync.native(dir))) return [];
    entries = readdirSync(dir, { withFileTypes: true });
  } catch {
    return [];
  }
  const out: EditorAssetInfo[] = [];
  for (const entry of entries) {
    if (!entry.isFile() || entry.isSymbolicLink()) continue;
    const ext = extname(entry.name).toLowerCase();
    if (kind === 'image' && !IMAGE_EXTENSIONS.has(ext)) continue;
    if (kind === 'audio' && !AUDIO_EXTENSIONS.has(ext)) continue;
    const abs = join(dir, entry.name);
    let st;
    try {
      st = statSync(abs);
    } catch {
      continue;
    }
    if (!st.isFile()) continue;
    const base: EditorAssetInfo = {
      id: assetIdFor(entry.name),
      uri: `editor/${entry.name}`,
      size: st.size,
      mtime: st.mtime.toISOString(),
      name: entry.name,
    };
    if (kind === 'image') {
      let header;
      try {
        header = parseImageHeader(readFileHead(abs));
      } catch {
        continue;
      }
      if (!header) continue;
      base.width = header.width;
      base.height = header.height;
    }
    out.push(base);
  }
  return out.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
}

// ---------- 端点实现 ----------

function parseKind(raw: string | null): 'image' | 'audio' {
  if (raw === 'image' || raw === 'audio') return raw;
  throw new EditorApiError(400, 'kind 必须是 image 或 audio');
}

function handlePing(req: IncomingMessage, res: ServerResponse): void {
  sendJson(req, res, 200, { ok: true, version: 1 });
}

function handleScenes(req: IncomingMessage, res: ServerResponse, repoReal: string): void {
  sendJson(req, res, 200, { scenes: listScenes(repoReal) });
}

/** 读取并校验一个场景（AI/脚本调用；只读）。 */
function handleScene(req: IncomingMessage, res: ServerResponse, repoReal: string, url: URL): void {
  const rel = normalizeScenePath(url.searchParams.get('path'), 'path');
  const scene = readSceneFile(repoReal, rel);
  sendJson(req, res, 200, { ok: true, path: rel, scene });
}

function handleAssets(req: IncomingMessage, res: ServerResponse, repoReal: string, url: URL): void {
  const kind = parseKind(url.searchParams.get('kind'));
  sendJson(req, res, 200, { assets: listEditorAssets(repoReal, kind) });
}

async function handleUpload(req: IncomingMessage, res: ServerResponse, repoReal: string, url: URL): Promise<void> {
  const kind = parseKind(url.searchParams.get('kind'));
  const body = await readBody(req, UPLOAD_BODY_LIMIT);
  if (body.length === 0) throw new EditorApiError(400, '上传内容为空');

  let header: ImageHeader | AudioHeader;
  if (kind === 'image') {
    const image = parseImageHeader(body);
    if (!image) throw new EditorApiError(400, '不是有效的 PNG/WebP 图片');
    if (image.width > MAX_IMAGE_SIDE || image.height > MAX_IMAGE_SIDE) {
      throw new EditorApiError(400, `图片单边不能超过 ${MAX_IMAGE_SIDE} 像素（${image.width}×${image.height}）`);
    }
    if (image.width * image.height > MAX_IMAGE_PIXELS) {
      throw new EditorApiError(400, `图片总像素不能超过 ${MAX_IMAGE_PIXELS}（${image.width}×${image.height}）`);
    }
    header = image;
  } else {
    const audio = parseAudioHeader(body);
    if (!audio) throw new EditorApiError(400, '不是有效的 WAV/MP3/OGG/M4A 音频');
    header = audio;
  }

  const stem = sanitizeUploadStem(url.searchParams.get('name') ?? 'asset');
  const overwrite = url.searchParams.get('overwrite') === '1';
  const dirRel = EDITOR_ASSETS_DIR;
  const dirAbs = resolve(repoReal, dirRel);
  assertRealInside(repoReal, dirAbs, '素材目录');
  try {
    mkdirSync(dirAbs, { recursive: true });
  } catch {
    throw new EditorApiError(500, '无法创建素材目录');
  }
  if (!isInside(repoReal, realpathSync.native(dirAbs))) {
    throw new EditorApiError(403, '素材目录越出仓库');
  }

  let fileName = `${stem}${header.ext}`;
  if (!overwrite) {
    for (let i = 2; existsSync(join(dirAbs, fileName)); i++) {
      if (i > 10_000) throw new EditorApiError(500, '无法分配不重复的文件名');
      fileName = `${stem}-${i}${header.ext}`;
    }
  }
  const rel = `${dirRel}/${fileName}`;
  const abs = prepareWriteFile(repoReal, rel, '上传文件');
  try {
    writeFileSync(abs, body);
  } catch {
    throw new EditorApiError(500, '写入上传文件失败');
  }

  const asset: Record<string, unknown> = {
    id: assetIdFor(fileName),
    uri: `editor/${fileName}`,
    size: body.length,
    sha256: sha256(body),
  };
  if (kind === 'image') {
    const image = header as ImageHeader;
    asset.width = image.width;
    asset.height = image.height;
  }
  sendJson(req, res, 200, { ok: true, asset });
}

async function handleSaveScene(req: IncomingMessage, res: ServerResponse, repoReal: string): Promise<void> {
  const body = await readJsonBody(req);
  const rel = normalizeScenePath(body.path, 'path');
  if (!body.scene || typeof body.scene !== 'object' || Array.isArray(body.scene)) {
    throw new EditorApiError(400, 'scene 必须是场景对象');
  }
  const scene = validateScene(body.scene);
  const text = sceneJsonText(scene);
  const abs = prepareWriteFile(repoReal, rel, '场景路径');

  let backup: string | undefined;
  if (body.backup === true) {
    let existing = false;
    try {
      existing = statSync(abs).isFile();
    } catch {
      existing = false;
    }
    if (existing) {
      const backupRel = `${rel}.bak.json`;
      const backupAbs = prepareWriteFile(repoReal, backupRel, '备份路径');
      try {
        copyFileSync(abs, backupAbs);
      } catch {
        throw new EditorApiError(500, '写入备份失败');
      }
      backup = backupRel;
    }
  }
  try {
    writeFileSync(abs, text);
  } catch {
    throw new EditorApiError(500, '写入场景失败');
  }
  const result: Record<string, unknown> = { ok: true, path: rel, bytes: Buffer.byteLength(text) };
  if (backup) result.backup = backup;
  sendJson(req, res, 200, result);
}

async function handleCommand(req: IncomingMessage, res: ServerResponse, repoReal: string): Promise<void> {
  const body = await readJsonBody(req);
  const hasScenePath = body.scene !== undefined;
  const hasSceneObject = body.sceneObject !== undefined;
  if (hasScenePath === hasSceneObject) {
    throw new EditorApiError(400, 'scene 和 sceneObject 必须且只能提供一个');
  }

  let scene: Scene;
  if (hasScenePath) {
    const rel = normalizeScenePath(body.scene, 'scene');
    scene = readSceneFile(repoReal, rel);
  } else {
    if (!body.sceneObject || typeof body.sceneObject !== 'object' || Array.isArray(body.sceneObject)) {
      throw new EditorApiError(400, 'sceneObject 必须是场景对象');
    }
    scene = validateScene(body.sceneObject);
  }

  const commands = body.commands;
  if (Array.isArray(commands)) {
    if (commands.length === 0) throw new EditorApiError(400, 'commands 不能为空');
  } else if (!commands || typeof commands !== 'object') {
    throw new EditorApiError(400, 'commands 必须是命令对象或命令数组');
  }

  const out = body.out === undefined ? null : normalizeScenePath(body.out, 'out');
  const dryRun = body.dryRun === true;
  const hasFrame = body.frame !== undefined;
  const frame = hasFrame ? body.frame : 0;
  if (typeof frame !== 'number' || !Number.isFinite(frame) || frame < 0) {
    throw new EditorApiError(400, 'frame 必须是非负数');
  }

  const working = structuredClone(scene);
  let result: EditResult;
  try {
    result = applyEditCommands(working, commands as EditCommand | EditCommand[], hasFrame ? { frame } : {});
  } catch (e) {
    throw new EditorApiError(400, errText(e));
  }
  // applyEditCommands 原地修改 working；返回值只有 changes/effects/preview，没有 scene。
  const next = validateScene(working);
  const preview = result.preview === undefined ? previewAt(next, frame) : result.preview;

  let written: string | null = null;
  if (out && !dryRun) {
    const abs = prepareWriteFile(repoReal, out, 'out');
    try {
      writeFileSync(abs, sceneJsonText(next));
    } catch {
      throw new EditorApiError(500, '写入 out 场景失败');
    }
    written = out;
  }

  sendJson(req, res, 200, {
    ok: true,
    scene: next,
    out: written,
    changed: result.changes.length > 0,
    effects: result.effects,
    diff: result.changes,
    preview,
  });
}

function handleExport(req: IncomingMessage, res: ServerResponse): void {
  sendJson(req, res, 501, { ok: false, error: 'use browser export or pnpm export:video' });
}

async function handleRequest(req: IncomingMessage, res: ServerResponse, repoReal: string, url: URL): Promise<void> {
  switch (url.pathname) {
    case `${EDITOR_API_PREFIX}/ping`:
      if (req.method !== 'GET' && req.method !== 'HEAD') return methodNotAllowed(req, res, 'GET, HEAD');
      return handlePing(req, res);
    case `${EDITOR_API_PREFIX}/scenes`:
      if (req.method !== 'GET' && req.method !== 'HEAD') return methodNotAllowed(req, res, 'GET, HEAD');
      return handleScenes(req, res, repoReal);
    case `${EDITOR_API_PREFIX}/scene`:
      if (req.method !== 'GET' && req.method !== 'HEAD') return methodNotAllowed(req, res, 'GET, HEAD');
      return handleScene(req, res, repoReal, url);
    case `${EDITOR_API_PREFIX}/assets`:
      if (req.method !== 'GET' && req.method !== 'HEAD') return methodNotAllowed(req, res, 'GET, HEAD');
      return handleAssets(req, res, repoReal, url);
    case `${EDITOR_API_PREFIX}/upload`:
      if (req.method !== 'POST') return methodNotAllowed(req, res, 'POST');
      return handleUpload(req, res, repoReal, url);
    case `${EDITOR_API_PREFIX}/save-scene`:
      if (req.method !== 'POST') return methodNotAllowed(req, res, 'POST');
      return handleSaveScene(req, res, repoReal);
    case `${EDITOR_API_PREFIX}/command`:
      if (req.method !== 'POST') return methodNotAllowed(req, res, 'POST');
      return handleCommand(req, res, repoReal);
    case `${EDITOR_API_PREFIX}/export`:
      if (req.method !== 'POST') return methodNotAllowed(req, res, 'POST');
      return handleExport(req, res);
    default:
      return sendError(req, res, 404, `unknown editor route: ${url.pathname}`);
  }
}

/**
 * 返回 Connect 风格的处理器：命中 /__charaforge/editor... 时处理并返回 true，
 * 否则返回 false（交给静态白名单中间件）。处理是异步的，但返回值同步给出。
 */
export function createEditorApi(repo: string): (req: IncomingMessage, res: ServerResponse) => boolean {
  const repoReal = repoRealPath(repo);
  return (req, res) => {
    let url: URL;
    try {
      url = new URL(req.url ?? '/', 'http://x');
    } catch {
      return false;
    }
    if (url.pathname !== EDITOR_API_PREFIX && !url.pathname.startsWith(`${EDITOR_API_PREFIX}/`)) return false;

    if (!isLoopbackAddress(req.socket?.remoteAddress) || !isAllowedOrigin(req.headers.origin)) {
      sendError(req, res, 403, '只允许本机访问编辑接口');
      return true;
    }

    void handleRequest(req, res, repoReal, url).catch((e: unknown) => {
      if (e instanceof EditorApiError) {
        sendError(req, res, e.status, e.message);
        return;
      }
      // 未预期异常：细节只留在服务端，客户端只看到通用错误。
      console.error('[charaforge] editor API error:', e);
      sendError(req, res, 500, '内部错误');
    });
    return true;
  };
}

export function createEditorApiHandler(repo: string): Connect.NextHandleFunction {
  const api = createEditorApi(repo);
  return (req, res, next) => {
    if (!api(req, res)) next();
  };
}
