// setup 脚本公用工具：下载（带缓存与 sha256 校验）和安全解压。
// 下载内容一律视为不可信数据：只解析、不执行；解压时拒绝路径穿越。
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { dirname, join, normalize, isAbsolute, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { unzipSync } from 'fflate';

export const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
export const CACHE = join(ROOT, '.cache', 'downloads');

export function sha256(buf) {
  return createHash('sha256').update(buf).digest('hex');
}

/** 下载到 .cache/downloads/<name>，已缓存且校验通过则跳过。 */
export async function download({ url, name, sha256: expected }) {
  mkdirSync(CACHE, { recursive: true });
  const file = join(CACHE, name);
  if (existsSync(file) && sha256(readFileSync(file)) === expected) {
    console.log(`[cache] ${name}`);
    return file;
  }
  console.log(`[download] ${url}`);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`下载失败 ${res.status} ${url}`);
  const buf = Buffer.from(await res.arrayBuffer());
  const got = sha256(buf);
  if (got !== expected) {
    throw new Error(`sha256 不匹配：${name}\n  期望 ${expected}\n  实际 ${got}`);
  }
  writeFileSync(file, buf);
  return file;
}

/**
 * 解压 zip。mapPath(entryName) 返回相对输出路径，或 null 表示跳过。
 * 只写普通文件，拒绝绝对路径和 `..`。
 */
export function extractZip(zipFile, outDir, mapPath) {
  const entries = unzipSync(readFileSync(zipFile), {
    filter: (f) => !f.name.endsWith('/') && mapPath(f.name) !== null,
  });
  rmSync(outDir, { recursive: true, force: true });
  let n = 0;
  for (const [name, data] of Object.entries(entries)) {
    const rel = mapPath(name);
    if (rel === null) continue;
    const norm = normalize(rel);
    if (isAbsolute(norm) || norm.split(sep).includes('..')) {
      throw new Error(`拒绝可疑路径：${name}`);
    }
    const dest = join(outDir, norm);
    mkdirSync(dirname(dest), { recursive: true });
    writeFileSync(dest, data);
    n++;
  }
  return n;
}
