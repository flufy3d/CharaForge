/**
 * 仓库相对路径（如 assets/…、scenes/…、vendor/…）→ 绝对 URL，相对于页面地址解析（document.baseURI）。
 * 页面可能挂在任意路径前缀下（反向代理，如 https://host/api/local-proxy/p/5173/），所以运行时的请求一律不用 / 开头的绝对路径。
 * 开头的 / 会被去掉：'/assets/x' 与 'assets/x' 等价。
 */
/** Cubism Core 的仓库相对路径（index.html / harness.html 的 <script> 也用同一个相对路径） */
export const CORE_PATH = 'vendor/cubism/core/live2dcubismcore.min.js';

export function pageUrl(rel: string): string {
  return new URL(rel.replace(/^\/+/, ''), document.baseURI).href;
}

export function bytesToBase64(bytes: Uint8Array): string {
  let s = '';
  const CH = 0x8000;
  for (let i = 0; i < bytes.length; i += CH) s += String.fromCharCode.apply(null, bytes.subarray(i, i + CH) as unknown as number[]);
  return btoa(s);
}

export async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const d = await crypto.subtle.digest('SHA-256', bytes as unknown as ArrayBuffer);
  return Array.from(new Uint8Array(d), (b) => b.toString(16).padStart(2, '0')).join('');
}

/** 帧号 → 时间码 hh:mm:ss:ff */
export function timecode(frame: number, fps: number): string {
  const f = Math.max(0, Math.floor(frame));
  const ff = f % Math.round(fps);
  const s = Math.floor(f / fps);
  const p = (n: number, w = 2) => String(n).padStart(w, '0');
  return `${p(Math.floor(s / 3600))}:${p(Math.floor(s / 60) % 60)}:${p(s % 60)}:${p(ff)}`;
}
