/*
 * 只含 ASCII 的请求 URL。
 *
 * 模型文件名是中文（大肥鱼.model3.json / .moc3 / .physics3.json / 贴图……）。有的反向代理（如 Tailscale 上的本地代理）
 * 会把百分号编码的路径解码成 UTF-8 再转发，Node 的 http.request 遇到非 ASCII 路径直接抛错，返回 500。
 * 所以发请求前把「解码后含非 ASCII 字符」的路径段换成 `_f.<base64url(该段的 UTF-8)>`；纯 ASCII 的段原样保留（日志好读）。
 * 服务端（apps/player/vite.config.ts 的白名单中间件，dev 与 preview 共用）先把这种段解码回原名，再做白名单与防穿越检查。
 *
 * 按段编码而不是整条路径：与路径前缀无关（页面可能挂在代理的任意前缀下），而且 model3.json 里的相对引用
 * 仍按原始 URL 解析，只有真正发请求时才编码。
 */
export const ASCII_SEGMENT_PREFIX = '_f.';

function base64url(s: string): string {
  const bytes = new TextEncoder().encode(s);
  let bin = '';
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]!);
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

const NON_ASCII = /[^\x20-\x7e]/;

/** URL（绝对或相对 base）→ 路径只含 ASCII 的绝对 URL；本来就全是 ASCII 时原样返回。 */
export function asciiUrl(url: string, base?: string): string {
  let u: URL;
  try {
    u = new URL(url, base);
  } catch {
    return url;
  }
  if (u.protocol !== 'http:' && u.protocol !== 'https:') return url;
  let changed = false;
  const segs = u.pathname.split('/').map((seg) => {
    let d: string;
    try {
      d = decodeURIComponent(seg);
    } catch {
      return seg;
    }
    if (!NON_ASCII.test(d)) return seg;
    changed = true;
    return ASCII_SEGMENT_PREFIX + base64url(d);
  });
  if (!changed) return url;
  u.pathname = segs.join('/');
  return u.href;
}
