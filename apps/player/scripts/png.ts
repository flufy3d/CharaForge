/*
 * 最小 PNG 编码器（RGBA8，无隔行，滤波 0）。输入为预乘 alpha 时先反预乘。
 * 用 fflate 固定压缩级别，输出字节只取决于像素，保证「同像素 → 同文件」。
 */
import { zlibSync } from 'fflate';

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(buf: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]!) & 0xff]! ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type: string, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + data.length);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  dv.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
}

/** rgba：RGBA8，行从上到下。premultiplied=true 时先反预乘。 */
export function encodePng(width: number, height: number, rgba: Uint8Array, premultiplied = true): Buffer {
  const raw = new Uint8Array((width * 4 + 1) * height);
  for (let y = 0; y < height; y++) {
    const o = y * (width * 4 + 1);
    raw[o] = 0;
    const row = rgba.subarray(y * width * 4, (y + 1) * width * 4);
    if (!premultiplied) {
      raw.set(row, o + 1);
      continue;
    }
    for (let i = 0; i < row.length; i += 4) {
      const a = row[i + 3]!;
      const d = o + 1 + i;
      if (a === 255 || a === 0) {
        raw[d] = a === 0 ? 0 : row[i]!; raw[d + 1] = a === 0 ? 0 : row[i + 1]!; raw[d + 2] = a === 0 ? 0 : row[i + 2]!;
      } else {
        raw[d] = Math.min(255, Math.round((row[i]! * 255) / a));
        raw[d + 1] = Math.min(255, Math.round((row[i + 1]! * 255) / a));
        raw[d + 2] = Math.min(255, Math.round((row[i + 2]! * 255) / a));
      }
      raw[d + 3] = a;
    }
  }
  const ihdr = new Uint8Array(13);
  const dv = new DataView(ihdr.buffer);
  dv.setUint32(0, width);
  dv.setUint32(4, height);
  ihdr[8] = 8; ihdr[9] = 6; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
  const sig = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
  const parts = [sig, chunk('IHDR', ihdr), chunk('IDAT', zlibSync(raw, { level: 6 })), chunk('IEND', new Uint8Array(0))];
  return Buffer.concat(parts);
}
