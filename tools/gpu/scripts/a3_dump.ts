/*
 * A3 光栅化对拍数据：在 harness 页面里用 runtime 渲染场景帧，同时把「runtime 实际绘制时」Core 给出的 drawable 数据
 * 以及该帧的全部模型参数值 dump 出来，供 tools/gpu/l2d/raster.py / composite.py 对拍，也供 A1（变形器）、A5 使用。
 *
 *   CHARAFORGE_GPU=egl CHARAFORGE_MSAA=0 GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA \
 *   node --experimental-strip-types --disable-warning=ExperimentalWarning tools/gpu/scripts/a3_dump.ts \
 *     --scene scenes/deepseek-chan/scene.json --frames 2400,2410 --tag s24 [--layers] [--scan]
 *
 * 参数：
 *   --scene   场景文件（Node 端读取后传给页面）
 *   --frames  帧列表，语法同 pnpm render（'0-60'、'1,5,9'）
 *   --tag     输出到 .data/gpu/frames/<tag>/
 *   --layers  另外 dump 每个可见图层的 runtime 中间缓冲（模型原始绘制、滤镜之后），用于分段定位误差
 *   --scan    只做规划扫描（不渲染）：输出 scan.json，每帧每个图层的 opacity / filters / clip / toScreen，用于挑帧
 * 帧按升序渲染（与 pnpm render 同一条顺序播放路径），参考 PNG 与 pnpm render 的输出逐字节相同。
 *
 * 输出格式见 tools/gpu/docs/raster.md 「dump 格式」一节。要点：
 *   manifest.json                     场景、GPU、合成尺寸、帧列表、模型表
 *   model_<模型ID>.json/.bin          静态数据（UV、索引、贴图号、常量标志、mask 列表、参数/部件表、画布信息）
 *   f<帧号6位>.json/.bin              每帧：各图层的变换与合成参数、各 drawable 的顶点（画布像素）/不透明度/
 *                                     乘算色/屏幕色/render order/动态标志、全部参数值、部件不透明度、Framework 的 clip context
 *   f<帧号>_ref.png                   runtime 合成结果（非预乘 RGBA，与 pnpm render 同一编码）
 *   f<帧号>_ref_premul.png            同一结果的预乘 RGBA 原始字节（PNG 只作无损容器，不做反预乘）
 *   f<帧号>_<图层>_model.png / _src.png  （--layers）图层缓冲：Cubism 绘制结果 / 滤镜之后，预乘原始字节
 *   f<帧号>_<图层>_mask<i>.png         （--layers）Framework 的 clipping mask 缓冲 i（RGBA 原始字节，PNG 第 0 行 = GL 第 0 行）
 * .bin 是若干小端数组首尾相接，.json 的 arrays 字段给出 { dtype, shape, offset }。
 */
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { openHarness, REPO } from '../../../apps/player/scripts/harness-node.ts';
import { encodePng } from '../../../apps/player/scripts/png.ts';
import { parseFrames } from '../../../apps/player/scripts/render.ts';

// ---------- 二进制 blob ----------
type DType = 'f4' | 'i4' | 'u1' | 'u2';
interface ArrSpec {
  dtype: DType;
  shape: number[];
  offset: number;
  nbytes: number;
}

class Blob {
  parts: Buffer[] = [];
  size = 0;
  arrays: Record<string, ArrSpec> = {};
  add(name: string, dtype: DType, shape: number[], b64: string): void {
    const buf = Buffer.from(b64, 'base64');
    const pad = (8 - (buf.length % 8)) % 8;
    this.arrays[name] = { dtype, shape, offset: this.size, nbytes: buf.length };
    this.parts.push(buf);
    if (pad) this.parts.push(Buffer.alloc(pad));
    this.size += buf.length + pad;
  }
  write(path: string): void {
    writeFileSync(path, Buffer.concat(this.parts));
  }
}

// ---------- 页面侧代码（在 harness 页面里执行，访问 window.__cfRuntime 的内部对象） ----------
/* eslint-disable @typescript-eslint/no-explicit-any */
function installPageHelpers(): void {
  const w = window as any;
  const b64 = (v: ArrayBufferView): string => {
    const u8 = new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
    let s = '';
    for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, Array.from(u8.subarray(i, i + 0x8000)));
    return btoa(s);
  };
  const cat = (arrs: ArrayLike<number>[], Ctor: any) => {
    let n = 0;
    for (const a of arrs) n += a.length;
    const out = new Ctor(n);
    let o = 0;
    for (const a of arrs) {
      out.set(a, o);
      o += a.length;
    }
    return out;
  };
  const rt = () => w.__cfRuntime;

  /** 一个模型资源的静态数据（与帧无关）。 */
  w.__a3Static = (layerId: string) => {
    const inst = rt().layerInstance(layerId);
    const core = inst.model.getModel();
    const d = core.drawables;
    const n = d.count;
    const res = inst.resource;
    const vtxOff = new Int32Array(n + 1);
    const idxOff = new Int32Array(n + 1);
    const maskOff = new Int32Array(n + 1);
    for (let i = 0; i < n; i++) {
      vtxOff[i + 1] = vtxOff[i] + d.vertexCounts[i];
      idxOff[i + 1] = idxOff[i] + d.indexCounts[i];
      maskOff[i + 1] = maskOff[i] + d.maskCounts[i];
    }
    const uvs = cat(Array.from({ length: n }, (_, i) => d.vertexUvs[i]), Float32Array);
    const idx = cat(Array.from({ length: n }, (_, i) => d.indices[i].subarray(0, d.indexCounts[i])), Uint16Array);
    const masks = cat(Array.from({ length: n }, (_, i) => d.masks[i].subarray(0, d.maskCounts[i])), Int32Array);
    const r = inst.renderer;
    const cm = r._drawableClippingManager;
    return {
      modelUrl: res.url,
      textures: res.setting.FileReferences.Textures,
      canvas: res.canvas,
      drawableCount: n,
      drawableIds: Array.from(d.ids),
      parameterIds: Array.from(core.parameters.ids),
      parameterMin: Array.from(core.parameters.minimumValues),
      parameterMax: Array.from(core.parameters.maximumValues),
      parameterDefault: Array.from(core.parameters.defaultValues),
      partIds: Array.from(core.parts.ids),
      partParents: Array.from(core.parts.parentIndices ?? []),
      offscreenCount: core.offscreens ? core.offscreens.count : 0,
      blendModeEnabled: !!inst.model.isBlendModeEnabled(),
      usesCulling: inst.usesCulling,
      maskBufferSize: cm ? cm.getClippingMaskBufferSize() : null,
      maskRenderTextureCount: cm ? cm.getRenderTextureCount() : 0,
      highPrecisionMask: !!r.isUsingHighPrecisionMask(),
      clipContexts: cm
        ? cm._clippingContextListForMask.map((c: any) => ({ masks: Array.from(c._clippingIdList), clipped: Array.from(c._clippedDrawableIndexList) }))
        : [],
      arrays: {
        vertexOffsets: ['i4', [n + 1], b64(vtxOff)],
        indexOffsets: ['i4', [n + 1], b64(idxOff)],
        maskOffsets: ['i4', [n + 1], b64(maskOff)],
        uvs: ['f4', [vtxOff[n], 2], b64(uvs)],
        indices: ['u2', [idxOff[n]], b64(idx)],
        masks: ['i4', [maskOff[n]], b64(masks)],
        textureIndices: ['i4', [n], b64(d.textureIndices)],
        constantFlags: ['u1', [n], b64(d.constantFlags)],
        blendModes: ['i4', [n], b64(d.blendModes ?? new Int32Array(n))],
        parentPartIndices: ['i4', [n], b64(d.parentPartIndices)],
      },
    };
  };

  /** 规划扫描：每帧每个图层的绘制参数（不求值、不绘制）。 */
  w.__a3Scan = (frames: number[]) => {
    const R = rt();
    const out: any[] = [];
    for (const f of frames) {
      const plans = R.plan(f);
      out.push({
        f,
        layers: plans.map((p: any) => ({ id: p.layer.id, g: p.g, visible: p.visible, toScreen: Array.from(p.toScreen), draw: p.draw })),
      });
    }
    return out;
  };

  /**
   * renderFrame 之后调用：取每个图层「刚刚实际绘制时」的 Core 数据与合成参数。
   * 同步路径里每个可见图层在 drawPlanned 之前做了 Core update；frameHold 缓存命中时 Core 停留在帧 g 的状态（上次绘制 g 时）。
   */
  w.__a3Frame = (f: number, withLayers: boolean) => {
    const R = rt();
    const comp = R.compositor;
    const plans = R.plan(f);
    const layers: any[] = [];
    for (const p of plans) {
      const inst = p.ev.instance;
      const core = inst.model.getModel();
      const d = core.drawables;
      const n = d.count;
      const c = inst.resource.canvas;
      const ppu = c.pixelsPerUnit;
      let nv = 0;
      for (let i = 0; i < n; i++) nv += d.vertexCounts[i];
      // 顶点：Cubism 单位（y 向上）→ 画布像素（y 向下）：px = originX + x·PPU，py = originY − y·PPU
      const pos = new Float32Array(nv * 2);
      let o = 0;
      for (let i = 0; i < n; i++) {
        const v = d.vertexPositions[i];
        for (let k = 0; k < d.vertexCounts[i]; k++) {
          pos[o++] = c.originX + v[2 * k] * ppu;
          pos[o++] = c.originY - v[2 * k + 1] * ppu;
        }
      }
      const r = inst.renderer;
      const cm = r._drawableClippingManager;
      const ctx = cm
        ? cm._clippingContextListForMask.map((cc: any) => ({
            isUsing: !!cc._isUsing,
            channel: cc._layoutChannelIndex,
            bufferIndex: cc._bufferIndex,
            layoutBounds: [cc._layoutBounds.x, cc._layoutBounds.y, cc._layoutBounds.width, cc._layoutBounds.height],
            allClippedDrawRect: [cc._allClippedDrawRect.x, cc._allClippedDrawRect.y, cc._allClippedDrawRect.width, cc._allClippedDrawRect.height],
            matrixForMask: Array.from(cc._matrixForMask.getArray()),
            matrixForDraw: Array.from(cc._matrixForDraw.getArray()),
          }))
        : [];
      const L: any = {
        id: p.layer.id,
        model: p.layer.model,
        g: p.g,
        visible: p.visible,
        toScreen: Array.from(p.toScreen),
        space: p.layer.space ?? 'world',
        draw: p.draw,
        antialias: p.antialias,
        modelOpacity: inst.model.getModelOapcity(),
        clipContexts: ctx,
        arrays: {
          positions: ['f4', [nv, 2], b64(pos)],
          opacities: ['f4', [n], b64(d.opacities)],
          multiplyColors: ['f4', [n, 4], b64(d.multiplyColors)],
          screenColors: ['f4', [n, 4], b64(d.screenColors)],
          renderOrders: ['i4', [n], b64(core.getRenderOrders().subarray(0, n))],
          drawOrders: ['i4', [n], b64(d.drawOrders)],
          dynamicFlags: ['u1', [n], b64(d.dynamicFlags)],
          params: ['f4', [core.parameters.count], b64(core.parameters.values)],
          partOpacities: ['f4', [core.parts.count], b64(core.parts.opacities)],
        },
      };
      if (withLayers && p.visible) {
        const W = comp.width, H = comp.height;
        const raw = comp.drawModel(inst, p.toScreen, undefined, p.antialias);
        // Framework 的 clipping mask 缓冲（drawModel 刚生成；GL 第 0 行 = mask 空间 y = −1）
        const gl = comp.gl;
        L.maskPixels = (r._drawableMasks ?? []).map((m: any) => {
          const s = m.getBufferWidth();
          const buf = new Uint8Array(s * s * 4);
          gl.bindFramebuffer(gl.FRAMEBUFFER, m.getRenderTexture());
          gl.pixelStorei(gl.PACK_ALIGNMENT, 1);
          gl.readPixels(0, 0, s, s, gl.RGBA, gl.UNSIGNED_BYTE, buf);
          gl.pixelStorei(gl.PACK_ALIGNMENT, 4);
          return [s, b64(buf)];
        });
        comp.resyncPixi();
        L.modelPixels = b64(comp.readRT(raw, 0, 0, W, H).data);
        const src = comp.layerSource(inst, p.toScreen, p.draw.filters, p.antialias);
        L.srcPixels = b64(comp.readRT(src, 0, 0, W, H).data);
      }
      layers.push(L);
    }
    return layers;
  };
}
/* eslint-enable @typescript-eslint/no-explicit-any */

async function main() {
  const argv = process.argv.slice(2).filter((a) => a !== '--');
  const { values } = parseArgs({
    args: argv,
    options: {
      scene: { type: 'string' },
      frames: { type: 'string' },
      tag: { type: 'string' },
      layers: { type: 'boolean', default: false },
      scan: { type: 'boolean', default: false },
    },
  });
  if (!values.scene || !values.tag) {
    console.error('用法：a3_dump.ts --scene <path> --frames <spec> --tag <名字> [--layers] [--scan]');
    process.exit(2);
  }
  const scenePath = resolve(process.cwd(), values.scene);
  const outDir = join(REPO, '.data/gpu/frames', values.tag);
  mkdirSync(outDir, { recursive: true });
  const scene = JSON.parse(readFileSync(scenePath, 'utf8'));

  const h = await openHarness();
  try {
    console.log(`[a3_dump] GPU: ${h.info.renderer}（msaa ${h.info.msaa}）`);
    const sum = await h.page.evaluate((s) => window.__charaforge.loadScene(s), scene);
    await h.page.evaluate(installPageHelpers);
    const frames = parseFrames(values.frames ?? `0-${sum.duration - 1}`, sum.duration);

    if (values.scan) {
      const out = [];
      for (let i = 0; i < frames.length; i += 200) {
        out.push(...(await h.page.evaluate((fs) => (window as any).__a3Scan(fs), frames.slice(i, i + 200))));
      }
      writeFileSync(join(outDir, 'scan.json'), JSON.stringify({ scene: values.scene, width: sum.width, height: sum.height, frames: out }));
      console.log(`[a3_dump] 扫描 ${frames.length} 帧 → ${join(outDir, 'scan.json')}`);
      return;
    }

    const models: Record<string, string> = {};
    const t0 = Date.now();
    for (const f of frames) {
      const r = await h.page.evaluate((f) => window.__charaforge.renderFrame(f, { read: { scale: 1 } }), f);
      const name = `f${String(f).padStart(6, '0')}`;
      const premul = Buffer.from(r.data!, 'base64');
      writeFileSync(join(outDir, `${name}_ref.png`), encodePng(r.width!, r.height!, premul));
      writeFileSync(join(outDir, `${name}_ref_premul.png`), encodePng(r.width!, r.height!, premul, false));
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const layers: any[] = await h.page.evaluate(([f, wl]) => (window as any).__a3Frame(f, wl), [f, values.layers] as const);
      const blob = new Blob();
      const meta = [];
      for (let li = 0; li < layers.length; li++) {
        const L = layers[li];
        if (!(L.model in models)) {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const st: any = await h.page.evaluate((id) => (window as any).__a3Static(id), L.id);
          const sb = new Blob();
          for (const [k, [dt, shape, data]] of Object.entries(st.arrays as Record<string, [DType, number[], string]>)) sb.add(k, dt, shape, data);
          const base = `model_${L.model}`;
          sb.write(join(outDir, `${base}.bin`));
          const { arrays: _drop, ...rest } = st;
          writeFileSync(join(outDir, `${base}.json`), JSON.stringify({ ...rest, arrays: sb.arrays }, null, 1));
          models[L.model] = base;
        }
        for (const [k, [dt, shape, data]] of Object.entries(L.arrays as Record<string, [DType, number[], string]>)) blob.add(`${li}/${k}`, dt, shape, data);
        if (L.modelPixels) {
          const safe = String(L.id).replace(/[^A-Za-z0-9_.-]/g, '_');
          writeFileSync(join(outDir, `${name}_${safe}_model.png`), encodePng(sum.width, sum.height, Buffer.from(L.modelPixels, 'base64'), false));
          writeFileSync(join(outDir, `${name}_${safe}_src.png`), encodePng(sum.width, sum.height, Buffer.from(L.srcPixels, 'base64'), false));
          (L.maskPixels as [number, string][]).forEach(([s, data], i) => {
            writeFileSync(join(outDir, `${name}_${safe}_mask${i}.png`), encodePng(s, s, Buffer.from(data, 'base64'), false));
          });
        }
        const { arrays: _a, modelPixels: _m, srcPixels: _s, maskPixels: _k, ...rest } = L;
        meta.push(rest);
      }
      blob.write(join(outDir, `${name}.bin`));
      writeFileSync(join(outDir, `${name}.json`), JSON.stringify({ frame: f, width: r.width, height: r.height, layers: meta, arrays: blob.arrays }));
    }
    writeFileSync(
      join(outDir, 'manifest.json'),
      JSON.stringify(
        {
          format: 'charaforge.a3dump v1',
          scene: values.scene,
          gpu: h.info.renderer,
          msaa: h.info.msaa,
          composition: { width: sum.width, height: sum.height, background: scene.composition.background ?? null },
          fps: sum.fps,
          frames,
          models,
          withLayers: values.layers,
        },
        null,
        1,
      ),
    );
    console.log(`[a3_dump] ${frames.length} 帧 → ${outDir}（${((Date.now() - t0) / 1000).toFixed(1)}s）`);
  } finally {
    await h.close();
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
