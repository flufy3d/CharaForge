/*
 * 注入到 harness 页面里的拟合引擎（window.__fit）。
 *
 * 渲染用的全部是 harness 页面里那份 @charaforge/runtime（window.__cfRuntime）的对象：
 *   - 每个图层一个 ModelInstance（runtime.createInstance），按 scene.ts 规范第 3、4 条逐帧推进：
 *     a 复位 → d 按数组顺序 setParamByIndex（与 LayerEvaluator 同一函数、同一顺序）→ e stepPhysics(1/fps)
 *     → f postPhysics 轨 → g update；预热：stabilize（帧 in 的输入）后从 in − warmup 推进，预热帧用帧 in 的输入；
 *   - 合成用 Compositor.begin / layerSource / outputLayer（与 runtime.renderFrame 的 drawPlanned 同一组函数），
 *     图层矩阵按规范第 2 条（与 runtime/math.ts 的 layerMatrix 同一公式）。
 * 所以拟合时看到的像素就是 `pnpm render` 渲染同一份 scene.json 得到的像素。
 *
 * 为什么不直接用 renderFrame 的一次性覆盖：Cubism 物理输出在帧 f 只取决于帧 f−1 及以前的输入（实测
 * AngleX 阶跃后 ParamAngleX 下一帧才变化），帧 f 的画面要靠改帧 f−1 的物理输入；而一次性覆盖只作用于被请求的那一帧，
 * 运行时的状态缓存（检查点/最近帧）也假定场景不变。这里自己持有逐帧规范状态 S(g)，改了哪一帧就从那一帧重新推进。
 *
 * 本函数以源码形式送进页面执行，必须自包含。
 */
export function installPageHelpers(): void {
  // biome-ignore lint: 页面内对象
  const w = window as any;
  const rt = w.__cfRuntime;
  if (!rt) throw new Error('harness 页面没有暴露 __cfRuntime');
  const WIDTH = 1280;
  const HEIGHT = 720;
  const FPS = 60;
  // 绕过 runtime 的清屏色 bug（详见 render.ts 文件头）：Cubism 绘制遮罩时把 GL 清屏色改成白色而 Pixi 不知道，
  // 之后 clear=true 的 pass（blur/color 滤镜、frameHold 缓存）在透明处输出白色。每次 pass 前把清屏色设回 0。
  // 只影响 GL 状态、不改变任何算法；runtime 修复后这几行变成空操作。
  if (!rt.compositor.__cfClearPatched) {
    const comp = rt.compositor;
    const orig = comp.pass.bind(comp);
    comp.pass = (...a: unknown[]) => { comp.gl.clearColor(0, 0, 0, 0); return orig(...a); };
    comp.__cfClearPatched = true;
  }

  interface Track { id: string; idx: number; post: boolean; values: number[] }
  interface Layer {
    id: string; in: number; out: number; z: number; warmup: number; stabilize: boolean;
    handle: number; inst: any;
    tracks: Track[]; byId: Map<string, Track>;
    tf: { x: number[]; y: number[]; scale: number[]; rotation: number[] }; anchorX: number; anchorY: number;
    opacity: number[] | null; blur: number[] | null;
    color: { brightness: number; contrast: number; saturation: number; multiply: [number, number, number] } | null;
    states: Map<number, unknown>;
    /** 实例当前参数/顶点对应的（规范）帧；null = 不确定 */
    ready: number | null;
    needsInit: boolean;
    iris: [number, number];
  }
  interface LayerVals {
    tf?: [number, number, number, number];
    opacity?: number;
    blur?: number;
    color?: Layer['color'];
    /** [帧, {参数: 值}]：写进逐帧参数轨 */
    set?: [number, Record<string, number>][];
    /** 眼睛对齐：求值后按相似变换把我们的虹膜（网格质心 + 标定偏移）对齐到这两个屏幕点，覆盖 tf */
    alignEyes?: [[number, number], [number, number]];
  }

  const layers = new Map<string, Layer>();
  /** 拟合时临时隐藏的图层（尚未拟合的前景实例等），不影响导出 */
  let hidden = new Set<string>();
  /** 虹膜网格顶点质心 → 青色质心的偏移（画布像素，参考姿态下标定），见 setup */
  let irisOffset: [[number, number], [number, number]] = [[0, 0], [0, 0]];
  const timing = { compose: 0, read: 0, loss: 0, n: 0, steps: 0 };
  const st = {
    ref: null as Uint8Array | null,
    refScaled: new Map<number, Float32Array>(),
    exclRects: [] as number[][],
    exclScaled: new Map<number, Uint8Array>(),
  };

  // ---------------- 模拟 ----------------

  const idx = (L: Layer, g: number) => Math.min(Math.max(Math.max(g, L.in) - L.in, 0), L.out - L.in - 1);

  /** 规范 a～g 的一步（g < in 为预热帧，输入取帧 in） */
  function step(L: Layer, g: number, final: boolean) {
    const inst = L.inst;
    const i = idx(L, g);
    inst.resetPose();
    for (const t of L.tracks) if (!t.post) inst.setParamByIndex(t.idx, t.values[i]!, 'set', 1);
    inst.stepPhysics(1 / FPS);
    for (const t of L.tracks) if (t.post) inst.setParamByIndex(t.idx, t.values[i]!, 'set', 1);
    if (final) inst.update();
    timing.steps++;
  }

  function init(L: Layer) {
    const inst = L.inst;
    inst.resetPhysics();
    if (L.stabilize) {
      inst.resetPose();
      for (const t of L.tracks) if (!t.post) inst.setParamByIndex(t.idx, t.values[0]!, 'set', 1);
      inst.stabilizePhysics();
    }
    L.states.clear();
    for (let g = L.in - L.warmup; g < L.in; g++) step(L, g, false);
    L.states.set(L.in - 1, inst.saveState());
    L.needsInit = false;
    L.ready = null;
  }

  /** 让实例处于 S(g)（规范状态，逐帧缓存） */
  function toState(L: Layer, g: number) {
    if (L.needsInit) init(L);
    let b = g;
    while (b >= L.in - 1 && !L.states.has(b)) b--;
    if (b < L.in - 1) { init(L); b = L.in - 1; }
    L.inst.restoreState(L.states.get(b));
    for (let k = b + 1; k <= g; k++) {
      step(L, k, false);
      L.states.set(k, L.inst.saveState());
    }
    L.ready = null;
  }

  function invalidateFrom(L: Layer, g: number) {
    if (g <= L.in) { L.needsInit = true; L.states.clear(); L.ready = null; return; }
    for (const k of [...L.states.keys()]) if (k >= g) L.states.delete(k);
    if (L.ready !== null && L.ready >= g) L.ready = null;
  }

  /** 实例处于帧 f 的规范结果（参数 + 顶点），可以绘制 */
  function ready(L: Layer, f: number) {
    if (L.ready === f) return;
    toState(L, f - 1);
    step(L, f, true);
    L.states.set(f, L.inst.saveState());
    L.ready = f;
  }

  function applyVals(L: Layer, f: number, v: LayerVals) {
    const i = f - L.in;
    if (v.tf) { L.tf.x[i] = v.tf[0]; L.tf.y[i] = v.tf[1]; L.tf.scale[i] = v.tf[2]; L.tf.rotation[i] = v.tf[3]; }
    if (v.opacity !== undefined && L.opacity) L.opacity[i] = v.opacity;
    if (v.blur !== undefined && L.blur) L.blur[i] = v.blur;
    if (v.color && L.color) Object.assign(L.color, v.color);
    let first = Infinity;
    for (const [g, ps] of v.set ?? []) {
      if (g >= L.out) continue;
      const j = idx(L, g);
      for (const k in ps) {
        const t = L.byId.get(k);
        if (!t) throw new Error(`图层 ${L.id} 没有参数轨 ${k}`);
        if (t.values[j] !== ps[k]) { t.values[j] = ps[k]!; first = Math.min(first, Math.max(g, L.in)); }
      }
    }
    if (first < Infinity) invalidateFrom(L, first);
  }

  function matrix(L: Layer, f: number): number[] {
    const i = f - L.in;
    const s = L.tf.scale[i]!, r = L.tf.rotation[i]!;
    const rad = (r * Math.PI) / 180;
    const c = r === 0 ? 1 : Math.cos(rad), sn = r === 0 ? 0 : Math.sin(rad);
    // translate(x,y)·rotate·scale(s,s)·translate(−anchor)，与 runtime/math.ts 的 layerMatrix 相同（无 flipX）
    const a = c * s, b = sn * s, cc = -sn * s, d = c * s;
    return [a, b, cc, d, L.tf.x[i]! - (a * L.anchorX + cc * L.anchorY), L.tf.y[i]! - (b * L.anchorX + d * L.anchorY)];
  }

  function compose(f: number) {
    const comp = rt.compositor;
    comp.begin(null);
    for (const L of [...layers.values()].sort((a, b) => a.z - b.z)) {
      if (f < L.in || f >= L.out || hidden.has(L.id)) continue;
      const opacity = L.opacity ? L.opacity[f - L.in]! : 1;
      if (!(opacity > 0)) continue;
      ready(L, f);
      const filters: unknown[] = [];
      if (L.color) filters.push({ type: 'color', ...L.color, offset: [0, 0, 0] });
      if (L.blur && L.blur[f - L.in]! > 0) filters.push({ type: 'blur', sigma: L.blur[f - L.in] });
      const src = comp.layerSource(L.inst, matrix(L, f), filters, true);
      comp.outputLayer(src, { filters, clip: null, opacity, blendMode: 'normal' });
    }
  }

  // ---------------- 读回与损失 ----------------

  function b64ToBytes(b64: string): Uint8Array {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  function bytesToB64(b: Uint8Array): string {
    let s = '';
    const CH = 0x8000;
    for (let i = 0; i < b.length; i += CH) s += String.fromCharCode.apply(null, Array.from(b.subarray(i, i + CH)));
    return btoa(s);
  }

  /**
   * 整幅逐像素读回后按 k×k 盒式平均下采样（预乘 RGBA）。不用运行时的缩放读回：
   * 它经过一次 Pixi pass，当前运行时里 pass 的输出在透明处是不透明白色（见汇报）。
   */
  function readScaled(scale: number) {
    const px = rt.readPixels(undefined, 1);
    const k = Math.round(1 / scale);
    if (k === 1) return px as { width: number; height: number; data: Uint8Array };
    const W2 = Math.round(WIDTH / k), H2 = Math.round(HEIGHT / k);
    const out = new Uint8Array(W2 * H2 * 4);
    const src = px.data as Uint8Array;
    const inv = 1 / (k * k);
    for (let y = 0; y < H2; y++)
      for (let x = 0; x < W2; x++) {
        let a = 0, b = 0, c = 0, d = 0;
        for (let dy = 0; dy < k; dy++) {
          let o = ((y * k + dy) * WIDTH + x * k) * 4;
          for (let dx = 0; dx < k; dx++, o += 4) { a += src[o]!; b += src[o + 1]!; c += src[o + 2]!; d += src[o + 3]!; }
        }
        const j = (y * W2 + x) * 4;
        out[j] = Math.round(a * inv); out[j + 1] = Math.round(b * inv); out[j + 2] = Math.round(c * inv); out[j + 3] = Math.round(d * inv);
      }
    return { width: W2, height: H2, data: out };
  }

  function refAt(scale: number): Float32Array {
    const k = Math.round(1 / scale);
    let r = st.refScaled.get(k);
    if (r) return r;
    const ref = st.ref!;
    const w2 = Math.round(WIDTH / k), h2 = Math.round(HEIGHT / k);
    r = new Float32Array(w2 * h2 * 3);
    const inv = 1 / (k * k);
    for (let y = 0; y < h2; y++)
      for (let x = 0; x < w2; x++) {
        let a = 0, b = 0, c = 0;
        for (let dy = 0; dy < k; dy++) {
          let o = ((y * k + dy) * WIDTH + x * k) * 3;
          for (let dx = 0; dx < k; dx++, o += 3) { a += ref[o]!; b += ref[o + 1]!; c += ref[o + 2]!; }
        }
        const j = (y * w2 + x) * 3;
        r[j] = a * inv; r[j + 1] = b * inv; r[j + 2] = c * inv;
      }
    st.refScaled.set(k, r);
    return r;
  }

  function exclAt(scale: number): Uint8Array {
    const k = Math.round(1 / scale);
    let e = st.exclScaled.get(k);
    if (e) return e;
    const w2 = Math.round(WIDTH / k), h2 = Math.round(HEIGHT / k);
    e = new Uint8Array(w2 * h2);
    for (const [x, y, ww, hh] of st.exclRects) {
      const x0 = Math.max(0, Math.ceil((x! - k / 2) / k)), x1 = Math.min(w2, Math.ceil((x! + ww! - k / 2) / k));
      const y0 = Math.max(0, Math.ceil((y! - k / 2) / k)), y1 = Math.min(h2, Math.ceil((y! + hh! - k / 2) / k));
      for (let yy = y0; yy < y1; yy++) if (x1 > x0) e.fill(1, yy * w2 + x0, yy * w2 + x1);
    }
    st.exclScaled.set(k, e);
    return e;
  }

  /** 虹膜网格质心（画布像素，加标定偏移）；实例须已求值 */
  function irisCanvas(L: Layer): [number, number][] {
    const m = L.inst.model;
    const c = L.inst.canvas;
    return L.iris.map((d, k) => {
      const v = m.getDrawableVertices(d);
      let x = 0, y = 0;
      const n = v.length / 2;
      for (let j = 0; j < n; j++) { x += v[2 * j]; y += v[2 * j + 1]; }
      return [c.originX + (x / n) * c.pixelsPerUnit + irisOffset[k]![0], c.originY - (y / n) * c.pixelsPerUnit + irisOffset[k]![1]] as [number, number];
    });
  }

  /** 相似变换：把画布上的两个虹膜点对齐到屏幕目标点，写入帧 f 的变换，返回 [x, y, scale, rotation] */
  function alignTo(L: Layer, f: number, tgt: [[number, number], [number, number]]): [number, number, number, number] {
    const [p0, p1] = irisCanvas(L);
    const dp = [p1![0] - p0![0], p1![1] - p0![1]], dt = [tgt[1][0] - tgt[0][0], tgt[1][1] - tgt[0][1]];
    const sc = Math.hypot(dt[0]!, dt[1]!) / Math.hypot(dp[0]!, dp[1]!);
    const rot = Math.atan2(dt[1]!, dt[0]!) - Math.atan2(dp[1]!, dp[0]!);
    const c = Math.cos(rot), sn = Math.sin(rot);
    // 中点对中点
    const pm = [(p0![0] + p1![0]) / 2 - L.anchorX, (p0![1] + p1![1]) / 2 - L.anchorY];
    const tm = [(tgt[0][0] + tgt[1][0]) / 2, (tgt[0][1] + tgt[1][1]) / 2];
    const x = tm[0]! - sc * (c * pm[0]! - sn * pm[1]!), y = tm[1]! - sc * (sn * pm[0]! + c * pm[1]!);
    const i = f - L.in;
    L.tf.x[i] = x; L.tf.y[i] = y; L.tf.scale[i] = sc; L.tf.rotation[i] = (rot * 180) / Math.PI;
    return [x, y, sc, (rot * 180) / Math.PI];
  }

  /** 虹膜主网格顶点质心（屏幕像素）；实例须处于帧 f */
  function irisScreen(L: Layer, f: number): [number, number][] {
    const m = L.inst.model;
    const c = L.inst.canvas;
    const M = matrix(L, f);
    return L.iris.map((d) => {
      const v = m.getDrawableVertices(d);
      let x = 0, y = 0;
      const n = v.length / 2;
      for (let k = 0; k < n; k++) { x += v[2 * k]; y += v[2 * k + 1]; }
      const px = c.originX + (x / n) * c.pixelsPerUnit, py = c.originY - (y / n) * c.pixelsPerUnit;
      return [M[0]! * px + M[2]! * py + M[4]!, M[1]! * px + M[3]! * py + M[5]!] as [number, number];
    });
  }

  /** 虹膜青色像素质心（下采样像素）；我们的合成图与参考帧用同一个检测器，眼睑遮挡造成的偏置两边一致 */
  function cyanCentroid(get: (i: number, k: number) => number, W: number, H: number, cx: number, cy: number, rad: number, excl: Uint8Array) {
    let sx = 0, sy = 0, n = 0;
    const r2 = rad * rad;
    for (let y = Math.max(0, Math.floor(cy - rad)); y <= Math.min(H - 1, Math.ceil(cy + rad)); y++)
      for (let x = Math.max(0, Math.floor(cx - rad)); x <= Math.min(W - 1, Math.ceil(cx + rad)); x++) {
        if ((x - cx) ** 2 + (y - cy) ** 2 > r2) continue;
        const i = y * W + x;
        if (excl[i]) continue;
        const R = get(i, 0), G = get(i, 1), B = get(i, 2);
        if (G - R > 45 && B - R > 55 && G > 0.6 * B && B > 110) { sx += x; sy += y; n++; }
      }
    return n ? { x: sx / n, y: sy / n, n } : { x: 0, y: 0, n: 0 };
  }

  /**
   * 损失（遮罩 M = α>0.1 且不在排除区）：
   *   (1 − 灰度 SSIM 均值，盒式窗口半径 r) + wcol·Charbonnier 颜色误差均值/255 + weye·虹膜位置项。
   * 虹膜项：同一窗口里我们的合成图与参考帧的青色质心距离（以两眼间距 5% 为单位，平方，封顶 36）；
   * 参考帧里有眼睛而我们没有时记封顶值，防止优化器靠「藏起眼睛」逃避此项。
   */
  /** 颜色类别：0 = 皮肤（偏粉的亮色），1 = 深色头发（暗且偏蓝）。背景的白色/浅紫圆点都不属于这两类 */
  function cls(R: number, G: number, B: number, k: number): number {
    if (k === 0) return R > 200 && R - B > 15 && R - G > 12 ? 1 : 0;
    return 0.299 * R + 0.587 * G + 0.114 * B < 110 && B - R > 40 ? 1 : 0;
  }

  /**
   * 类别重叠项：在我们图层外接框（外扩 15%、扣除排除区）内，比较「我们的图层 α × 类别」与「参考帧类别」
   * 经盒式模糊（半径 rb）后的软图，按 Σ|o−r| / Σ(o+r) 计（0 = 完全重合）。
   * 与 SSIM 不同，它也惩罚「参考帧里有角色、我们没画到」的区域，而且对背景纹理不敏感。
   */
  function segTerm(P: Uint8Array, W: number, H: number, ref: Float32Array, excl: Uint8Array, bx0: number, by0: number, bx1: number, by1: number, rb: number) {
    const mw = Math.round((bx1 - bx0) * 0.15), mh = Math.round((by1 - by0) * 0.15);
    const X0 = Math.max(0, bx0 - mw), X1 = Math.min(W - 1, bx1 + mw), Y0 = Math.max(0, by0 - mh), Y1 = Math.min(H - 1, by1 + mh);
    const w2 = X1 - X0 + 1, h2 = Y1 - Y0 + 1;
    let tot = 0;
    for (let k = 0; k < 2; k++) {
      const So = new Float64Array((w2 + 1) * (h2 + 1)), Sr = new Float64Array((w2 + 1) * (h2 + 1));
      for (let y = 0; y < h2; y++) {
        let ao = 0, ar = 0;
        for (let x = 0; x < w2; x++) {
          const i = (Y0 + y) * W + X0 + x;
          const a = P[4 * i + 3]! / 255;
          const o = a > 0.02 ? a * cls(P[4 * i]! / a, P[4 * i + 1]! / a, P[4 * i + 2]! / a, k) : 0;
          ao += o; ar += cls(ref[3 * i]!, ref[3 * i + 1]!, ref[3 * i + 2]!, k);
          const q = (y + 1) * (w2 + 1) + x + 1, up = y * (w2 + 1) + x + 1;
          So[q] = So[up]! + ao; Sr[q] = Sr[up]! + ar;
        }
      }
      let num = 0, den = 0;
      for (let y = 0; y < h2; y++)
        for (let x = 0; x < w2; x++) {
          if (excl[(Y0 + y) * W + X0 + x]) continue;
          const ax = Math.max(0, x - rb), bx = Math.min(w2, x + rb + 1), ay = Math.max(0, y - rb), by = Math.min(h2, y + rb + 1);
          const N = (bx - ax) * (by - ay);
          const qo = (So[by * (w2 + 1) + bx]! - So[ay * (w2 + 1) + bx]! - So[by * (w2 + 1) + ax]! + So[ay * (w2 + 1) + ax]!) / N;
          const qr = (Sr[by * (w2 + 1) + bx]! - Sr[ay * (w2 + 1) + bx]! - Sr[by * (w2 + 1) + ax]! + Sr[ay * (w2 + 1) + ax]!) / N;
          num += Math.abs(qo - qr); den += qo + qr;
        }
      tot += den > 0 ? num / den : 0;
    }
    return tot / 2;
  }

  function lossOf(P: Uint8Array, W: number, H: number, scale: number, r: number, wcol: number, eyesList: [number, number][][], weye: number, wseg = 1) {
    const ref = refAt(scale);
    const excl = exclAt(scale);
    const n = W * H;
    const gc = new Float32Array(n), gr = new Float32Array(n);
    const m = new Uint8Array(n);
    let x0 = W, y0 = H, x1 = -1, y1 = -1, cnt = 0, col = 0;
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        const i = y * W + x;
        const a = P[4 * i + 3]! / 255;
        const rr = ref[3 * i]!, rg = ref[3 * i + 1]!, rb = ref[3 * i + 2]!;
        const cr = P[4 * i]! + (1 - a) * rr, cg = P[4 * i + 1]! + (1 - a) * rg, cb = P[4 * i + 2]! + (1 - a) * rb;
        gc[i] = 0.299 * cr + 0.587 * cg + 0.114 * cb;
        gr[i] = 0.299 * rr + 0.587 * rg + 0.114 * rb;
        if (a > 0.1 && !excl[i]) {
          m[i] = 1;
          cnt++;
          if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
          const dr = cr - rr, dg = cg - rg, db = cb - rb;
          col += Math.sqrt(dr * dr + dg * dg + db * db + 4) - 2;
        }
      }
    if (cnt === 0) return { loss: 1 + wcol + 36 * weye + wseg, area: 0, ssim: 0, eye: -1, seg: 1 };
    const X0 = Math.max(0, x0 - r), X1 = Math.min(W - 1, x1 + r), Y0 = Math.max(0, y0 - r), Y1 = Math.min(H - 1, y1 + r);
    const w2 = X1 - X0 + 2, h2 = Y1 - Y0 + 2;
    const S = [new Float64Array(w2 * h2), new Float64Array(w2 * h2), new Float64Array(w2 * h2), new Float64Array(w2 * h2), new Float64Array(w2 * h2)];
    for (let y = 1; y < h2; y++) {
      let s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0;
      for (let x = 1; x < w2; x++) {
        const i = (Y0 + y - 1) * W + (X0 + x - 1);
        const a = gc[i]!, b = gr[i]!;
        s0 += a; s1 += b; s2 += a * a; s3 += b * b; s4 += a * b;
        const o = y * w2 + x, up = (y - 1) * w2 + x;
        S[0]![o] = S[0]![up]! + s0; S[1]![o] = S[1]![up]! + s1; S[2]![o] = S[2]![up]! + s2; S[3]![o] = S[3]![up]! + s3; S[4]![o] = S[4]![up]! + s4;
      }
    }
    const C1 = (0.01 * 255) ** 2, C2 = (0.03 * 255) ** 2;
    let ss = 0;
    for (let y = y0; y <= y1; y++)
      for (let x = x0; x <= x1; x++) {
        if (!m[y * W + x]) continue;
        const ax = Math.max(X0, x - r) - X0, bx = Math.min(X1, x + r) - X0 + 1;
        const ay = Math.max(Y0, y - r) - Y0, by = Math.min(Y1, y + r) - Y0 + 1;
        const N = (bx - ax) * (by - ay);
        const q = (k: number) => S[k]![by * w2 + bx]! - S[k]![ay * w2 + bx]! - S[k]![by * w2 + ax]! + S[k]![ay * w2 + ax]!;
        const ma = q(0) / N, mb = q(1) / N;
        const va = q(2) / N - ma * ma, vb = q(3) / N - mb * mb, cab = q(4) / N - ma * mb;
        ss += ((2 * ma * mb + C1) * (2 * cab + C2)) / ((ma * ma + mb * mb + C1) * (va + vb + C2));
      }
    const ssim = ss / cnt;
    let eyeTerm = 0, eyeErr = -1;
    if (weye > 0 && eyesList.length) {
      const k = 1 / scale;
      let acc = 0, m2 = 0;
      for (const eyes of eyesList) {
        const ed = Math.hypot(eyes[1]![0] - eyes[0]![0], eyes[1]![1] - eyes[0]![1]) / k;
        const minN = Math.max(6, 0.004 * ed * ed);
        for (const [ex, ey] of eyes) {
          const cx = ex / k - 0.5, cy = ey / k - 0.5;
          const theirs = cyanCentroid((i, c) => ref[3 * i + c]!, W, H, cx, cy, 0.42 * ed, excl);
          if (theirs.n < minN) {
            // 参考帧在我们预测位置附近没有眼睛，但更大范围内有：说明我们错位很远，记封顶值
            const far = cyanCentroid((i, c) => ref[3 * i + c]!, W, H, cx, cy, 0.65 * ed, excl);
            if (far.n >= 2 * minN) { m2++; acc += 36; }
            continue;
          }
          m2++;
          const ours = cyanCentroid((i, c) => P[4 * i + c]! + (1 - P[4 * i + 3]! / 255) * ref[3 * i + c]!, W, H, cx, cy, 0.3 * ed, excl);
          if (ours.n < minN) { acc += 36; continue; }
          acc += Math.min(((ours.x - theirs.x) ** 2 + (ours.y - theirs.y) ** 2) / (0.05 * ed) ** 2, 36);
        }
        if (m2) eyeErr = Math.sqrt(acc / m2) * 0.05 * ed * k;
      }
      if (m2) eyeTerm = (weye * acc) / m2;
    }
    const seg = wseg > 0 ? segTerm(P, W, H, ref, excl, x0, y0, x1, y1, Math.max(2, Math.round(8 * scale))) : 0;
    return { loss: 1 - ssim + (wcol * col) / cnt / 255 + eyeTerm + wseg * seg, area: cnt, ssim, eye: eyeErr, seg };
  }

  // ---------------- 接口 ----------------

  w.__fit = {
    /** 建立图层（scene.json 的模型图层对象：transform / opacity / blur / params 必须是 baked 曲线，或常量关键帧） */
    async setup(sceneLayers: any[], modelUri: string, offset?: [[number, number], [number, number]]) {
      if (offset) irisOffset = offset;
      for (const L of layers.values()) rt.destroyInstance(L.handle);
      layers.clear();
      for (const l of sceneLayers) {
        const handle = await rt.createInstance(modelUri);
        const inst = rt.instance(handle);
        const n = l.out - l.in;
        const tracks: Track[] = (l.params ?? []).map((p: any) => {
          if (p.curve.keys && p.curve.keys.length > 1) throw new Error(`拟合场景里的关键帧轨必须是常量：${p.id}`);
          if (p.mode && p.mode !== 'set') throw new Error(`拟合场景只支持 set 轨：${p.id}`);
          const values: number[] = p.curve.baked ? p.curve.baked.values.slice() : Array(n).fill(p.curve.keys[0].v);
          return { id: p.id, idx: inst.paramIndexOf(p.id), post: p.stage === 'postPhysics', values };
        });
        for (const t of tracks) if (t.idx < 0) throw new Error(`模型没有参数 ${t.id}`);
        const blurF = (l.filters ?? []).find((x: any) => x.type === 'blur');
        const colorF = (l.filters ?? []).find((x: any) => x.type === 'color');
        const m = inst.model;
        const find = (name: string) => {
          for (let d = 0; d < m.getDrawableCount(); d++) {
            const id = m.getDrawableId(d);
            if ((id.getString().s ?? String(id.getString())) === name) return d;
          }
          throw new Error(`模型里没有 ${name}`);
        };
        layers.set(l.id, {
          id: l.id, in: l.in, out: l.out, z: l.z ?? 0, warmup: l.physics?.warmup ?? 0, stabilize: !!l.physics?.stabilize,
          handle, inst, tracks, byId: new Map(tracks.map((t) => [t.id, t])),
          tf: { x: l.transform.x.baked.values.slice(), y: l.transform.y.baked.values.slice(), scale: l.transform.scale.baked.values.slice(), rotation: l.transform.rotation.baked.values.slice() },
          anchorX: l.transform.anchorX, anchorY: l.transform.anchorY,
          opacity: l.opacity?.baked ? l.opacity.baked.values.slice() : null,
          blur: blurF ? blurF.sigma.baked.values.slice() : null,
          color: colorF ? { brightness: colorF.brightness, contrast: colorF.contrast, saturation: colorF.saturation, multiply: colorF.multiply.slice() } : null,
          states: new Map(), ready: null, needsInit: true,
          iris: [find('ArtMesh236'), find('ArtMesh134')],
        });
      }
      return [...layers.keys()];
    },
    setRef(b64: string) {
      st.ref = b64ToBytes(b64);
      st.refScaled.clear();
    },
    setExcl(rects: number[][]) {
      st.exclRects = rects;
      st.exclScaled.clear();
    },
    /**
     * 批量评估：每个候选 {图层ID: 值}；写入后从受影响的最早帧重新推进到 f，合成、读回、算损失。
     * 状态缓存只保存规范状态（被改动的帧之后的缓存会失效），候选之间互不影响。eyesOf：算虹膜项的图层。
     */
    evalBatch(f: number, cands: Record<string, LayerVals>[], opt: { scale: number; r: number; wcol: number; weye: number; wseg?: number; eyesOf?: string[] }) {
      const out: { loss: number; area: number; ssim: number; eye: number; seg: number; tf?: number[] }[] = [];
      for (const c of cands) {
        let tfOut: number[] | undefined;
        for (const lid in c) {
          const L = layers.get(lid)!;
          applyVals(L, f, c[lid]!);
          L.ready = null;
          const al = c[lid]!.alignEyes;
          if (al) { ready(L, f); tfOut = alignTo(L, f, al); }
        }
        const t1 = performance.now();
        compose(f);
        const t2 = performance.now();
        const px = readScaled(opt.scale);
        const t3 = performance.now();
        const eyes = (opt.eyesOf ?? [])
          .map((id) => layers.get(id))
          .filter((L): L is Layer => !!L && f >= L.in && f < L.out && L.ready === f)
          .map((L) => irisScreen(L, f));
        out.push({ ...lossOf(px.data, px.width, px.height, opt.scale, opt.r, opt.wcol, eyes, opt.weye, opt.wseg ?? 1), tf: tfOut });
        timing.compose += t2 - t1; timing.read += t3 - t2; timing.loss += performance.now() - t3; timing.n++;
      }
      return out;
    },
    /** 写入选定值，并推进规范状态到 f */
    commit(f: number, vals: Record<string, LayerVals>) {
      for (const lid in vals) {
        const L = layers.get(lid)!;
        applyVals(L, f, vals[lid]!);
        L.ready = null;
        ready(L, f);
      }
    },
    /** 把图层逐帧数组在 [a, b) 设为帧 src 的值 */
    copyFrames(lid: string, src: number, a: number, b: number, ids?: string[]) {
      const L = layers.get(lid)!;
      const arrs = ids
        ? ids.map((id) => L.byId.get(id)!.values)
        : [L.tf.x, L.tf.y, L.tf.scale, L.tf.rotation, ...(L.opacity ? [L.opacity] : []), ...(L.blur ? [L.blur] : []), ...L.tracks.map((t) => t.values)];
      a = Math.max(a, L.in); b = Math.min(b, L.out);
      if (src < L.in || src >= L.out) return;
      for (const arr of arrs) for (let g = a; g < b; g++) arr[g - L.in] = arr[src - L.in]!;
      if (b > a) invalidateFrom(L, a);
    },
    /** 取回图层的全部逐帧数组（写 scene.json 用） */
    dump(lid: string) {
      const L = layers.get(lid)!;
      return { tf: L.tf, opacity: L.opacity, blur: L.blur, color: L.color, params: Object.fromEntries(L.tracks.map((t) => [t.id, t.values])) };
    },
    /** 合成帧 f 并读回 RGBA（预览用，base64） */
    render(f: number, scale: number) {
      compose(f);
      const px = readScaled(scale);
      return { width: px.width, height: px.height, data: bytesToB64(px.data) };
    },
    /** 图层在帧 f 的参数（含物理输出） */
    params(lid: string, f: number) {
      const L = layers.get(lid)!;
      ready(L, f);
      return L.inst.getParams();
    },
    /** 图层在帧 f 的虹膜网格质心（屏幕像素） */
    irisScreen(lid: string, f: number) {
      const L = layers.get(lid)!;
      ready(L, f);
      return irisScreen(L, f);
    },
    setHidden(ids: string[]) {
      hidden = new Set(ids);
    },
    timing() {
      return timing;
    },
  };
}
