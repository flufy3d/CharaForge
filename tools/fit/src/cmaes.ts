/*
 * CMA-ES（Hansen 2016 教程版，满协方差，无依赖）。
 * 按代给出整批候选（ask），由调用方批量评估后 tell，便于把一整代候选一次性送进浏览器渲染。
 * 维度很小（≤ 25），特征分解用 Jacobi 迭代。
 */

export interface CmaOptions {
  sigma: number;
  /** 每维的尺度（参数空间 x = mean + sigma·scale·z），缺省 1 */
  scales?: number[];
  lambda?: number;
  seed?: number;
}

/** 确定性伪随机（mulberry32 + Box-Muller），保证同输入同输出 */
export class Rng {
  private s: number;
  private spare: number | null = null;
  constructor(seed: number) {
    this.s = seed >>> 0;
  }
  next(): number {
    let t = (this.s += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  gauss(): number {
    if (this.spare !== null) {
      const v = this.spare;
      this.spare = null;
      return v;
    }
    let u = 0, v = 0;
    while (u === 0) u = this.next();
    v = this.next();
    const r = Math.sqrt(-2 * Math.log(u));
    this.spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  }
}

function jacobiEigen(A: number[][]): { values: number[]; vectors: number[][] } {
  const n = A.length;
  const a = A.map((r) => r.slice());
  const v = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) off += a[i]![j]! ** 2;
    if (off < 1e-22) break;
    for (let p = 0; p < n; p++)
      for (let q = p + 1; q < n; q++) {
        const apq = a[p]![q]!;
        if (Math.abs(apq) < 1e-300) continue;
        const theta = (a[q]![q]! - a[p]![p]!) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const akp = a[k]![p]!, akq = a[k]![q]!;
          a[k]![p] = c * akp - s * akq;
          a[k]![q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = a[p]![k]!, aqk = a[q]![k]!;
          a[p]![k] = c * apk - s * aqk;
          a[q]![k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k]![p]!, vkq = v[k]![q]!;
          v[k]![p] = c * vkp - s * vkq;
          v[k]![q] = s * vkp + c * vkq;
        }
      }
  }
  return { values: a.map((r, i) => r[i]!), vectors: v };
}

export class CMAES {
  readonly n: number;
  readonly lambda: number;
  private mu: number;
  private w: number[];
  private mueff: number;
  private cc: number; private cs: number; private c1: number; private cmu: number; private damps: number; private chiN: number;
  mean: number[];
  sigma: number;
  private scales: number[];
  private C: number[][];
  private B: number[][];
  private D: number[];
  private pc: number[];
  private ps: number[];
  private rng: Rng;
  private gen = 0;
  private pending: { z: number[]; y: number[]; x: number[] }[] = [];
  best: { x: number[]; f: number } = { x: [], f: Infinity };

  constructor(x0: number[], o: CmaOptions) {
    const n = (this.n = x0.length);
    this.lambda = o.lambda ?? 4 + Math.floor(3 * Math.log(n));
    this.mu = Math.floor(this.lambda / 2);
    const raw = Array.from({ length: this.mu }, (_, i) => Math.log(this.mu + 0.5) - Math.log(i + 1));
    const s = raw.reduce((p, c) => p + c, 0);
    this.w = raw.map((x) => x / s);
    this.mueff = 1 / this.w.reduce((p, c) => p + c * c, 0);
    this.cc = (4 + this.mueff / n) / (n + 4 + (2 * this.mueff) / n);
    this.cs = (this.mueff + 2) / (n + this.mueff + 5);
    this.c1 = 2 / ((n + 1.3) ** 2 + this.mueff);
    this.cmu = Math.min(1 - this.c1, (2 * (this.mueff - 2 + 1 / this.mueff)) / ((n + 2) ** 2 + this.mueff));
    this.damps = 1 + 2 * Math.max(0, Math.sqrt((this.mueff - 1) / (n + 1)) - 1) + this.cs;
    this.chiN = Math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n));
    this.mean = x0.slice();
    this.sigma = o.sigma;
    this.scales = o.scales ?? x0.map(() => 1);
    this.C = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
    this.B = this.C.map((r) => r.slice());
    this.D = Array(n).fill(1);
    this.pc = Array(n).fill(0);
    this.ps = Array(n).fill(0);
    this.rng = new Rng(o.seed ?? 12345);
  }

  /** 生成一代候选（参数空间） */
  ask(): number[][] {
    const n = this.n;
    this.pending = [];
    for (let k = 0; k < this.lambda; k++) {
      const z = Array.from({ length: n }, () => this.rng.gauss());
      const y = Array(n).fill(0);
      for (let i = 0; i < n; i++) {
        let acc = 0;
        for (let j = 0; j < n; j++) acc += this.B[i]![j]! * this.D[j]! * z[j]!;
        y[i] = acc;
      }
      const x = this.mean.map((m, i) => m + this.sigma * this.scales[i]! * y[i]!);
      this.pending.push({ z, y, x });
    }
    return this.pending.map((p) => p.x);
  }

  tell(fit: number[]): void {
    const n = this.n;
    const idx = fit.map((f, i) => [f, i] as const).sort((a, b) => a[0] - b[0]).map((p) => p[1]);
    if (fit[idx[0]!]! < this.best.f) this.best = { x: this.pending[idx[0]!]!.x.slice(), f: fit[idx[0]!]! };
    const old = this.mean.slice();
    const yw = Array(n).fill(0);
    for (let k = 0; k < this.mu; k++) {
      const y = this.pending[idx[k]!]!.y;
      for (let i = 0; i < n; i++) yw[i] += this.w[k]! * y[i]!;
    }
    this.mean = old.map((m, i) => m + this.sigma * this.scales[i]! * yw[i]!);
    // C^{-1/2} yw = B D^{-1} B^T yw
    const bt = Array(n).fill(0);
    for (let j = 0; j < n; j++) {
      let acc = 0;
      for (let i = 0; i < n; i++) acc += this.B[i]![j]! * yw[i]!;
      bt[j] = acc / this.D[j]!;
    }
    const cinv = Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      let acc = 0;
      for (let j = 0; j < n; j++) acc += this.B[i]![j]! * bt[j]!;
      cinv[i] = acc;
    }
    const csn = Math.sqrt(this.cs * (2 - this.cs) * this.mueff);
    for (let i = 0; i < n; i++) this.ps[i] = (1 - this.cs) * this.ps[i]! + csn * cinv[i]!;
    const psn = Math.sqrt(this.ps.reduce((p, c) => p + c * c, 0));
    this.gen++;
    const hsig = psn / Math.sqrt(1 - (1 - this.cs) ** (2 * this.gen)) / this.chiN < 1.4 + 2 / (n + 1) ? 1 : 0;
    const ccn = Math.sqrt(this.cc * (2 - this.cc) * this.mueff);
    for (let i = 0; i < n; i++) this.pc[i] = (1 - this.cc) * this.pc[i]! + hsig * ccn * yw[i]!;
    for (let i = 0; i < n; i++)
      for (let j = 0; j <= i; j++) {
        let rankmu = 0;
        for (let k = 0; k < this.mu; k++) {
          const y = this.pending[idx[k]!]!.y;
          rankmu += this.w[k]! * y[i]! * y[j]!;
        }
        const v =
          (1 - this.c1 - this.cmu) * this.C[i]![j]! +
          this.c1 * (this.pc[i]! * this.pc[j]! + (1 - hsig) * this.cc * (2 - this.cc) * this.C[i]![j]!) +
          this.cmu * rankmu;
        this.C[i]![j] = v;
        this.C[j]![i] = v;
      }
    this.sigma *= Math.exp((this.cs / this.damps) * (psn / this.chiN - 1));
    const e = jacobiEigen(this.C);
    this.B = e.vectors;
    this.D = e.values.map((v) => Math.sqrt(Math.max(v, 1e-20)));
  }
}
