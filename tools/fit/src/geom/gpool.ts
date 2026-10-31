/*
 * 几何 Worker 池：N 个 worker_threads，每个一份 Cubism Core + runtime（无 GL）。
 * 一次雅可比请求按帧切成连续的块分给各 Worker（每个 Worker 从图层起点前向推进到自己的帧）。
 */
import { availableParallelism } from 'node:os';
import { join } from 'node:path';
import { Worker } from 'node:worker_threads';
import { REPO } from '../paths.ts';
import type { DimSpec, JacFrame } from './worker.ts';

export class GeomPool {
  private workers: Worker[] = [];
  private seq = 0;
  private pending = new Map<number, { resolve: (r: JacFrame[]) => void; reject: (e: Error) => void }>();

  static async open(modelPath: string, n = Math.max(1, availableParallelism() - 2)): Promise<GeomPool> {
    const p = new GeomPool();
    const file = join(REPO, 'tools/fit/src/geom/worker.ts');
    await Promise.all(
      Array.from({ length: n }, () =>
        new Promise<void>((resolve, reject) => {
          const w = new Worker(file, {
            workerData: { modelPath },
            execArgv: ['--import', join(REPO, 'tools/fit/src/geom/register.mjs'), '--experimental-strip-types', '--disable-warning=ExperimentalWarning'],
          });
          w.on('error', reject);
          w.on('message', (m: any) => {
            if (m.ready) { p.workers.push(w); resolve(); return; }
            const pend = p.pending.get(m.id);
            if (!pend) return;
            p.pending.delete(m.id);
            if (m.error) pend.reject(new Error(m.error));
            else pend.resolve(m.r);
          });
        }),
      ),
    );
    return p;
  }

  get size(): number {
    return this.workers.length;
  }

  private call(w: Worker, req: unknown): Promise<JacFrame[]> {
    const id = ++this.seq;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      w.postMessage({ id, req });
    });
  }

  /** 各帧的采样顶点画布坐标与雅可比 */
  async jacobians(layer: unknown, frames: number[], dims: DimSpec[], sample: Int32Array): Promise<Map<number, JacFrame>> {
    const fs = [...frames].sort((a, b) => a - b);
    const n = Math.min(this.workers.length, fs.length);
    const chunks: number[][] = Array.from({ length: n }, () => []);
    // 交错分配：每个 Worker 都要从起点推进到自己最后一帧，交错让负载更均衡
    fs.forEach((f, i) => chunks[i % n]!.push(f));
    const rs = await Promise.all(chunks.map((ch, i) => this.call(this.workers[i]!, { layer, frames: ch, dims, sample })));
    const out = new Map<number, JacFrame>();
    for (const r of rs) for (const x of r) out.set(x.frame, x);
    return out;
  }

  /** 物理脉冲响应（见 worker.ts 的 impulse） */
  impulse(req: { base: Record<string, number>; inputs: string[]; outputs: string[]; taps: number; delta: number }): Promise<Record<string, Record<string, number[]>>> {
    const id = ++this.seq;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as never, reject });
      this.workers[0]!.postMessage({ id, impulse: req });
    });
  }

  /** 每个 drawable 的顶点数 */
  vertexCounts(): Promise<number[]> {
    const id = ++this.seq;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as never, reject });
      this.workers[0]!.postMessage({ id, counts: true });
    });
  }

  async close(): Promise<void> {
    await Promise.all(this.workers.map((w) => w.terminate()));
  }
}
