/*
 * Worker 池（见 workerMain.ts）。
 * 图层 i 的「主场」Worker = i mod K（持有物理状态链）；Core update 交给主线程挑选的最空闲 Worker。
 * 调度由调用方做（拉模式）：只在有空闲 Worker 时才发任务，过期的预取任务可以在发出前丢掉。
 */
import type { DynamicsPacket } from './instance.ts';
import type { WorkerRequest, WorkerResponse } from './workerProtocol.ts';

type Req = WorkerRequest extends infer T ? (T extends unknown ? Omit<T, 'id'> : never) : never;

export interface UpdateResult {
  packet: DynamicsPacket;
  /** Worker 内 Core update + 导出耗时 */
  ms: number;
  /** 主场 Worker 内求值耗时 */
  evalMs: number;
  steps: number;
  /** 做 update 的 Worker 序号（归还缓冲用） */
  from: number;
}

interface Slot {
  w: Worker;
  busy: number;
}

export class WorkerPool {
  private readonly workers: Slot[];
  private readonly pending = new Map<number, { resolve: (v: WorkerResponse) => void; reject: (e: Error) => void; slot: Slot }>();
  private nextId = 1;
  private home = new Map<string, number>();

  constructor(count: number) {
    this.workers = Array.from({ length: count }, () => {
      const w = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module', name: 'charaforge-core' });
      const slot: Slot = { w, busy: 0 };
      w.onmessage = (e: MessageEvent<WorkerResponse>) => {
        const p = this.pending.get(e.data.id);
        if (!p) return;
        this.pending.delete(e.data.id);
        p.slot.busy--;
        if (e.data.type === 'error') p.reject(new Error(`[worker] ${e.data.message}`));
        else p.resolve(e.data);
      };
      w.onerror = (e) => console.error('[worker]', e.message);
      return slot;
    });
  }

  get size(): number {
    return this.workers.length;
  }

  /** 发请求；响应可能来自另一个 Worker（accountTo 为负责回复、计入忙碌度的那个）。 */
  private call(to: Slot, req: Req, accountTo: Slot = to, transfer: Transferable[] = []): Promise<WorkerResponse> {
    const id = this.nextId++;
    accountTo.busy++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, slot: accountTo });
      to.w.postMessage({ ...req, id } as WorkerRequest, transfer);
    });
  }

  async init(coreUrl: string, assetBase: string, memoryBytes: number): Promise<void> {
    const n = this.workers.length;
    // Worker 之间两两建立 MessageChannel
    const ports: (MessagePort | null)[][] = Array.from({ length: n }, () => Array<MessagePort | null>(n).fill(null));
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      const ch = new MessageChannel();
      ports[i]![j] = ch.port1;
      ports[j]![i] = ch.port2;
    }
    await Promise.all(this.workers.map((s, i) => {
      const ps = ports[i]!;
      return this.call(s, { type: 'init', coreUrl, assetBase, memoryBytes, index: i, ports: ps }, s, ps.filter((p): p is MessagePort => !!p));
    }));
  }

  async loadScene(scene: unknown, layerIds: string[]): Promise<void> {
    const n = this.workers.length;
    this.home = new Map(layerIds.map((id, i) => [id, i % n]));
    await Promise.all(this.workers.map((s, i) => this.call(s, { type: 'scene', scene, homeLayers: layerIds.filter((_, k) => k % n === i) })));
  }

  /** 是否有 Worker 的排队数 < maxDepth。 */
  hasIdle(maxDepth = 1): boolean {
    return this.workers.some((s) => s.busy < maxDepth);
  }

  private homeSlot(layerId: string): Slot {
    const i = this.home.get(layerId);
    if (i === undefined) throw new Error(`未知图层 ${layerId}`);
    return this.workers[i]!;
  }

  /** 求值 + Core update：主场 Worker 求值，最空闲的 Worker 做 update。 */
  async evalUpdate(layerId: string, frame: number, overrides: unknown): Promise<UpdateResult> {
    let best = 0;
    for (let i = 1; i < this.workers.length; i++) if (this.workers[i]!.busy < this.workers[best]!.busy) best = i;
    const r = await this.call(this.homeSlot(layerId), { type: 'eval', layerId, frame, overrides, mode: 'params', target: best }, this.workers[best]!);
    if (r.type !== 'update') throw new Error('Worker 响应类型错误');
    return { packet: r.packet, ms: r.ms, evalMs: r.evalMs, steps: r.steps, from: r.from };
  }

  /** 把用完的动态数据缓冲还给产生它的 Worker。 */
  recycle(r: UpdateResult): void {
    const s = this.workers[r.from];
    if (s && r.packet.buf.byteLength) s.w.postMessage({ type: 'recycle', id: 0, buf: r.packet.buf } as WorkerRequest, [r.packet.buf]);
  }

  /** 只推进物理（不可见图层）。 */
  async advance(layerId: string, frame: number, overrides: unknown): Promise<number> {
    const home = this.homeSlot(layerId);
    const r = await this.call(home, { type: 'eval', layerId, frame, overrides, mode: 'advance', target: 0 });
    return r.type === 'done' ? r.steps : 0;
  }

  dispose(): void {
    for (const s of this.workers) s.w.terminate();
    for (const p of this.pending.values()) p.reject(new Error('Worker 池已关闭'));
    this.pending.clear();
  }
}
