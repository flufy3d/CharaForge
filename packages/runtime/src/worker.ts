/*
 * Worker 入口（引导）：收到 init 后先在全局作用域执行 Cubism Core，再动态加载 workerMain
 * （Framework 有模块在加载时就引用全局 Live2DCubismCore）。
 * 主线程消息与其他 Worker 经 MessagePort 发来的 update 消息进入同一个串行队列。
 */
import { waitForCubismCore } from './coreReady.ts';
import type { PeerUpdate, WorkerRequest, WorkerResponse } from './workerProtocol.ts';

const scope = self as unknown as {
  onmessage: ((e: MessageEvent<WorkerRequest>) => void) | null;
  postMessage(msg: WorkerResponse): void;
};

let main: Promise<typeof import('./workerMain.ts')> | null = null;
let queue: Promise<void> = Promise.resolve();

function enqueue(id: number, job: () => Promise<void>): void {
  queue = queue.then(job).catch((err: unknown) => {
    scope.postMessage({ type: 'error', id, message: String((err as Error)?.stack ?? err) });
  });
}

const enqueuePeer = (m: PeerUpdate) => enqueue(m.id, async () => (await main!).handlePeer(m));

async function dispatch(msg: WorkerRequest): Promise<void> {
  if (msg.type === 'init' && !main) {
    const src = await (await fetch(msg.coreUrl)).text();
    // 间接 eval：在 Worker 全局作用域执行 vendor 里的官方 Core 脚本，定义全局 Live2DCubismCore
    (0, eval)(src);
    // Core 的 WASM 是异步实例化的，就绪之前不能启动 Framework（见 coreReady.ts）
    await waitForCubismCore();
    main = import('./workerMain.ts');
    for (const p of msg.ports) if (p) p.onmessage = (e: MessageEvent<PeerUpdate>) => enqueuePeer(e.data);
  }
  if (!main) throw new Error('Worker 尚未 init');
  await (await main).handle(msg, enqueuePeer);
}

scope.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === 'recycle' && main) {
    void main.then((m) => m.handle(msg, enqueuePeer));
    return;
  }
  enqueue(msg.id, () => dispatch(msg));
};
