import type { CoreInputs, DynamicsPacket } from './instance.ts';

/** 主线程 → Worker */
export type WorkerRequest =
  | { type: 'init'; id: number; coreUrl: string; assetBase: string; memoryBytes: number; index: number; ports: (MessagePort | null)[] }
  /** 加载场景；homeLayers 是这个 Worker 负责求值（持有物理状态链）的图层 */
  | { type: 'scene'; id: number; scene: unknown; homeLayers: string[] }
  /**
   * 求值图层在帧 g 的结果（规范 a～g，不含 Core update）。
   * mode = 'advance' 只推进物理并回复 done；'params' 则把 Core 输入交给第 target 个 Worker 做 Core update，
   * 由那个 Worker 直接回复主线程。
   */
  | { type: 'eval'; id: number; layerId: string; frame: number; overrides?: unknown; mode: 'params' | 'advance'; target: number }
  /** 归还用过的动态数据缓冲，供下次导出复用（减少主线程 GC 压力）；不回复 */
  | { type: 'recycle'; id: number; buf: ArrayBuffer };

/** Worker → Worker（经 MessagePort） */
export interface PeerUpdate {
  type: 'update';
  id: number;
  modelUrl: string;
  inputs: CoreInputs;
  steps: number;
  evalMs: number;
}

/** Worker → 主线程 */
export type WorkerResponse =
  | { type: 'ok'; id: number }
  | { type: 'done'; id: number; steps: number }
  | { type: 'update'; id: number; ms: number; evalMs: number; steps: number; packet: DynamicsPacket; from: number }
  | { type: 'error'; id: number; message: string };
