/*
 * CubismPhysics 的完整状态快照。
 *
 * Framework 没有提供保存/恢复接口，这里直接读写它的内部字段（字段名对应 Framework R5
 * physics/cubismphysics.ts 与 cubismphysicsinternal.ts）。快照是逐位精确的拷贝：
 * 粒子的 position/lastPosition/lastGravity/velocity/force（JS number，按值复制），
 * 上一次与当前的 rig 输出、未消耗的时间 _currentRemainTime、参数缓存（Float32Array 拷贝）。
 * 恢复后继续推进，结果与「从未保存过」逐位相同。
 */
import type { CubismPhysics } from '@cubism/framework/physics/cubismphysics';
import { CubismVector2 } from '@cubism/framework/math/cubismvector2';

export interface PhysicsSnapshot {
  /** 每个粒子 10 个数：position.xy, lastPosition.xy, lastGravity.xy, velocity.xy, force.xy */
  particles: Float64Array;
  current: Float64Array[];
  previous: Float64Array[];
  remainTime: number;
  parameterCaches: Float32Array | null;
  parameterInputCaches: Float32Array | null;
}

interface Vec { x: number; y: number }
interface Particle { position: Vec; lastPosition: Vec; lastGravity: Vec; velocity: Vec; force: Vec }
interface PhysicsInternals {
  _physicsRig: { particles: Particle[] };
  _currentRigOutputs: { outputs: number[] }[];
  _previousRigOutputs: { outputs: number[] }[];
  _currentRemainTime: number;
  _parameterCaches: Float32Array | null;
  _parameterInputCaches: Float32Array | null;
}

const internals = (p: CubismPhysics) => p as unknown as PhysicsInternals;

export function savePhysics(physics: CubismPhysics): PhysicsSnapshot {
  const p = internals(physics);
  const ps = p._physicsRig.particles;
  const particles = new Float64Array(ps.length * 10);
  for (let i = 0, o = 0; i < ps.length; i++, o += 10) {
    const q = ps[i]!;
    particles[o] = q.position.x; particles[o + 1] = q.position.y;
    particles[o + 2] = q.lastPosition.x; particles[o + 3] = q.lastPosition.y;
    particles[o + 4] = q.lastGravity.x; particles[o + 5] = q.lastGravity.y;
    particles[o + 6] = q.velocity.x; particles[o + 7] = q.velocity.y;
    particles[o + 8] = q.force.x; particles[o + 9] = q.force.y;
  }
  return {
    particles,
    current: p._currentRigOutputs.map((o) => Float64Array.from(o.outputs)),
    previous: p._previousRigOutputs.map((o) => Float64Array.from(o.outputs)),
    remainTime: p._currentRemainTime,
    parameterCaches: p._parameterCaches ? p._parameterCaches.slice() : null,
    parameterInputCaches: p._parameterInputCaches ? p._parameterInputCaches.slice() : null,
  };
}

export function restorePhysics(physics: CubismPhysics, s: PhysicsSnapshot): void {
  const p = internals(physics);
  const ps = p._physicsRig.particles;
  if (ps.length * 10 !== s.particles.length) throw new Error('物理快照与实例不匹配');
  const a = s.particles;
  for (let i = 0, o = 0; i < ps.length; i++, o += 10) {
    const q = ps[i]!;
    // 每次新建向量对象，避免 Framework 内部可能存在的向量对象共享（别名）被我们改坏。
    q.position = vec(a[o]!, a[o + 1]!);
    q.lastPosition = vec(a[o + 2]!, a[o + 3]!);
    q.lastGravity = vec(a[o + 4]!, a[o + 5]!);
    q.velocity = vec(a[o + 6]!, a[o + 7]!);
    q.force = vec(a[o + 8]!, a[o + 9]!);
  }
  copyOutputs(p._currentRigOutputs, s.current);
  copyOutputs(p._previousRigOutputs, s.previous);
  p._currentRemainTime = s.remainTime;
  p._parameterCaches = s.parameterCaches ? s.parameterCaches.slice() : null;
  p._parameterInputCaches = s.parameterInputCaches ? s.parameterInputCaches.slice() : null;
}

function vec(x: number, y: number): Vec {
  return new CubismVector2(x, y) as Vec;
}

function copyOutputs(dst: { outputs: number[] }[], src: Float64Array[]): void {
  for (let i = 0; i < dst.length; i++) {
    const d = dst[i]!.outputs;
    const v = src[i]!;
    for (let j = 0; j < v.length; j++) d[j] = v[j]!;
  }
}
