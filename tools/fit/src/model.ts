/*
 * 拟合参数集合（每个模型图层一份状态向量）。
 *
 * 依据：大肥鱼.vtube.json 的 ParameterSettings —— 原视频里由面捕连续驱动的就是这一组（VTube Studio 的输入 → 模型参数）：
 *   FaceAngleX/Y/Z → AngleX/Y/Z 与 PositionX/Y/Z（同一面捕输入，但 VTS 平滑系数不同：15/4/30 vs 20/20/20）
 *   EyeOpenLeft/Right → ParamEyeLOpen/ROpen（0..1.5）；EyeRightX/Y → ParamEyeBallX/Y
 *   Brows → ParamBrowLY、ParamBrowRY（同一输入）；MouthSmile → EyeSmile、ParamMouthForm；MouthOpen → ParamMouthOpenY
 *   ParamBreath：VTS 的 UseBreathing（自动呼吸），是物理输入（身体 Y、头发、袖子……）
 * AngleX/Y/Z、PositionX/Y/Z、LeanInput 都是「控制器」参数，ParamAngleX、ParamBodyAngleX 等由物理从它们算出，
 * 直接写 ParamBodyAngle* 会被物理覆盖，所以拟合的是物理输入，物理正向推进。
 *
 * 第二轮改动：
 *   - 身体（PositionX/Y/Z，经物理驱动 ParamBodyAngleX/Y/Z、盆骨、手臂……）与头部（AngleX/Y/Z）解耦，独立拟合；
 *     第一轮按 VTS 映射强绑定，s24 头身相对姿态不同的帧因此失败；
 *   - 体型：Clothes2（体型1 = 0.4、体型2 = 1）、LeanInput（前倾 = −30）作为镜头常量；
 *   - 呼吸：ParamBreath = 0.5 + 0.5·sin(2π·t/T + φ)，T、φ 为镜头常量；
 *   - 头发：可选的 postPhysics 叠加（add）——按发束分组，每组 1 个自由度（见 HAIR_GROUPS）。
 */

export interface Dim {
  key: string;
  lo: number;
  hi: number;
  /** CMA-ES 的初始搜索尺度 */
  sd: number;
  /** 时间正则的尺度（相邻拟合帧之间变化 tau 记 1 个单位的惩罚） */
  tau: number;
}

export const TF_DIMS: Dim[] = [
  { key: 'x', lo: -2000, hi: 3300, sd: 4, tau: 40 },
  { key: 'y', lo: -2000, hi: 2800, sd: 4, tau: 40 },
  { key: 'ls', lo: Math.log(0.02), hi: Math.log(3), sd: 0.015, tau: 0.08 },
  { key: 'rot', lo: -60, hi: 60, sd: 1.2, tau: 6 },
];

export const HEAD_DIMS: Dim[] = [
  { key: 'hx', lo: -30, hi: 30, sd: 4, tau: 12 },
  { key: 'hy', lo: -20, hi: 20, sd: 3, tau: 10 },
  { key: 'hz', lo: -30, hi: 30, sd: 3, tau: 12 },
];

/** 身体（PositionX/Y/Z），与头部独立 */
export const BODY_DIMS: Dim[] = [
  { key: 'bx', lo: -30, hi: 30, sd: 4, tau: 10 },
  { key: 'by', lo: -30, hi: 30, sd: 4, tau: 10 },
  { key: 'bz', lo: -30, hi: 30, sd: 4, tau: 10 },
];

export const FACE_DIMS: Dim[] = [
  { key: 'eyeL', lo: 0, hi: 1.5, sd: 0.2, tau: 0.8 },
  { key: 'eyeR', lo: 0, hi: 1.5, sd: 0.2, tau: 0.8 },
  { key: 'es', lo: 0, hi: 1, sd: 0.15, tau: 0.6 },
  { key: 'ebx', lo: -1, hi: 1, sd: 0.15, tau: 0.6 },
  { key: 'eby', lo: -1, hi: 1, sd: 0.15, tau: 0.6 },
  { key: 'brow', lo: -1, hi: 1, sd: 0.2, tau: 0.8 },
  { key: 'mo', lo: 0, hi: 1, sd: 0.1, tau: 0.5 },
  { key: 'mf', lo: -1, hi: 1, sd: 0.2, tau: 0.8 },
];

export const BLUR_DIM: Dim = { key: 'blur', lo: 0, hi: 20, sd: 1, tau: 4 };
export const OPACITY_DIM: Dim = { key: 'opacity', lo: 0, hi: 1, sd: 0.1, tau: 0.5 };

/** 镜头常量（锚点帧全局搜索里拟合，之后整段不变）。key 以 c: 开头 */
export const SHOT_CONST_DIMS: Dim[] = [
  { key: 'c:LeanInput', lo: -30, hi: 0, sd: 3, tau: 10 },
  { key: 'c:Clothes2', lo: 0, hi: 1, sd: 0.08, tau: 1 },
  { key: 'c:breathPhase', lo: -Math.PI, hi: Math.PI, sd: 0.8, tau: 10 },
  { key: 'c:breathPeriod', lo: 2, hi: 6, sd: 0.5, tau: 10 },
  { key: 'c:blur', lo: 0, hi: 3, sd: 0.3, tau: 10 },
];

/**
 * 头发 postPhysics 叠加：每组一个自由度 k，组内每个参数叠加 k·gain（参数是物理输出，范围多为 ±1 或 ±10）。
 * 只在正向物理对不上时启用（LayerPlan.hairBake）。
 */
export const HAIR_GROUPS: { key: string; params: string[]; gain: number }[] = [
  { key: 'hFX', params: ['FrontHairX1', 'FrontHairX2', 'FrontHairX3', 'FrontHairX4'], gain: 1 },
  { key: 'hFY', params: ['FrontHairY1', 'FrontHairY2', 'FrontHairY3', 'FrontHairY4'], gain: 1 },
  { key: 'hSX', params: ['SideHairX1', 'SideHairX2', 'SideHairX3', 'SideHairX4'], gain: 1 },
  { key: 'hSY', params: ['SideHairY1', 'SideHairY2', 'SideHairY3', 'SideHairY4'], gain: 1 },
  { key: 'hLX', params: ['LateHairX1', 'LateHairX2', 'LateHairX3', 'LateHairX4', 'LateHairX5'], gain: 1 },
  { key: 'hLY', params: ['LateHairY1', 'LateHairY2', 'LateHairY3', 'LateHairY4', 'LateHairY5'], gain: 1 },
  { key: 'hLSX', params: ['LateSideHairX1', 'LateSideHairX2', 'LateSideHairX3', 'LateSideHairX4', 'LateSideHairX5'], gain: 1 },
  { key: 'hLSY', params: ['LateSideHairY1', 'LateSideHairY2', 'LateSideHairY3', 'LateSideHairY4', 'LateSideHairY5'], gain: 1 },
];
export const HAIR_DIMS: Dim[] = HAIR_GROUPS.map((g) => ({ key: g.key, lo: -1, hi: 1, sd: 0.15, tau: 0.3 }));

/**
 * 姿态输出的 postPhysics set 轨（第二轮）：实测物理阶跃响应——AngleX→ParamAngleX 一帧到位，但 AngleZ→ParamAngleZ、
 * Position*→ParamBodyAngle*、LeanInput 有 5~15 帧的低通滞后（tools/fit/scripts/probe_step.ts）。逐帧拟合时，
 * 帧 f 的输入几乎控制不了帧 f 的姿态，锚点的「静止」假设也与动态历史对不上。所以把头、身体的主角度输出直接写成
 * postPhysics set（值 = 输入的稳态映射），物理输入照常写（驱动头发、衣物等次级物理）。
 */
export const POSE_POST_IDS = ['ParamAngleX', 'ParamAngleY', 'ParamAngleZ', 'ParamBodyAngleX', 'ParamBodyAngleY', 'ParamBodyAngleZ'];

/** 写进场景的逐帧参数轨（prePhysics、set） */
export const FIT_PARAM_IDS = [
  'AngleX', 'AngleY', 'AngleZ', 'PositionX', 'PositionY', 'PositionZ', 'ParamBreath',
  'ParamEyeLOpen', 'ParamEyeROpen', 'EyeSmile', 'ParamEyeBallX', 'ParamEyeBallY',
  'ParamBrowLY', 'ParamBrowRY', 'ParamMouthOpenY', 'ParamMouthForm',
] as const;

export const DEFAULT_STATE: Record<string, number> = {
  hx: 0, hy: 0, hz: 0, bx: 0, by: 0, bz: 0, eyeL: 1, eyeR: 1, es: 0, ebx: 0, eby: 0, brow: 0, mo: 0, mf: 0, blur: 0, opacity: 1,
  'c:LeanInput': 0, 'c:Clothes2': 0, 'c:breathPhase': 0, 'c:breathPeriod': 3.5, 'c:blur': 0.6,
  ...Object.fromEntries(HAIR_GROUPS.map((g) => [g.key, 0])),
};

/** 物理输入（控制器参数）：帧 f 的画面由帧 f−1 的值决定（Cubism 物理输出滞后一帧），拟合时写在 f−1 与 f */
export const PHYS_INPUT_IDS = ['AngleX', 'AngleY', 'AngleZ', 'PositionX', 'PositionY', 'PositionZ', 'ParamBreath', 'LeanInput', 'Clothes2'];

const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));

export function breathAt(s: Record<string, number>, frame: number): number {
  const T = s['c:breathPeriod'] ?? 3.5, ph = s['c:breathPhase'] ?? 0;
  return 0.5 + 0.5 * Math.sin((2 * Math.PI * frame) / 60 / T + ph);
}

/** 状态（具名）→ 帧 g 的模型输入参数 */
export function toParams(s: Record<string, number>, g: number): Record<string, number> {
  return {
    AngleX: clamp(s.hx!, -30, 30),
    AngleY: clamp(1.5 * s.hy!, -30, 30),
    AngleZ: clamp(s.hz!, -30, 30),
    PositionX: clamp(s.bx ?? 0, -30, 30),
    PositionY: clamp(s.by ?? 0, -30, 30),
    PositionZ: clamp(s.bz ?? 0, -30, 30),
    ParamBreath: breathAt(s, g),
    ParamEyeLOpen: s.eyeL!,
    ParamEyeROpen: s.eyeR!,
    EyeSmile: s.es!,
    ParamEyeBallX: s.ebx!,
    ParamEyeBallY: s.eby!,
    ParamBrowLY: s.brow!,
    ParamBrowRY: s.brow!,
    ParamMouthOpenY: s.mo!,
    ParamMouthForm: s.mf!,
    // 姿态输出（postPhysics set）：稳态增益取自阶跃响应（AngleX/Y/Z → 28/30，PositionX/Y → 1，PositionZ → −0.7）
    'post:ParamAngleX': clamp(s.hx!, -30, 30) * (28 / 30),
    'post:ParamAngleY': clamp(1.5 * s.hy!, -30, 30) * (28 / 30),
    'post:ParamAngleZ': clamp(s.hz!, -30, 30) * (28 / 30),
    'post:ParamBodyAngleX': clamp(s.bx ?? 0, -30, 30),
    'post:ParamBodyAngleY': clamp(s.by ?? 0, -30, 30),
    'post:ParamBodyAngleZ': clamp(s.bz ?? 0, -30, 30) * -0.7,
  };
}

/** 头发组叠加 → postPhysics add 轨的值 */
export function hairParams(s: Record<string, number>): Record<string, number> {
  const out: Record<string, number> = {};
  for (const g of HAIR_GROUPS) for (const p of g.params) out[p] = (s[g.key] ?? 0) * g.gain;
  return out;
}
