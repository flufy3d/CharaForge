/*
 * 拟合参数集合（每个模型图层一份向量）。
 *
 * 依据：大肥鱼.vtube.json 的 ParameterSettings —— 原视频里由面捕连续驱动的就是这一组（VTube Studio 的输入 → 模型参数）：
 *   FaceAngleX → AngleX、PositionX（同一输入，量程 ±30 → ±30）
 *   FaceAngleY → AngleY（±20 → ±30）、PositionY（±30 → ±30）
 *   FaceAngleZ → AngleZ、PositionZ（±30 → ±30）
 *   EyeOpenLeft/Right → ParamEyeLOpen/ROpen（0..1.5）；EyeRightX/Y → ParamEyeBallX/Y
 *   Brows → ParamBrowLY、ParamBrowRY（同一输入）；MouthSmile → EyeSmile、ParamMouthForm；MouthOpen → ParamMouthOpenY
 * 注意 AngleX/Y/Z、PositionX/Y/Z 是「控制器」参数，ParamAngleX 等由物理（PhysicsSetting1~3 等）从它们算出，
 * 所以拟合的是物理输入，物理正向推进。头身联动（Position*）按 VTS 映射与头部角度绑定，减少自由度。
 * EyeSmile 与 ParamMouthForm 在 VTS 里来自同一输入，但两者的量程映射未必与面捕实际值吻合，这里分开拟合。
 * 其余 ARKit 细项（MouthX、CheekPuff、舌头、嘴唇卷曲等）在这些镜头的画面尺寸下几乎不可辨，保持默认值。
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

/** 写进场景的参数轨（全部 prePhysics、set、逐帧 baked） */
export const FIT_PARAM_IDS = [
  'AngleX', 'AngleY', 'AngleZ', 'PositionX', 'PositionY', 'PositionZ',
  'ParamEyeLOpen', 'ParamEyeROpen', 'EyeSmile', 'ParamEyeBallX', 'ParamEyeBallY',
  'ParamBrowLY', 'ParamBrowRY', 'ParamMouthOpenY', 'ParamMouthForm',
] as const;

export const DEFAULT_STATE: Record<string, number> = {
  hx: 0, hy: 0, hz: 0, eyeL: 1, eyeR: 1, es: 0, ebx: 0, eby: 0, brow: 0, mo: 0, mf: 0, blur: 0, opacity: 1,
};

/** 物理输入（控制器参数）：帧 f 的画面由帧 f−1 的值决定（Cubism 物理输出滞后一帧），拟合时写在 f−1 与 f */
export const PHYS_INPUT_IDS = ['AngleX', 'AngleY', 'AngleZ', 'PositionX', 'PositionY', 'PositionZ'];

const clamp = (v: number, a: number, b: number) => Math.min(b, Math.max(a, v));

/** 状态（具名）→ 模型输入参数 */
export function toParams(s: Record<string, number>): Record<string, number> {
  return {
    AngleX: clamp(s.hx!, -30, 30),
    PositionX: clamp(s.hx!, -30, 30),
    AngleY: clamp(1.5 * s.hy!, -30, 30),
    PositionY: clamp(s.hy!, -30, 30),
    AngleZ: clamp(s.hz!, -30, 30),
    PositionZ: clamp(s.hz!, -30, 30),
    ParamEyeLOpen: s.eyeL!,
    ParamEyeROpen: s.eyeR!,
    EyeSmile: s.es!,
    ParamEyeBallX: s.ebx!,
    ParamEyeBallY: s.eby!,
    ParamBrowLY: s.brow!,
    ParamBrowRY: s.brow!,
    ParamMouthOpenY: s.mo!,
    ParamMouthForm: s.mf!,
  };
}
