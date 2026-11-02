/*
 * VTube Studio 前向模型：面捕输入 → 区间线性映射（不夹紧，模型参数本身由 Cubism 夹到范围内）→ 平滑 → 模型参数。
 * 映射表读自模型的 *.vtube.json（ParameterSettings），配置即数据，换模型同样适用。
 *
 * 平滑：VTS 的 Smoothing（0~100）公式未公开，按指数平滑建模（60 fps 逐帧）：
 *   y_f = y_{f−1} + α(s)·(x_f − y_{f−1})，α(s) = 1 / (1 + c·s)，c 为全片共用的全局常量（s = 0 时不平滑）。
 * 自动呼吸（UseBreathing）：ParamBreath = 0.5 + 0.5·sin(2π·t/T + φ)，T、φ 为镜头常量。
 */
import { readFileSync } from 'node:fs';
import { MODELX } from './modelx.ts';

export interface VtsSetting {
  name: string;
  input: string;
  output: string;
  inLo: number;
  inHi: number;
  outLo: number;
  outHi: number;
  smoothing: number;
  breathing: boolean;
}

export interface VtsModel {
  settings: VtsSetting[];
  /** 本模块拟合的面捕输入（其余输入固定在中性值） */
  inputs: string[];
  /** 各输入的中性值（映射到模型参数缺省值的那个输入值；没有映射的取 0） */
  neutral: Record<string, number>;
}

/** 面捕输入的拟合范围与时间平滑尺度（tau：相邻节点间变化的「正常」幅度） */
export const INPUT_DIMS: Record<string, { lo: number; hi: number; sd: number; tau: number }> = {
  FaceAngleX: { lo: -40, hi: 40, sd: 3, tau: 8 },
  FaceAngleY: { lo: -30, hi: 30, sd: 2, tau: 6 },
  FaceAngleZ: { lo: -40, hi: 40, sd: 3, tau: 8 },
  EyeOpenLeft: { lo: 0, hi: 1, sd: 0.06, tau: 0.4 },
  EyeOpenRight: { lo: 0, hi: 1, sd: 0.06, tau: 0.4 },
  EyeRightX: { lo: -0.8, hi: 0.8, sd: 0.05, tau: 0.3 },
  EyeRightY: { lo: -0.9, hi: 0.9, sd: 0.05, tau: 0.3 },
  MouthSmile: { lo: -0.2, hi: 1.2, sd: 0.06, tau: 0.3 },
  MouthOpen: { lo: 0, hi: 1, sd: 0.05, tau: 0.3 },
  Brows: { lo: 0, hi: 0.9, sd: 0.04, tau: 0.2 },
  // MODELX.decouple：身体独立输入（映射同 vtube.json 里 PositionX/Y/Z 那几行，只换输入源）
  BodyX: { lo: -40, hi: 40, sd: 3, tau: 8 },
  BodyY: { lo: -40, hi: 40, sd: 3, tau: 8 },
  BodyZ: { lo: -40, hi: 40, sd: 3, tau: 8 },
};

export function loadVts(path: string, modelDefaults: Record<string, number>): VtsModel {
  const v = JSON.parse(readFileSync(path, 'utf8'));
  const settings: VtsSetting[] = v.ParameterSettings.map((p: any) => ({
    name: p.Name, input: p.Input ?? '', output: p.OutputLive2D, inLo: p.InputRangeLower, inHi: p.InputRangeUpper,
    outLo: p.OutputRangeLower, outHi: p.OutputRangeUpper, smoothing: p.Smoothing ?? 0, breathing: !!p.UseBreathing,
  }));
  if (MODELX.decouple) for (const s of settings) if (/^Position[XYZ]$/.test(s.output)) s.input = `Body${s.output.slice(-1)}`;
  const neutral: Record<string, number> = {};
  for (const s of settings) {
    if (!s.input || neutral[s.input] !== undefined) continue;
    const d = modelDefaults[s.output] ?? 0;
    neutral[s.input] = s.outHi === s.outLo ? s.inLo : s.inLo + ((d - s.outLo) / (s.outHi - s.outLo)) * (s.inHi - s.inLo);
  }
  return { settings, inputs: Object.keys(INPUT_DIMS).filter((k) => settings.some((s) => s.input === k)), neutral };
}

export const alphaOf = (smoothing: number, c: number) => 1 / (1 + c * smoothing);

export interface VtsConsts {
  /** 全局平滑系数 c */
  smoothC: number;
  breathPeriod: number;
  breathPhase: number;
}

/**
 * 逐帧输入（frames × 每个输入）→ 逐帧模型参数。inputs[name][i] 对应帧 f0 + i。
 * 平滑从首帧的映射值起步（等价于在首帧之前已静止）。
 */
export function forward(m: VtsModel, inputs: Record<string, Float64Array>, n: number, f0: number, k: VtsConsts): Record<string, Float64Array> {
  const out: Record<string, Float64Array> = {};
  for (const s of m.settings) {
    const y = new Float64Array(n);
    if (s.breathing) {
      for (let i = 0; i < n; i++) y[i] = 0.5 + 0.5 * Math.sin((2 * Math.PI * (f0 + i)) / 60 / k.breathPeriod + k.breathPhase);
      out[s.output] = y;
      continue;
    }
    const x = inputs[s.input];
    const slope = (s.outHi - s.outLo) / (s.inHi - s.inLo);
    const map = (u: number) => s.outLo + (u - s.inLo) * slope;
    const a = alphaOf(s.smoothing, k.smoothC);
    let prev = map(x ? x[0]! : m.neutral[s.input] ?? 0);
    for (let i = 0; i < n; i++) {
      const v = map(x ? x[i]! : m.neutral[s.input] ?? 0);
      prev = prev + a * (v - prev);
      y[i] = prev;
    }
    out[s.output] = y;
  }
  return out;
}

/**
 * 线性灵敏度：模型参数 output 在帧 i 对「输入 input 在帧 j 的值」的偏导（映射斜率 × 指数平滑的脉冲响应）。
 * 返回 (output, input, slope, alpha) 列表，调用方用 alpha 生成卷积核。
 */
export function linearLinks(m: VtsModel, smoothC: number): { output: string; input: string; slope: number; alpha: number }[] {
  return m.settings
    .filter((s) => s.input && !s.breathing)
    .map((s) => ({ output: s.output, input: s.input, slope: (s.outHi - s.outLo) / (s.inHi - s.inLo), alpha: alphaOf(s.smoothing, smoothC) }));
}
