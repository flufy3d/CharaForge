/*
 * 内置演示场景 `?scene=demo:N`：N 个实例同屏（用于多实例帧率测量）。
 * 每个实例：IDLE 循环 + 不同相位的转头参数 + 物理，1280×720。
 */
const MODEL_DIR = 'models/deepseek-chan/DeepSeek';

export function demoScene(n: number): object {
  const W = 1280;
  const H = 720;
  const dur = 3600; // 60 秒，够基准测试用
  const layers = Array.from({ length: n }, (_, i) => {
    const x = ((i + 0.5) / n) * W;
    const phase = (i * 37) % 120;
    const keys = Array.from({ length: 31 }, (_, k) => k).map((k) => ({ t: phase + k * 120, v: k % 2 ? 25 : -25, ease: 'smooth' as const }));
    return {
      id: `m${i}`,
      kind: 'model',
      model: 'ds',
      in: 0,
      out: dur,
      transform: { x, y: H / 2, scale: Math.min(0.09, (W / n) / 3749) },  // 模型画布约 3749×7697，锚点缺省为画布中心
      physics: { warmup: 60 },
      motions: [{ motion: 'idle', start: 0, loop: true, fadeIn: 0 }],
      params: [
        { id: 'ParamAngleX', curve: { keys } },
        { id: 'ParamBodyAngleX', curve: { keys: keys.map((k) => ({ ...k, v: k.v / 3 })) } },
      ],
    };
  });
  return {
    format: 'charaforge.scene',
    version: 1,
    meta: {
      title: `演示：${n} 个实例同屏`,
      credits: ['DeepSeek-chan — Illustration & PSD: Kamiya Ryoko / Live2D Rigging: 霓虹猫_NeonNeko'],
    },
    timebase: { fps: 60 },
    duration: dur,
    composition: { width: W, height: H, background: '#f4f6fa' },
    assets: {
      models: [{ id: 'ds', uri: `${MODEL_DIR}/大肥鱼.model3.json` }],
      motions: [{ id: 'idle', model: 'ds', uri: `${MODEL_DIR}/Animation/IDLE.motion3.json` }],
      audio: [{ id: 'bgm', uri: 'video/DeepSeek-chan.wav' }],
    },
    layers,
    audio: [{ audio: 'bgm', start: 0 }],
    markers: [{ id: 'all', start: 0, end: dur, kind: 'shot', label: `${n} instances` }],
  };
}
