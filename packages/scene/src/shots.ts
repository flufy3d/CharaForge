import { Type, type Static } from '@sinclair/typebox';
import { Frame, Id } from './common.ts';

/*
 * 镜头表格式（charaforge.shots v1）：tools/recon 的产出，tools/fit 和 tools/qa 的输入。
 * 它描述「参考视频里有什么」，不描述怎么渲染。所有坐标都是视频像素（原点左上，y 向下），
 * 所有时间都是视频帧号（0 起，区间一律左闭右开）。
 *
 * 这里的几何量都是粗估值，用来给拟合提供初值和约束；精确的变换由 tools/fit 求解。
 */

export const Rect = Type.Tuple([Type.Number(), Type.Number(), Type.Number(), Type.Number()], {
  description: '[x, y, width, height]，视频像素',
});

export const Confidence = Type.Number({ minimum: 0, maximum: 1 });

/** 稀疏关键帧采样，相邻采样之间线性插值，区间外保持首/尾值。 */
export const TrackSample = Type.Object(
  {
    frame: Frame,
    bbox: Type.Optional(Type.Union([Rect, Type.Null()], { description: '角色可见部分的外接框；null 表示该帧不可见' })),
    head: Type.Optional(Type.Union([Rect, Type.Null()], { description: '头部（含头发，不含发尾）的外接框' })),
    eyes: Type.Optional(
      Type.Tuple([Type.Tuple([Type.Number(), Type.Number()]), Type.Tuple([Type.Number(), Type.Number()])], {
        description: '画面左侧眼睛中心、右侧眼睛中心，视频像素',
      }),
    ),
    rotation: Type.Optional(Type.Number({ description: '画面内整体旋转（度，顺时针为正）' })),
    opacity: Type.Optional(Type.Number({ minimum: 0, maximum: 1 })),
    blurSigma: Type.Optional(Type.Number({ minimum: 0, description: '估计的高斯模糊 sigma（像素）' })),
  },
  { additionalProperties: false },
);

export const SwitchEvent = Type.Object(
  {
    frame: Frame,
    param: Type.String({ description: '开关参数 ID，如 key10、Clothes1、Movement3' }),
    value: Type.Number(),
    confidence: Type.Optional(Confidence),
    note: Type.Optional(Type.String()),
  },
  { additionalProperties: false },
);

export const MotionHint = Type.Object(
  {
    motion: Type.String({ description: 'motion3 文件名（不含扩展名），如 "脱捕"' }),
    start: Frame,
    confidence: Type.Optional(Confidence),
    note: Type.Optional(Type.String()),
  },
  { additionalProperties: false },
);

export const ShotInstance = Type.Object(
  {
    id: Id,
    role: Type.String({ description: '如 main、pip、side、chibi、fullBody、background' }),
    z: Type.Integer({ description: '叠放顺序，越大越靠前' }),
    variant: Type.Record(Type.String(), Type.Number(), {
      description: '镜头开始时开关类参数的取值（key*、Clothes*、Movement*），只列非默认值',
    }),
    switchEvents: Type.Array(SwitchEvent, { description: '镜头内的开关切换' }),
    motionHints: Type.Array(MotionHint),
    track: Type.Array(TrackSample, { minItems: 1 }),
    clip: Type.Optional(
      Type.Union([
        Type.Object(
          { rect: Rect, radius: Type.Optional(Type.Number({ minimum: 0 })) },
          { additionalProperties: false, description: '画中画等窗口裁切（屏幕空间，镜头内固定）' },
        ),
        Type.Null(),
      ]),
    ),
    flipX: Type.Optional(Type.Boolean()),
    colorGrade: Type.Optional(
      Type.String({ description: '对角色本身的调色/压暗的文字描述，如 "整体压暗约 40%、偏蓝"' }),
    ),
    notes: Type.Optional(Type.String()),
  },
  { additionalProperties: false },
);

export const Exclusion = Type.Object(
  {
    start: Frame,
    end: Frame,
    reason: Type.String({ description: '如 finger、bubble、check、cursor、text、effect、silhouette' }),
    rects: Type.Array(Type.Object({ frame: Frame, rect: Rect }, { additionalProperties: false }), {
      minItems: 1,
      description: '遮挡区域关键帧，线性插值；拟合和质检时这些区域不计入角色遮罩',
    }),
  },
  { additionalProperties: false },
);

export const Transition = Type.Object(
  {
    type: Type.Union([
      Type.Literal('cut'),
      Type.Literal('dissolve'),
      Type.Literal('fadeFromBlack'),
      Type.Literal('fadeFromWhite'),
      Type.Literal('wipe'),
      Type.Literal('other'),
    ]),
    duration: Type.Integer({ minimum: 0, description: '过渡持续帧数，硬切为 0' }),
    note: Type.Optional(Type.String()),
  },
  { additionalProperties: false },
);

export const Shot = Type.Object(
  {
    id: Id,
    start: Frame,
    end: Frame,
    transitionIn: Transition,
    category: Type.String({
      description: '如 halfBody、closeUp、extremeCloseUp、fullBody、multi、pip、noCharacter',
    }),
    hasCharacter: Type.Boolean({ description: 'false 表示纯 UI/转场镜头，不参与拟合和质检' }),
    instances: Type.Array(ShotInstance),
    exclusions: Type.Array(Exclusion),
    effects: Type.Array(
      Type.Object(
        {
          kind: Type.String({ description: '如 motionBlur、cameraShake、vignette、colorGrade、silhouette、particles' }),
          start: Frame,
          end: Frame,
          note: Type.Optional(Type.String()),
        },
        { additionalProperties: false },
      ),
      { description: '镜头内会影响角色像素的效果，供拟合/质检参考' },
    ),
    thumbnail: Type.Optional(Type.String({ description: '缩略图拼图路径，相对于 shots.json 所在目录' })),
    confidence: Type.Optional(Confidence),
    notes: Type.Optional(Type.String()),
  },
  { additionalProperties: false },
);

export const Shots = Type.Object(
  {
    $schema: Type.Optional(Type.String()),
    format: Type.Literal('charaforge.shots'),
    version: Type.Literal(1),
    video: Type.Object(
      {
        uri: Type.String({ description: '相对于仓库根目录的路径' }),
        sha256: Type.String(),
        width: Type.Integer(),
        height: Type.Integer(),
        fps: Type.Number(),
        frameCount: Type.Integer(),
      },
      { additionalProperties: false },
    ),
    shots: Type.Array(Shot, { description: '按时间排序，首尾相接覆盖 [0, frameCount)' }),
  },
  { additionalProperties: false, $id: 'https://charaforge.dev/schema/shots.v1.json', title: 'CharaForge Shots v1' },
);

export type Rect = Static<typeof Rect>;
export type TrackSample = Static<typeof TrackSample>;
export type SwitchEvent = Static<typeof SwitchEvent>;
export type MotionHint = Static<typeof MotionHint>;
export type ShotInstance = Static<typeof ShotInstance>;
export type Exclusion = Static<typeof Exclusion>;
export type Transition = Static<typeof Transition>;
export type Shot = Static<typeof Shot>;
export type Shots = Static<typeof Shots>;
