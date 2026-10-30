import { Type, type Static, type TSchema } from '@sinclair/typebox';

/**
 * 时间一律以「帧」为单位，帧率由文档的 timebase.fps 决定。
 * 允许小数帧（编辑器拖拽产生），但播放和拟合只在整数帧上求值。
 */
export const Frame = Type.Number({ minimum: 0, description: '帧号（以 timebase.fps 为单位）' });

export const Id = Type.String({ minLength: 1, pattern: '^[A-Za-z0-9_.:-]+$', description: '文档内唯一标识' });

/**
 * 语义标签（核心功能 10 预留）。任何片段、轨道、图层、标记都可以挂。
 * 本期只存不用；第二期的 AI 生成和行为语义系统会读写这里。
 */
export const Semantics = Type.Object(
  {
    tags: Type.Optional(Type.Array(Type.String(), { description: '自由标签，如 "greeting"、"shy"' })),
    emotion: Type.Optional(Type.String({ description: '情绪，如 "happy"、"embarrassed"' })),
    intent: Type.Optional(Type.String({ description: '行为意图，如 "point_at_viewer"' })),
    intensity: Type.Optional(Type.Number({ minimum: 0, maximum: 1 })),
    description: Type.Optional(Type.String({ description: '自然语言描述，供 AI 检索和生成使用' })),
  },
  { additionalProperties: true, description: '语义信息（预留，允许扩展字段）' },
);

/** 来源信息：这段数据是谁产生的（手工、拟合、AI 生成），便于第二期混合编辑。 */
export const Provenance = Type.Object(
  {
    source: Type.Union([Type.Literal('manual'), Type.Literal('fit'), Type.Literal('ai'), Type.Literal('import')]),
    tool: Type.Optional(Type.String()),
    detail: Type.Optional(Type.Record(Type.String(), Type.Unknown())),
  },
  { additionalProperties: false },
);

/**
 * 缓动：描述「从本关键帧到下一个关键帧」这一段怎么插值。
 * - step：保持本帧值直到下一个关键帧
 * - linear：线性
 * - smooth：等价于 cubic-bezier(0.42,0,0.58,1)
 * - [x1,y1,x2,y2]：CSS 风格的归一化三次贝塞尔
 */
export const Ease = Type.Union([
  Type.Literal('step'),
  Type.Literal('linear'),
  Type.Literal('smooth'),
  Type.Tuple([Type.Number(), Type.Number(), Type.Number(), Type.Number()]),
]);

const keyframe = <T extends TSchema>(v: T) =>
  Type.Object(
    { t: Frame, v, ease: Type.Optional(Ease) },
    { additionalProperties: false, description: '关键帧；ease 缺省为 linear' },
  );

const baked = <T extends TSchema>(v: T) =>
  Type.Object(
    {
      start: Frame,
      values: Type.Array(v, { description: 'values[i] 对应帧 start+i；超出范围时取首/尾值' }),
    },
    { additionalProperties: false },
  );

/**
 * 曲线：关键帧或逐帧烘焙，二选一。
 * 关键帧按 t 升序；t 在首帧之前取首值，末帧之后取末值。
 */
export const curveOf = <T extends TSchema>(v: T) =>
  Type.Union([
    Type.Object({ keys: Type.Array(keyframe(v), { minItems: 1 }) }, { additionalProperties: false }),
    Type.Object({ baked: baked(v) }, { additionalProperties: false }),
  ]);

/** 可动画属性：常量，或曲线。 */
export const animatable = <T extends TSchema>(v: T) => Type.Union([v, curveOf(v)]);

export const Vec3 = Type.Tuple([Type.Number(), Type.Number(), Type.Number()]);

export const AnimNumber = animatable(Type.Number());
export const AnimVec3 = animatable(Vec3);
export const CurveNumber = curveOf(Type.Number());

export type Frame = Static<typeof Frame>;
export type Semantics = Static<typeof Semantics>;
export type Provenance = Static<typeof Provenance>;
export type Ease = Static<typeof Ease>;
export type Vec3 = Static<typeof Vec3>;
export type Keyframe<V> = { t: number; v: V; ease?: Ease };
export type Curve<V> = { keys: Keyframe<V>[] } | { baked: { start: number; values: V[] } };
export type Animatable<V> = V | Curve<V>;
