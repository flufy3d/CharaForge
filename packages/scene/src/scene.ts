import { Type, type Static } from '@sinclair/typebox';
import { AnimNumber, AnimVec3, CurveNumber, Frame, Id, Provenance, Semantics } from './common.ts';

/*
 * CharaForge 场景格式（charaforge.scene v1）
 *
 * 一个场景 = 合成画布 + 资源表 + 相机 + 若干图层（角色实例）+ 音频 + 标记。
 * 规范性约定（播放器、拟合器、编辑器必须一致）：
 *
 * 1. 时间：所有时间以帧为单位，帧率为 timebase.fps。帧 f 的时刻为 f / fps 秒。
 *    曲线关键帧、clip、片段的 start/end 等一律是场景绝对帧号（不是相对图层 in）。
 *    例外：motion/expression 片段的 offset、fadeIn、fadeOut 以秒为单位。
 *
 * 2. 坐标：
 *    - 模型空间 = moc3 画布像素坐标，原点在画布左上角，y 向下，单位为画布像素
 *      （画布尺寸 = CanvasInfo.Size，像素/单位换算 = CanvasInfo.PixelsPerUnit）。
 *    - 世界空间 = 合成像素坐标系（与画布同尺寸），相机作用之前。
 *    - 屏幕空间 = 合成像素坐标，原点左上，y 向下。
 *    - 图层：world = [x,y] + R(rotation) · diag(sx, sy) · (p − [anchorX, anchorY])，
 *      其中 sx = scale · scaleX · (flipX ? −1 : 1)，sy = scale · scaleY（scaleX、scaleY 缺省 1），
 *      rotation 单位为度、顺时针为正（y 向下）。
 *    - 相机：screen = [W/2, H/2] + zoom · R(rotation) · (world − [x, y])。
 *      space = "screen" 的图层不受相机影响（world 直接当作 screen）。
 *
 * 3. 单帧求值顺序（每个图层独立）：
 *    a. 所有参数复位为模型默认值，部件不透明度复位为默认值；
 *    b. 按数组顺序叠加处于激活区间的 motion 片段（Cubism motion 语义，含淡入淡出）；
 *    c. 按数组顺序叠加 expression 片段（exp3 的 Add / Multiply / Overwrite 语义）；
 *    d. 按数组顺序应用 stage = "prePhysics" 的参数轨；
 *    e. 物理前进一步，dt = 1 / fps（固定步长，不用墙钟时间）；
 *    f. 按数组顺序应用 stage = "postPhysics" 的参数轨（用于烘焙物理输出或强制覆盖）；
 *    g. 应用部件不透明度轨，更新模型并绘制。
 *    参数轨 mode：set → v' = v + (x − v)·w；add → v' = v + x·w；multiply → v' = v·(1 + (x − 1)·w)，
 *    w 为 weight（缺省 1）。
 *    不存在任何隐式的自动行为（自动眨眼、呼吸、口型同步等），一切都要显式写成轨道。
 *
 * 4. 物理状态与确定性：图层在帧 f 的状态，定义为从帧 (in − physics.warmup) 开始
 *    逐帧按上述顺序推进到 f 的结果；预热阶段使用帧 in 的输入。跳转（seek）时必须重放，
 *    使任意访问路径得到逐位相同的结果。
 *
 * 3b. motion / expression 片段（片段时间 τ = (f − start)/fps 秒）：
 *    - motion 内部时间 = offset + τ·speed；loop 缺省 false，循环时每圈不重新淡入；
 *    - 淡入：从 start 起 fadeIn 秒（缺省取文件自带值，文件未写时为 1 秒），Cubism 正弦缓动；
 *    - 结束时刻 = end（若给出）；否则非循环 motion 为自然结束 (duration − offset)/speed，
 *      循环 motion 与 expression 无结束（持续到图层 out，不淡出）；
 *    - 淡出：在结束时刻之前 fadeOut 秒内淡到 0；fade 时长不受 speed 影响；
 *    - 片段在帧 f 的效果只由 (片段定义, f) 决定，没有跨帧状态。
 *
 * 4b. 帧保持（frameHold）：在某段 [start, end) 内，图层在显示帧 f 呈现的画面
 *    等于它在帧 g = start + floor((f − start) / every) · every 的画面（所有属性都在 g 求值）。
 *    状态推进不受影响，仍然逐帧推进物理；它只改变「显示哪一帧」。用于还原低帧率素材。
 *    相机属于场景而非图层：图层在 g 呈现时，相机也取 g 时刻的值。
 *
 * 5. 合成：图层按数组顺序从下到上绘制。每个图层先渲染到独立的预乘 alpha 缓冲
 *    （合成分辨率），再依次应用 filters、clip、opacity，最后按 blendMode 合成。
 *    filters 的边缘处理、采样方式、clip 羽化的具体算法以 packages/runtime 的参考实现为准。
 */

// ---------- 资源 ----------

export const ModelAsset = Type.Object(
  {
    id: Id,
    uri: Type.String({ description: 'model3.json 路径，相对于资源根目录（assetRoot）' }),
    name: Type.Optional(Type.String()),
    credits: Type.Optional(Type.String({ description: '署名信息，播放器需要展示' })),
  },
  { additionalProperties: false },
);

export const MotionAsset = Type.Object(
  {
    id: Id,
    model: Id,
    uri: Type.String({ description: 'motion3.json 路径，相对于资源根目录' }),
    semantics: Type.Optional(Semantics),
  },
  { additionalProperties: false },
);

export const ExpressionAsset = Type.Object(
  {
    id: Id,
    model: Id,
    uri: Type.String({ description: 'exp3.json 路径，相对于资源根目录' }),
    semantics: Type.Optional(Semantics),
  },
  { additionalProperties: false },
);

export const AudioAsset = Type.Object(
  { id: Id, uri: Type.String({ description: '音频文件路径，相对于资源根目录' }) },
  { additionalProperties: false },
);

/**
 * 外部图片资源（饰品、贴纸等）。图片是不可信数据：运行时只解析和上传，不执行。
 * uri 相对于资源根目录；也允许 data:image/png;base64,... / data:image/webp;base64,...（示例、测试用）。
 * width/height 是图片像素尺寸，必须与实际解码结果一致；单边上限 4096，总像素上限 16 MP（16,777,216）。
 */
export const MAX_IMAGE_SIDE = 4096;
export const MAX_IMAGE_PIXELS = 16_777_216;

export const ImageAsset = Type.Object(
  {
    id: Id,
    uri: Type.String({ minLength: 1, description: '图片路径（相对于资源根目录）或 PNG/WebP data URI' }),
    width: Type.Integer({ minimum: 1, maximum: MAX_IMAGE_SIDE, description: '图片像素宽度' }),
    height: Type.Integer({ minimum: 1, maximum: MAX_IMAGE_SIDE, description: '图片像素高度' }),
    sha256: Type.Optional(Type.String({ pattern: '^[0-9a-f]{64}$', description: '文件内容摘要，供目录/去重使用' })),
  },
  { additionalProperties: false },
);

// ---------- 轨道与片段 ----------

export const ParamTrack = Type.Object(
  {
    id: Type.String({ description: '模型参数 ID，如 ParamAngleX、key10' }),
    curve: CurveNumber,
    mode: Type.Optional(
      Type.Union([Type.Literal('set'), Type.Literal('add'), Type.Literal('multiply')], {
        description: '缺省 set',
      }),
    ),
    stage: Type.Optional(
      Type.Union([Type.Literal('prePhysics'), Type.Literal('postPhysics')], { description: '缺省 prePhysics' }),
    ),
    weight: Type.Optional(AnimNumber),
    range: Type.Optional(
      Type.Tuple([Frame, Frame], { description: '仅在 [start, end) 内生效；缺省为整个图层区间' }),
    ),
    semantics: Type.Optional(Semantics),
    provenance: Type.Optional(Provenance),
  },
  { additionalProperties: false },
);

export const PartOpacityTrack = Type.Object(
  { id: Type.String({ description: '部件 ID' }), curve: CurveNumber, provenance: Type.Optional(Provenance) },
  { additionalProperties: false },
);

export const MotionClip = Type.Object(
  {
    motion: Id,
    start: Frame,
    end: Type.Optional(Type.Number({ description: '结束帧（不含）；缺省为 motion 自然结束或图层结束' })),
    offset: Type.Optional(Type.Number({ description: '从 motion 的第几秒开始播放，缺省 0' })),
    speed: Type.Optional(Type.Number({ exclusiveMinimum: 0, description: '缺省 1' })),
    loop: Type.Optional(Type.Boolean()),
    weight: Type.Optional(AnimNumber),
    fadeIn: Type.Optional(Type.Number({ minimum: 0, description: '秒；缺省用 motion3 自带值' })),
    fadeOut: Type.Optional(Type.Number({ minimum: 0, description: '秒；缺省用 motion3 自带值' })),
    semantics: Type.Optional(Semantics),
    provenance: Type.Optional(Provenance),
  },
  { additionalProperties: false },
);

export const ExpressionClip = Type.Object(
  {
    expression: Id,
    start: Frame,
    end: Type.Optional(Type.Number({ description: '结束帧（不含）；缺省持续到图层结束' })),
    weight: Type.Optional(AnimNumber),
    fadeIn: Type.Optional(Type.Number({ minimum: 0, description: '秒；缺省用 exp3 自带值' })),
    fadeOut: Type.Optional(Type.Number({ minimum: 0, description: '秒；缺省用 exp3 自带值' })),
    semantics: Type.Optional(Semantics),
    provenance: Type.Optional(Provenance),
  },
  { additionalProperties: false },
);

// ---------- 图层属性 ----------

export const Transform2D = Type.Object(
  {
    x: AnimNumber,
    y: AnimNumber,
    scale: AnimNumber,
    scaleX: Type.Optional(AnimNumber),
    scaleY: Type.Optional(AnimNumber),
    rotation: Type.Optional(AnimNumber),
    anchorX: Type.Optional(Type.Number({ description: '图层本地像素（模型画布或图片）；缺省为对应画布/图片中心' })),
    anchorY: Type.Optional(Type.Number({ description: '图层本地像素（模型画布或图片）；缺省为对应画布/图片中心' })),
    flipX: Type.Optional(Type.Boolean()),
  },
  { additionalProperties: false },
);

export const CameraTransform = Type.Object(
  {
    x: AnimNumber,
    y: AnimNumber,
    zoom: AnimNumber,
    rotation: Type.Optional(AnimNumber),
  },
  { additionalProperties: false },
);

export const Clip = Type.Object(
  {
    x: AnimNumber,
    y: AnimNumber,
    width: AnimNumber,
    height: AnimNumber,
    radius: Type.Optional(AnimNumber),
    feather: Type.Optional(AnimNumber),
  },
  { additionalProperties: false, description: '屏幕空间圆角矩形裁切' },
);

export const BlurFilter = Type.Object(
  {
    type: Type.Literal('blur'),
    sigma: AnimNumber,
    enabled: Type.Optional(Type.Boolean()),
  },
  { additionalProperties: false, description: '高斯模糊，sigma 为屏幕像素' },
);

export const MotionBlurFilter = Type.Object(
  {
    type: Type.Literal('motionBlur'),
    angle: AnimNumber,
    length: AnimNumber,
    enabled: Type.Optional(Type.Boolean()),
  },
  { additionalProperties: false, description: '方向模糊：angle 为度，length 为屏幕像素' },
);

export const ColorFilter = Type.Object(
  {
    type: Type.Literal('color'),
    brightness: Type.Optional(AnimNumber),
    contrast: Type.Optional(AnimNumber),
    saturation: Type.Optional(AnimNumber),
    multiply: Type.Optional(AnimVec3),
    offset: Type.Optional(AnimVec3),
    enabled: Type.Optional(Type.Boolean()),
  },
  {
    additionalProperties: false,
    description:
      '颜色调整，作用于非预乘颜色，按顺序：c = (c − 0.5)·contrast + 0.5 + brightness；' +
      '饱和度按 Rec.709 亮度插值；再 c = c·multiply + offset。缺省值 brightness 0、contrast 1、saturation 1、multiply [1,1,1]、offset [0,0,0]',
  },
);

export const ResampleFilter = Type.Object(
  {
    type: Type.Literal('resample'),
    factor: AnimNumber,
    method: Type.Optional(Type.Union([Type.Literal('bilinear'), Type.Literal('bicubic')], { description: '放大回原尺寸时的插值，缺省 bilinear' })),
    enabled: Type.Optional(Type.Boolean()),
  },
  {
    additionalProperties: false,
    description: '先按 factor（0~1）缩小（区域平均）再放大回原尺寸，模拟低分辨率素材放大后的软化',
  },
);

export const Filter = Type.Union([BlurFilter, MotionBlurFilter, ColorFilter, ResampleFilter]);

export const BlendMode = Type.Union([
  Type.Literal('normal'),
  Type.Literal('add'),
  Type.Literal('multiply'),
  Type.Literal('screen'),
]);

export const ModelLayer = Type.Object(
  {
    id: Id,
    name: Type.Optional(Type.String()),
    kind: Type.Literal('model'),
    model: Id,
    in: Frame,
    out: Type.Number({ description: '结束帧（不含）' }),
    space: Type.Optional(Type.Union([Type.Literal('world'), Type.Literal('screen')], { description: '缺省 world' })),
    transform: Transform2D,
    opacity: Type.Optional(AnimNumber),
    clip: Type.Optional(Clip),
    filters: Type.Optional(Type.Array(Filter)),
    blendMode: Type.Optional(BlendMode),
    frameHold: Type.Optional(
      Type.Array(
        Type.Object(
          { start: Frame, end: Frame, every: Type.Integer({ minimum: 2 }) },
          { additionalProperties: false },
        ),
        { description: '帧保持段，见文件头约定 4b；各段不得重叠' },
      ),
    ),
    physics: Type.Optional(
      Type.Object(
        {
          enabled: Type.Optional(Type.Boolean({ description: '缺省 true' })),
          warmup: Type.Optional(Type.Integer({ minimum: 0, description: '预热帧数，缺省 0' })),
          stabilize: Type.Optional(
            Type.Boolean({ description: '预热之前先用帧 in 的输入把物理置于静止平衡态（Cubism stabilization），缺省 false' }),
          ),
        },
        { additionalProperties: false },
      ),
    ),
    motions: Type.Optional(Type.Array(MotionClip)),
    expressions: Type.Optional(Type.Array(ExpressionClip)),
    params: Type.Optional(Type.Array(ParamTrack)),
    partOpacity: Type.Optional(Type.Array(PartOpacityTrack)),
    semantics: Type.Optional(Semantics),
    provenance: Type.Optional(Provenance),
  },
  { additionalProperties: false },
);

/**
 * 图片图层绑定到模型图层：只跟随目标图层的整体变换（平移/旋转/缩放/flip/相机），
 * 不解析骨骼、参数或部件。anchor 是目标模型画布的归一化坐标（0..1）。
 * offset 与 transform.x/y 在 attach 时都使用目标模型画布像素。
 */
export const ImageAttach = Type.Object(
  {
    layer: Id,
    anchor: Type.Tuple([Type.Number({ minimum: 0, maximum: 1 }), Type.Number({ minimum: 0, maximum: 1 })], {
      description: '目标模型画布归一化坐标 [x, y]，0..1',
    }),
    offset: Type.Optional(Type.Tuple([Type.Number(), Type.Number()], { description: '目标模型画布像素偏移' })),
    rotateWith: Type.Optional(Type.Boolean({ description: '是否继承目标旋转，缺省 true' })),
    scaleWith: Type.Optional(Type.Boolean({ description: '是否继承目标缩放和 flipX，缺省 true' })),
  },
  { additionalProperties: false },
);

/**
 * 图片图层。transform 与模型图层同语义，但 anchor 的单位是图片像素（缺省图片中心）。
 * 有 attach 时 space 被忽略，transform.x/y 与 offset 使用目标模型画布像素。
 */
export const ImageLayer = Type.Object(
  {
    id: Id,
    name: Type.Optional(Type.String()),
    kind: Type.Literal('image'),
    image: Id,
    in: Frame,
    out: Type.Number({ description: '结束帧（不含）' }),
    space: Type.Optional(
      Type.Union([Type.Literal('world'), Type.Literal('screen')], {
        description: '缺省 world；有 attach 时忽略，由目标图层的 space 决定',
      }),
    ),
    transform: Transform2D,
    opacity: Type.Optional(AnimNumber),
    clip: Type.Optional(Clip),
    filters: Type.Optional(Type.Array(Filter)),
    blendMode: Type.Optional(BlendMode),
    attach: Type.Optional(ImageAttach),
    semantics: Type.Optional(Semantics),
    provenance: Type.Optional(Provenance),
  },
  { additionalProperties: false },
);

/** 图层联合类型；后续可加入 group 等图层。 */
export const Layer = Type.Union([ModelLayer, ImageLayer]);

export const AudioClip = Type.Object(
  {
    audio: Id,
    start: Frame,
    offset: Type.Optional(Type.Number({ minimum: 0, description: '从音频第几秒开始，缺省 0' })),
    duration: Type.Optional(Type.Number({ minimum: 0, description: '秒；缺省播放到结束' })),
    gain: Type.Optional(Type.Number({ minimum: 0, description: '线性增益，缺省 1' })),
  },
  { additionalProperties: false },
);

export const Marker = Type.Object(
  {
    id: Id,
    start: Frame,
    end: Type.Optional(Type.Number()),
    label: Type.Optional(Type.String()),
    kind: Type.Optional(Type.String({ description: '如 "shot"、"beat"、"note"' })),
    semantics: Type.Optional(Semantics),
  },
  { additionalProperties: false },
);

export const Scene = Type.Object(
  {
    $schema: Type.Optional(Type.String()),
    format: Type.Literal('charaforge.scene'),
    version: Type.Literal(1),
    meta: Type.Object(
      {
        title: Type.String(),
        description: Type.Optional(Type.String()),
        credits: Type.Optional(Type.Array(Type.String())),
        provenance: Type.Optional(Provenance),
      },
      { additionalProperties: true },
    ),
    timebase: Type.Object({ fps: Type.Number({ exclusiveMinimum: 0 }) }, { additionalProperties: false }),
    duration: Type.Integer({ minimum: 1, description: '总帧数' }),
    composition: Type.Object(
      {
        width: Type.Integer({ minimum: 1 }),
        height: Type.Integer({ minimum: 1 }),
        background: Type.Optional(Type.Union([Type.String(), Type.Null()], { description: 'CSS 颜色；null 为透明' })),
        antialias: Type.Optional(Type.Boolean({ description: '模型绘制是否启用 MSAA，缺省 true；影响像素，拟合与播放必须一致' })),
      },
      { additionalProperties: false },
    ),
    assets: Type.Object(
      {
        models: Type.Array(ModelAsset),
        motions: Type.Optional(Type.Array(MotionAsset)),
        expressions: Type.Optional(Type.Array(ExpressionAsset)),
        audio: Type.Optional(Type.Array(AudioAsset)),
        images: Type.Optional(Type.Array(ImageAsset)),
      },
      { additionalProperties: false },
    ),
    camera: Type.Optional(CameraTransform),
    layers: Type.Array(Layer),
    audio: Type.Optional(Type.Array(AudioClip)),
    markers: Type.Optional(Type.Array(Marker)),
    semantics: Type.Optional(Semantics),
  },
  { additionalProperties: false, $id: 'https://charaforge.dev/schema/scene.v1.json', title: 'CharaForge Scene v1' },
);

export type ModelAsset = Static<typeof ModelAsset>;
export type MotionAsset = Static<typeof MotionAsset>;
export type ExpressionAsset = Static<typeof ExpressionAsset>;
export type AudioAsset = Static<typeof AudioAsset>;
export type ImageAsset = Static<typeof ImageAsset>;
export type ParamTrack = Static<typeof ParamTrack>;
export type PartOpacityTrack = Static<typeof PartOpacityTrack>;
export type MotionClip = Static<typeof MotionClip>;
export type ExpressionClip = Static<typeof ExpressionClip>;
export type Transform2D = Static<typeof Transform2D>;
export type CameraTransform = Static<typeof CameraTransform>;
export type Clip = Static<typeof Clip>;
export type Filter = Static<typeof Filter>;
export type BlendMode = Static<typeof BlendMode>;
export type ModelLayer = Static<typeof ModelLayer>;
export type ImageAttach = Static<typeof ImageAttach>;
export type ImageLayer = Static<typeof ImageLayer>;
export type Layer = Static<typeof Layer>;
export type AudioClip = Static<typeof AudioClip>;
export type Marker = Static<typeof Marker>;
export type Scene = Static<typeof Scene>;
