/*
 * 合成器用的 GLSL ES 3.0（必须显式写 #version 300 es，否则 Pixi v8 按 WebGL1 方式预处理）。
 * 约定：所有 RenderTexture 的 GL 第 0 行 = 图像顶部（Pixi 对非根渲染目标的朝向），
 * 因此 gl_FragCoord.xy 就是「y 向下的像素坐标 + 0.5」，与输入纹理的 texelFetch 坐标一一对应。
 * 所有颜色均为预乘 alpha。
 */

export const PASS_VERT = /* glsl */ `#version 300 es
precision highp float;
precision highp sampler2D;
in vec2 aPosition;
in vec2 aUV;
out vec2 vUV;
uniform mat3 uProjectionMatrix;
uniform mat3 uWorldTransformMatrix;
uniform mat3 uTransformMatrix;
void main() {
  mat3 m = uProjectionMatrix * uWorldTransformMatrix * uTransformMatrix;
  gl_Position = vec4((m * vec3(aPosition, 1.0)).xy, 0.0, 1.0);
  vUV = aUV;
}`;

/** 可分离高斯模糊的一个方向。范围外视为透明；权重按全部采样点归一化。 */
export const BLUR_FRAG = /* glsl */ `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uTexture;
uniform vec2 uDir;
uniform float uSigma;
in vec2 vUV;
out vec4 finalColor;
void main() {
  ivec2 size = textureSize(uTexture, 0);
  ivec2 p = ivec2(gl_FragCoord.xy);
  int r = int(ceil(3.0 * uSigma));
  float k = -0.5 / (uSigma * uSigma);
  vec4 acc = vec4(0.0);
  float wsum = 0.0;
  ivec2 d = ivec2(uDir);
  for (int i = -r; i <= r; i++) {
    float w = exp(float(i * i) * k);
    wsum += w;
    ivec2 q = p + d * i;
    if (q.x >= 0 && q.y >= 0 && q.x < size.x && q.y < size.y) acc += w * texelFetch(uTexture, q, 0);
  }
  finalColor = acc / wsum;
}`;

/** 方向模糊：沿 uDir 方向在 [−L/2, L/2] 上等距采样（双线性）取平均，范围外视为透明。 */
export const MOTION_BLUR_FRAG = /* glsl */ `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uTexture;
uniform vec2 uDir;
uniform float uLength;
in vec2 vUV;
out vec4 finalColor;
void main() {
  vec2 size = vec2(textureSize(uTexture, 0));
  int n = max(2, int(ceil(uLength)) + 1);
  vec4 acc = vec4(0.0);
  for (int i = 0; i < n; i++) {
    float t = -0.5 * uLength + uLength * float(i) / float(n - 1);
    vec2 q = gl_FragCoord.xy + uDir * t;
    if (q.x >= 0.0 && q.y >= 0.0 && q.x <= size.x && q.y <= size.y) acc += texture(uTexture, q / size);
  }
  finalColor = acc / float(n);
}`;

/**
 * 颜色调整（作用于非预乘颜色）：
 *   c = (c − 0.5)·contrast + 0.5 + brightness
 *   c = mix(luma709(c), c, saturation)
 *   c = c·multiply + offset
 */
export const COLOR_FRAG = /* glsl */ `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uTexture;
uniform float uBrightness;
uniform float uContrast;
uniform float uSaturation;
uniform vec3 uMultiply;
uniform vec3 uOffset;
in vec2 vUV;
out vec4 finalColor;
void main() {
  vec4 s = texelFetch(uTexture, ivec2(gl_FragCoord.xy), 0);
  if (s.a <= 0.0) { finalColor = vec4(0.0); return; }
  vec3 c = s.rgb / s.a;
  c = (c - 0.5) * uContrast + 0.5 + uBrightness;
  float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
  c = mix(vec3(l), c, uSaturation);
  c = c * uMultiply + uOffset;
  c = clamp(c, 0.0, 1.0);
  finalColor = vec4(c * s.a, s.a);
}`;

/**
 * 图层最终输出：圆角矩形羽化裁切 × 不透明度。混合由 Pixi 的 blendMode 完成。
 * 裁切覆盖率 = clamp(0.5 − d / max(feather, 1), 0, 1)，d 为到圆角矩形边界的有符号距离（像素，内部为负），
 * 即羽化带以边界为中心、宽 feather 像素；feather = 0 时为 1 像素抗锯齿边。
 */
export const OUTPUT_FRAG = /* glsl */ `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uTexture;
uniform float uOpacity;
uniform float uClipOn;
uniform vec4 uClipRect;   // x, y, w, h（合成像素）
uniform vec2 uClipShape;  // radius, feather
uniform vec2 uOrigin;     // 当前输出目标左上角在合成坐标中的位置
in vec2 vUV;
out vec4 finalColor;
float sdRoundRect(vec2 p, vec2 b, float r) {
  vec2 q = abs(p) - b + r;
  return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}
void main() {
  vec4 c = texelFetch(uTexture, ivec2(gl_FragCoord.xy), 0);
  float a = uOpacity;
  if (uClipOn > 0.5) {
    vec2 p = gl_FragCoord.xy + uOrigin;
    vec2 half_ = 0.5 * uClipRect.zw;
    float r = clamp(uClipShape.x, 0.0, min(half_.x, half_.y));
    float d = sdRoundRect(p - (uClipRect.xy + half_), half_, r);
    a *= clamp(0.5 - d / max(uClipShape.y, 1.0), 0.0, 1.0);
  }
  finalColor = c * a;
}`;

/**
 * 区域 + 缩放读回：输出像素 (ox, oy) 对应源区域 [rx + ox/s, rx + (ox+1)/s)，
 * 在其中均匀取 k×k 个双线性样本求平均（k = ceil(1/s)，上限 8）；s ≥ 1 时取 1 个样本。
 */
export const RESAMPLE_FRAG = /* glsl */ `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uTexture;
uniform vec4 uRegion;   // x, y, w, h（源像素）
uniform vec2 uOutSize;
uniform float uTaps;
in vec2 vUV;
out vec4 finalColor;
void main() {
  vec2 src = vec2(textureSize(uTexture, 0));
  vec2 o = floor(gl_FragCoord.xy);
  vec2 cell = uRegion.zw / uOutSize;
  int k = int(uTaps);
  vec4 acc = vec4(0.0);
  for (int j = 0; j < k; j++) for (int i = 0; i < k; i++) {
    vec2 q = uRegion.xy + (o + (vec2(float(i), float(j)) + 0.5) / float(k)) * cell;
    acc += texture(uTexture, q / src);
  }
  finalColor = acc / float(k * k);
}`;

/** 调试视图：A=渲染，B=参考视频帧（非预乘、不透明）。mode 0 = 叠加（mix），1 = 差异 |A−B|·gain。 */
export const DEBUG_FRAG = /* glsl */ `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uTexture;
uniform sampler2D uVideo;
uniform float uMode;
uniform float uMix;
uniform float uGain;
in vec2 vUV;
out vec4 finalColor;
void main() {
  vec4 a = texture(uTexture, vUV);
  vec3 b = texture(uVideo, vUV).rgb;
  vec3 ca = a.rgb + (1.0 - a.a) * vec3(0.0); // 渲染结果（背景已在合成时铺好）
  if (uMode < 0.5) finalColor = vec4(mix(ca, b, uMix), 1.0);
  else finalColor = vec4(clamp(abs(ca - b) * uGain, 0.0, 1.0), 1.0);
}`;

/** 逐像素拷贝。 */
export const COPY_FRAG = /* glsl */ `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uTexture;
in vec2 vUV;
out vec4 finalColor;
void main() {
  finalColor = texelFetch(uTexture, ivec2(gl_FragCoord.xy), 0);
}`;
