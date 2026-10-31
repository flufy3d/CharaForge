/**
 * 二维仿射变换 [a, b, c, d, e, f]：x' = a·x + c·y + e，y' = b·x + d·y + f。
 * 所有坐标系都是 y 向下（见 scene.ts 规范第 2 条）。
 */
export type Affine = readonly [number, number, number, number, number, number];

export const IDENTITY: Affine = [1, 0, 0, 1, 0, 0];

/** m1 ∘ m2：先作用 m2，再作用 m1。 */
export function mul(m1: Affine, m2: Affine): Affine {
  const [a1, b1, c1, d1, e1, f1] = m1;
  const [a2, b2, c2, d2, e2, f2] = m2;
  return [
    a1 * a2 + c1 * b2,
    b1 * a2 + d1 * b2,
    a1 * c2 + c1 * d2,
    b1 * c2 + d1 * d2,
    a1 * e2 + c1 * f2 + e1,
    b1 * e2 + d1 * f2 + f1,
  ];
}

export function translate(x: number, y: number): Affine {
  return [1, 0, 0, 1, x, y];
}

export function scale(sx: number, sy: number): Affine {
  return [sx, 0, 0, sy, 0, 0];
}

/** 旋转（度）。y 向下的坐标系中正角度为顺时针。 */
export function rotate(deg: number): Affine {
  if (deg === 0) return IDENTITY;
  const r = (deg * Math.PI) / 180;
  const c = Math.cos(r);
  const s = Math.sin(r);
  return [c, s, -s, c, 0, 0];
}

export function apply(m: Affine, x: number, y: number): [number, number] {
  return [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]];
}

/**
 * 图层变换：world = [x,y] + R(rot)·diag(sx, sy)·(p − anchor)，
 * sx = scale·scaleX·(flipX ? −1 : 1)，sy = scale·scaleY（scaleX、scaleY 缺省 1）。
 * scaleX = scaleY = 1 时与旧公式逐位相同（乘 1、乘 −1 都是精确运算）。
 */
export function layerMatrix(t: {
  x: number;
  y: number;
  scale: number;
  scaleX?: number;
  scaleY?: number;
  rotation: number;
  anchorX: number;
  anchorY: number;
  flipX: boolean;
}): Affine {
  const sx = t.scale * (t.scaleX ?? 1) * (t.flipX ? -1 : 1);
  const sy = t.scale * (t.scaleY ?? 1);
  return mul(translate(t.x, t.y), mul(rotate(t.rotation), mul(scale(sx, sy), translate(-t.anchorX, -t.anchorY))));
}

/** 相机：screen = [W/2, H/2] + zoom·R(rot)·(world − [x, y])。 */
export function cameraMatrix(c: { x: number; y: number; zoom: number; rotation: number }, W: number, H: number): Affine {
  return mul(translate(W / 2, H / 2), mul(rotate(c.rotation), mul(scale(c.zoom, c.zoom), translate(-c.x, -c.y))));
}
