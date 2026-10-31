/*
 * 参考视频的逐帧精确定位。
 *
 * 帧 f 显示于 [f/fps, (f+1)/fps)。seek 到区间中点 (f + 0.5)/fps，可以避开容器时间戳取整
 * （mp4 time_base 1/15360、webm 1ms）带来的边界歧义；等 'seeked' 之后再等一次
 * requestVideoFrameCallback（若可用），确保新帧已经解码并成为当前帧。
 * 验证方法与结果见 docs/runtime-eval.md（scripts/verify-video.ts，与 ffmpeg 抽帧逐帧比对）。
 */

export function frameTime(frame: number, fps: number): number {
  return (frame + 0.5) / fps;
}

/** 由 currentTime 反推帧号（与 frameTime 互逆）。 */
export function timeToFrame(t: number, fps: number): number {
  return Math.floor(t * fps + 1e-6);
}

export async function seekVideoFrame(video: HTMLVideoElement, frame: number, fps: number): Promise<void> {
  const t = frameTime(frame, fps);
  if (Math.abs(video.currentTime - t) < 1e-4 && video.readyState >= 2) return;
  const presented = waitPresented(video);
  await new Promise<void>((resolve, reject) => {
    const ok = () => { cleanup(); resolve(); };
    const bad = () => { cleanup(); reject(new Error('视频 seek 失败')); };
    const cleanup = () => { video.removeEventListener('seeked', ok); video.removeEventListener('error', bad); };
    video.addEventListener('seeked', ok);
    video.addEventListener('error', bad);
    video.currentTime = t;
  });
  await presented;
}

function waitPresented(video: HTMLVideoElement): Promise<void> {
  const v = video as HTMLVideoElement & { requestVideoFrameCallback?: (cb: () => void) => number };
  if (!v.requestVideoFrameCallback) return Promise.resolve();
  // 超时保护：某些情况下（目标帧与当前帧相同）不会触发回调
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, 250);
    v.requestVideoFrameCallback!(() => { clearTimeout(timer); resolve(); });
  });
}

export function loadVideo(url: string): Promise<HTMLVideoElement> {
  const v = document.createElement('video');
  v.muted = true;
  v.preload = 'auto';
  v.playsInline = true;
  v.crossOrigin = 'anonymous';
  v.src = url;
  return new Promise((resolve, reject) => {
    v.addEventListener('loadeddata', () => resolve(v), { once: true });
    v.addEventListener('error', () => reject(new Error(`参考视频加载失败：${url}`)), { once: true });
  });
}
