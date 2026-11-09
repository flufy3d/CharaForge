/*
 * 离线逐帧导出：把运行时已经渲染好的帧 + 场景音频编码成 MP4 / WebM。
 *
 * 画面只是帧号的函数（运行时保证），所以导出就是按顺序调用 renderFrame、把 WebGL 画布拷到
 * 一张 2D 画布、再交给 mediabunny 编码。不使用墙钟时间，不需要用户手势。
 *
 * 音频按 clock.ts 的片段语义（start 帧、offset 秒、duration 秒、gain）在
 * OfflineAudioContext 上重新混音；导出的音频时间轴从 0 开始，只包含 [from, to) 这段。
 */
import { asciiUrl } from '@charaforge/runtime';
import type { Scene } from '@charaforge/scene';
import {
  AudioBufferSource,
  BufferTarget,
  CanvasSource,
  getFirstEncodableAudioCodec,
  getFirstEncodableVideoCodec,
  Mp4OutputFormat,
  Output,
  Quality,
  WebMOutputFormat,
  type AudioCodec,
  type OutputFormat,
  type VideoCodec,
} from 'mediabunny';

export interface ExportVideoOptions {
  scene: Scene;
  /** 运行时的 WebGL 画布（实际合成分辨率）。 */
  canvas: HTMLCanvasElement;
  /** 渲染并上屏第 frame 帧（调用方负责 renderFrameAsync + present）。 */
  renderFrame: (frame: number) => Promise<void> | void;
  /** runtime.resolveAsset */
  resolveAsset: (uri: string) => string;
  /** [from, to)，缺省 0..scene.duration */
  from?: number;
  to?: number;
  /** 缺省 scene.timebase.fps */
  fps?: number;
  /** 缺省场景合成分辨率 */
  width?: number;
  height?: number;
  /** 缺省 scene.composition.background ?? '#000' */
  background?: string;
  /** 缺省 auto：MP4/AVC/AAC，不可用时回退 WebM/VP9/Opus */
  format?: 'mp4' | 'webm' | 'auto';
  quality?: 'high' | 'medium' | 'low';
  onProgress?: (p: { phase: 'audio' | 'video' | 'finalizing'; frame: number; total: number }) => void;
  signal?: AbortSignal;
}

export interface ExportVideoResult {
  data: Uint8Array;
  mimeType: string;
  extension: 'mp4' | 'webm';
  width: number;
  height: number;
  fps: number;
  frames: number;
  duration: number;
  hasAudio: boolean;
  codecs: { video: string; audio?: string };
}

const MP4_VIDEO_CODECS: VideoCodec[] = ['avc', 'hevc', 'av1', 'vp9'];
const MP4_AUDIO_CODECS: AudioCodec[] = ['aac', 'opus', 'mp3'];
const WEBM_VIDEO_CODECS: VideoCodec[] = ['vp9', 'vp8', 'av1'];
const WEBM_AUDIO_CODECS: AudioCodec[] = ['opus', 'vorbis'];

const AUDIO_SAMPLE_RATE = 48000;
const AUDIO_CHANNELS = 2;

interface FormatAttempt {
  extension: 'mp4' | 'webm';
  format: OutputFormat;
  videoCodecs: VideoCodec[];
  audioCodecs: AudioCodec[];
}

interface PickedFormat extends FormatAttempt {
  videoCodec: VideoCodec;
  audioCodec: AudioCodec | null;
}

/** 按浏览器实际可编码性挑选容器与编解码器；auto/mp4 首选 MP4，失败回退 WebM。 */
function formatAttempts(format: 'mp4' | 'webm' | 'auto'): FormatAttempt[] {
  const mp4 = (): FormatAttempt => ({
    extension: 'mp4',
    format: new Mp4OutputFormat({ fastStart: 'in-memory' }),
    videoCodecs: MP4_VIDEO_CODECS,
    audioCodecs: MP4_AUDIO_CODECS,
  });
  const webm = (): FormatAttempt => ({
    extension: 'webm',
    format: new WebMOutputFormat(),
    videoCodecs: WEBM_VIDEO_CODECS,
    audioCodecs: WEBM_AUDIO_CODECS,
  });
  return format === 'webm' ? [webm()] : [mp4(), webm()];
}

function supportedCodecs<T extends string>(wanted: readonly T[], supported: readonly string[]): T[] {
  return wanted.filter((c) => supported.includes(c));
}

async function pickFormat(
  format: 'mp4' | 'webm' | 'auto',
  video: { width: number; height: number; fps: number },
  audio: AudioBuffer | null,
  quality: Quality,
  signal?: AbortSignal,
): Promise<PickedFormat> {
  const errors: string[] = [];
  for (const attempt of formatAttempts(format)) {
    throwIfAborted(signal);
    const supportedVideo = attempt.format.getSupportedVideoCodecs();
    const candidates = supportedCodecs(attempt.videoCodecs, supportedVideo);
    if (!candidates.length) {
      errors.push(`${attempt.extension}: 容器不支持所选视频编码（支持：${supportedVideo.join(', ') || '无'}）`);
      continue;
    }
    let videoCodec: VideoCodec | null;
    try {
      videoCodec = await getFirstEncodableVideoCodec(candidates, {
        width: video.width,
        height: video.height,
        frameRate: video.fps,
        quality,
      });
    } catch (e) {
      errors.push(`${attempt.extension} 视频编码检测失败：${errorMessage(e)}`);
      continue;
    }
    if (!videoCodec) {
      errors.push(`${attempt.extension}: 没有可编码的视频编码器（尝试：${candidates.join(', ')}）`);
      continue;
    }

    let audioCodec: AudioCodec | null = null;
    if (audio) {
      const audioCandidates = supportedCodecs(attempt.audioCodecs, attempt.format.getSupportedAudioCodecs());
      if (!audioCandidates.length) {
        errors.push(`${attempt.extension}: 容器不支持所选音频编码`);
        continue;
      }
      try {
        audioCodec = await getFirstEncodableAudioCodec(audioCandidates, {
          numberOfChannels: audio.numberOfChannels,
          sampleRate: audio.sampleRate,
          quality,
        });
      } catch (e) {
        errors.push(`${attempt.extension} 音频编码检测失败：${errorMessage(e)}`);
        continue;
      }
      if (!audioCodec) {
        errors.push(`${attempt.extension}: 没有可编码的音频编码器（尝试：${audioCandidates.join(', ')}）`);
        continue;
      }
    }
    return { ...attempt, videoCodec, audioCodec };
  }
  throw new Error(
    `没有可用的视频/音频编码器，无法导出（${format}）。浏览器需要 WebCodecs 的 VideoEncoder/AudioEncoder 支持。\n` +
      errors.map((e) => `  - ${e}`).join('\n'),
  );
}

function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

function throwIfAborted(signal?: AbortSignal): void {
  if (!signal?.aborted) return;
  if (typeof DOMException !== 'undefined') throw new DOMException('Aborted', 'AbortError');
  const e = new Error('Aborted');
  e.name = 'AbortError';
  throw e;
}

function intOption(value: number | undefined, fallback: number): number {
  return Number.isFinite(value) ? Math.floor(value!) : fallback;
}

/**
 * 把 [from, to) 范围内的音频片段混成一条 48 kHz 立体声缓冲。
 * 没有片段、没有可用缓冲时返回 null（导出就不加音轨）。
 */
export async function renderAudioMix(
  scene: Scene,
  resolveAsset: (uri: string) => string,
  from: number,
  to: number,
  fps: number,
  signal?: AbortSignal,
): Promise<AudioBuffer | null> {
  const clips = scene.audio ?? [];
  if (!clips.length) return null;
  const assets = scene.assets.audio ?? [];
  if (!assets.length) return null;

  const frameCount = to - from;
  if (!(frameCount > 0) || !(fps > 0)) return null;
  const length = Math.ceil((frameCount / fps) * AUDIO_SAMPLE_RATE);
  if (length <= 0) return null;

  throwIfAborted(signal);

  const ctx = new OfflineAudioContext(AUDIO_CHANNELS, length, AUDIO_SAMPLE_RATE);
  const used = new Set(clips.map((c) => c.audio));
  const buffers = new Map<string, AudioBuffer>();
  await Promise.all(
    assets
      .filter((a) => used.has(a.id))
      .map(async (a) => {
        const res = await fetch(asciiUrl(resolveAsset(a.uri)), { signal });
        if (!res.ok) throw new Error(`音频加载失败：${a.uri}（HTTP ${res.status}）`);
        buffers.set(a.id, await ctx.decodeAudioData(await res.arrayBuffer()));
        throwIfAborted(signal);
      }),
  );
  throwIfAborted(signal);

  const fromSec = from / fps;
  const toSec = to / fps;
  const span = toSec - fromSec;
  let scheduled = 0;
  for (const clip of clips) {
    const buf = buffers.get(clip.audio);
    if (!buf) continue;
    const clipStart = clip.start / fps;
    if (clipStart >= toSec) continue;
    const offset0 = clip.offset ?? 0;
    if (!(offset0 < buf.duration)) continue;
    const dur0 = Math.min(clip.duration ?? buf.duration - offset0, buf.duration - offset0);
    if (!(dur0 > 0)) continue;
    // 片段在导出起点之前就开始了：从中间接上（与 clock.ts 的 into/offset 逻辑一致）
    const into = Math.max(0, fromSec - clipStart);
    if (into >= dur0) continue;
    const when = Math.max(0, clipStart - fromSec);
    const offset = offset0 + into;
    // 不要超出导出范围
    const dur = Math.min(dur0 - into, span - when);
    if (!(dur > 0)) continue;

    const src = ctx.createBufferSource();
    src.buffer = buf;
    const gain = ctx.createGain();
    gain.gain.value = clip.gain ?? 1;
    src.connect(gain).connect(ctx.destination);
    src.start(when, offset, dur);
    scheduled++;
  }
  if (!scheduled) return null;

  throwIfAborted(signal);
  const mix = await ctx.startRendering();
  throwIfAborted(signal);
  return mix;
}

export async function exportSceneVideo(opts: ExportVideoOptions): Promise<ExportVideoResult> {
  const { scene, canvas, renderFrame, resolveAsset, signal } = opts;
  throwIfAborted(signal);

  const fps = opts.fps ?? scene.timebase.fps;
  if (!(fps > 0)) throw new Error(`fps 无效：${fps}`);

  const duration = scene.duration;
  const from = Math.max(0, Math.min(duration, intOption(opts.from, 0)));
  const to = Math.max(0, Math.min(duration, intOption(opts.to, duration)));
  if (to <= from) throw new Error(`导出范围无效：[${from}, ${to})（场景共 ${duration} 帧）`);

  const width = Math.max(1, intOption(opts.width, scene.composition.width));
  const height = Math.max(1, intOption(opts.height, scene.composition.height));
  const background = opts.background ?? scene.composition.background ?? '#000000';
  const frames = to - from;
  const quality = new Quality(opts.quality ?? 'high');

  if (canvas.width <= 0 || canvas.height <= 0) throw new Error('WebGL 画布尺寸为 0，无法导出');

  opts.onProgress?.({ phase: 'audio', frame: 0, total: frames });
  const mix = await renderAudioMix(scene, resolveAsset, from, to, fps, signal);
  opts.onProgress?.({ phase: 'audio', frame: frames, total: frames });

  const picked = await pickFormat(opts.format ?? 'auto', { width, height, fps }, mix, quality, signal);

  const out = document.createElement('canvas');
  out.width = width;
  out.height = height;
  const ctx = out.getContext('2d', { alpha: true });
  if (!ctx) {
    out.width = out.height = 0;
    throw new Error('无法创建 2D 画布上下文');
  }
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';

  const target = new BufferTarget();
  let output: Output<OutputFormat, BufferTarget> | null = null;
  let videoSource: CanvasSource | null = null;
  let audioSource: AudioBufferSource | null = null;
  let finalized = false;
  try {
    output = new Output({ format: picked.format, target });
    videoSource = new CanvasSource(out, { codec: picked.videoCodec, quality });
    output.addVideoTrack(videoSource, { frameRate: fps });
    if (mix && picked.audioCodec) {
      audioSource = new AudioBufferSource({ codec: picked.audioCodec, quality });
      output.addAudioTrack(audioSource);
    }
    await output.start();

    // 音频时间轴从 0 开始：只混了导出范围
    if (audioSource && mix) {
      await audioSource.add(mix);
      audioSource.close();
      audioSource = null;
    }

    // 源画布与导出尺寸不一致时按 contain 缩放（留黑边），否则原样 1:1 拷贝
    const sw = canvas.width;
    const sh = canvas.height;
    const sameAspect = Math.abs(sw / sh - width / height) < 1e-6;
    const scale = sameAspect ? 1 : Math.min(width / sw, height / sh);
    const dw = sw * scale;
    const dh = sh * scale;
    const dx = (width - dw) / 2;
    const dy = (height - dh) / 2;

    for (let f = from; f < to; f++) {
      throwIfAborted(signal);
      // 顺序很重要：先让运行时渲染并上屏，再把 WebGL 画布拷到 2D 画布
      await renderFrame(f);
      throwIfAborted(signal);
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = background;
      ctx.fillRect(0, 0, width, height);
      ctx.drawImage(canvas, dx, dy, dw, dh);
      await videoSource.add((f - from) / fps, 1 / fps);
      opts.onProgress?.({ phase: 'video', frame: f - from + 1, total: frames });
    }

    videoSource.close();
    videoSource = null;
    opts.onProgress?.({ phase: 'finalizing', frame: frames, total: frames });
    await output.finalize();
    finalized = true;

    const buffer = target.buffer;
    if (!buffer) throw new Error('导出失败：输出缓冲为空');
    const data = new Uint8Array(buffer.slice(0));
    return {
      data,
      mimeType: picked.format.mimeType,
      extension: picked.extension,
      width,
      height,
      fps,
      frames,
      duration: frames / fps,
      hasAudio: !!mix,
      codecs: { video: picked.videoCodec, ...(mix && picked.audioCodec ? { audio: picked.audioCodec } : {}) },
    };
  } catch (e) {
    closeSource(videoSource, output);
    closeSource(audioSource, output);
    videoSource = null;
    audioSource = null;
    if (output && !finalized && output.state !== 'canceled' && output.state !== 'finalized') {
      await output.cancel().catch(() => {});
    }
    throw e;
  } finally {
    closeSource(videoSource, output);
    closeSource(audioSource, output);
    out.width = 0;
    out.height = 0;
  }
}

/** error 路径上安全关闭 source：output 还没 start（pending）或已取消时 close 会抛错。 */
function closeSource(source: CanvasSource | AudioBufferSource | null, output: Output<OutputFormat, BufferTarget> | null): void {
  if (!source || !output) return;
  if (output.state !== 'started' && output.state !== 'finalizing') return;
  try {
    source.close();
  } catch {
    /* 已经关闭 / 编码器已停止 */
  }
}
