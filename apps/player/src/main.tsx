/*
 * CharaForge 播放器 UI。
 *
 * URL 参数：
 *   scene=<仓库相对路径>   缺省 scenes/deepseek-chan/scene.json，不存在时回落 packages/scene/examples/minimal.scene.json；
 *                          demo:N 为内置的 N 实例演示场景
 *   video=<仓库相对路径>   参考视频，缺省 assets/video/DeepSeek-chan.mp4
 *   debug=1                打开调试面板
 *   compare=side|overlay|diff  初始对照模式（需 debug=1 才显示面板）
 *   frame=<n>              初始帧
 *   workers=<n>            求值 Worker 数（缺省 clamp(逻辑核数/2 − 1, 2, 6)；0 = 主线程同步求值）
 * 快捷键：空格 播放/暂停；←/→ ±1 帧；Shift+←/→ ±10 帧；Home/End；D 调试面板；[ / ] 上/下一个标记
 */
import type { Scene } from '@charaforge/scene';
import { StrictMode, useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { demoScene } from './demo.ts';
import { PlayerEngine, type CompareMode, type EngineStatus } from './engine.ts';
import { timecode } from './util.ts';

const params = new URLSearchParams(location.search);
const DEFAULT_SCENE = 'scenes/deepseek-chan/scene.json';
const FALLBACK_SCENE = 'packages/scene/examples/minimal.scene.json';

async function resolveScene(): Promise<{ label: string; scene: string | object }> {
  const p = params.get('scene');
  if (p?.startsWith('demo:')) return { label: p, scene: demoScene(Math.max(1, Number(p.slice(5)) || 4)) };
  if (p) return { label: p, scene: `/${p.replace(/^\/+/, '')}` };
  const head = await fetch(`/${DEFAULT_SCENE}`, { method: 'HEAD' }).catch(() => null);
  if (head?.ok) return { label: DEFAULT_SCENE, scene: `/${DEFAULT_SCENE}` };
  return { label: `${FALLBACK_SCENE}（${DEFAULT_SCENE} 不存在，已回落）`, scene: `/${FALLBACK_SCENE}` };
}

function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoBoxRef = useRef<HTMLDivElement>(null);
  const engineRef = useRef<PlayerEngine | null>(null);
  const [scene, setScene] = useState<Scene | null>(null);
  const [label, setLabel] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [st, setSt] = useState<EngineStatus | null>(null);
  const [debug, setDebug] = useState(params.get('debug') === '1');
  const [compare, setCompare] = useState<CompareMode>('off');
  const initialCompare = (params.get('compare') as CompareMode | null) ?? null;
  const [mix, setMix] = useState(0.5);
  const [gain, setGain] = useState(4);
  const [videoOk, setVideoOk] = useState<boolean | null>(null);
  const [gotoText, setGotoText] = useState('');

  useEffect(() => {
    let cancelled = false;
    const eng = new PlayerEngine();
    engineRef.current = eng;
    let last = 0;
    eng.onStatus = (s) => {
      const now = performance.now();
      // 播放时状态栏 4Hz 刷新即可（React 重渲染会占用主线程）
      if (now - last > 250 || !s.playing) {
        last = now;
        setSt({ ...s });
      }
    };
    (async () => {
      // 缺省 Worker 数：逻辑核数的一半减一，限制在 [2, 6]（12 核 → 5，实测 3 实例 60fps 所需）
      const nWorkers = params.has('workers') ? Number(params.get('workers')) : Math.max(2, Math.min(6, Math.floor((navigator.hardwareConcurrency || 4) / 2) - 1));
      await eng.init(canvasRef.current!, nWorkers);
      (window as unknown as { __player: PlayerEngine }).__player = eng;
      const { label, scene } = await resolveScene();
      if (cancelled) return;
      setLabel(label);
      setScene(await eng.load(scene));
      if (params.has('frame')) eng.seek(Number(params.get('frame')));
      eng.start();
      const vid = params.get('video') ?? 'assets/video/DeepSeek-chan.mp4';
      eng.loadVideo(vid).then(() => {
        setVideoOk(true);
        if (videoBoxRef.current && eng.video) {
          eng.video.className = 'ref-video';
          videoBoxRef.current.appendChild(eng.video);
        }
        if (initialCompare && ['side', 'overlay', 'diff'].includes(initialCompare)) {
          setCompare(initialCompare);
          eng.setCompare(initialCompare);
        }
      }, () => setVideoOk(false));
    })().catch((e) => setError(String(e?.stack ?? e)));
    return () => {
      cancelled = true;
      eng.stop();
    };
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const eng = engineRef.current;
      if (!eng?.scene || (e.target as HTMLElement).tagName === 'INPUT') return;
      const n = e.shiftKey ? 10 : 1;
      if (e.key === ' ') { e.preventDefault(); void eng.togglePlay(); }
      else if (e.key === 'ArrowRight') { e.preventDefault(); eng.step(n); }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); eng.step(-n); }
      else if (e.key === 'Home') eng.seek(0);
      else if (e.key === 'End') eng.seek(eng.duration - 1);
      else if (e.key === 'd' || e.key === 'D') setDebug((d) => !d);
      else if (e.key === '[' || e.key === ']') jumpMarker(e.key === ']' ? 1 : -1);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  const eng = engineRef.current;
  const fps = scene?.timebase.fps ?? 60;
  const frame = st?.frame ?? 0;
  const marker = eng?.scene ? eng.currentMarker() : undefined;
  const markers = scene?.markers ?? [];

  function jumpMarker(dir: 1 | -1) {
    if (!eng) return;
    const starts = markers.map((m) => m.start).sort((a, b) => a - b);
    const next = dir > 0 ? starts.find((s) => s > eng.frame) : [...starts].reverse().find((s) => s < eng.frame);
    if (next !== undefined) eng.seek(next);
  }

  function applyCompare(m: CompareMode) {
    setCompare(m);
    eng?.setCompare(m);
  }

  const credits = scene ? [...(scene.meta.credits ?? []), ...scene.assets.models.flatMap((m) => (m.credits ? [m.credits] : []))] : [];

  return (
    <div className="app">
      <header>
        <strong>CharaForge 播放器</strong>
        <span className="muted">{scene?.meta.title}</span>
        <span className="muted small">{label}</span>
      </header>
      {error && <pre className="error">{error}</pre>}
      <div className={`stage compare-${debug ? compare : 'off'}`}>
        <div className="canvas-box">
          <canvas ref={canvasRef} />
        </div>
        <div className="video-box" ref={videoBoxRef} style={{ display: debug && compare === 'side' ? undefined : 'none' }} />
      </div>
      <div className="transport">
        <button onClick={() => eng && void eng.togglePlay()}>{st?.playing ? '暂停' : '播放'}</button>
        <button onClick={() => eng?.step(-10)}>−10</button>
        <button onClick={() => eng?.step(-1)}>−1</button>
        <button onClick={() => eng?.step(1)}>+1</button>
        <button onClick={() => eng?.step(10)}>+10</button>
        <input
          className="scrub"
          type="range"
          min={0}
          max={Math.max(0, (scene?.duration ?? 1) - 1)}
          value={frame}
          onChange={(e) => eng?.seek(Number(e.target.value))}
        />
        <span className="mono">
          帧 {frame} / {(scene?.duration ?? 1) - 1}　{timecode(frame, fps)}
        </span>
        <span className="mono">实测 {st ? st.fps.toFixed(1) : '—'} fps · 每帧渲染 {st ? st.renderMs.toFixed(1) : '—'} ms · workers {eng?.workers ?? '—'}</span>
      </div>
      <div className="info">
        <span>标记：{marker ? `${marker.label ?? marker.id}（${marker.kind ?? ''} ${marker.start}–${marker.end ?? ''}）` : '—'}</span>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            const m = markers.find((m) => m.id === gotoText || m.label === gotoText);
            eng?.seek(m ? m.start : Number(gotoText));
          }}
        >
          <input value={gotoText} onChange={(e) => setGotoText(e.target.value)} placeholder="跳到帧号或标记" />
          <button type="submit">跳转</button>
        </form>
        <select value="" onChange={(e) => e.target.value && eng?.seek(Number(e.target.value))}>
          <option value="">标记列表（{markers.length}）</option>
          {markers.map((m) => (
            <option key={m.id} value={m.start}>
              {m.start} · {m.label ?? m.id}
            </option>
          ))}
        </select>
        <button onClick={() => setDebug((d) => !d)}>{debug ? '关闭调试' : '调试模式'}</button>
      </div>
      {debug && (
        <div className="debug">
          <span>原视频对照：</span>
          {(['off', 'side', 'overlay', 'diff'] as CompareMode[]).map((m) => (
            <label key={m}>
              <input type="radio" checked={compare === m} onChange={() => applyCompare(m)} disabled={m !== 'off' && !videoOk} />
              {{ off: '关', side: '并排', overlay: '叠加', diff: '差异图' }[m]}
            </label>
          ))}
          {compare === 'overlay' && (
            <label>
              视频透明度 {mix.toFixed(2)}
              <input type="range" min={0} max={1} step={0.01} value={mix} onChange={(e) => { setMix(Number(e.target.value)); eng?.setOverlay(Number(e.target.value), gain); }} />
            </label>
          )}
          {compare === 'diff' && (
            <label>
              放大 ×{gain}
              <input type="range" min={1} max={32} step={1} value={gain} onChange={(e) => { setGain(Number(e.target.value)); eng?.setOverlay(mix, Number(e.target.value)); }} />
            </label>
          )}
          <span className="mono">
            视频帧 {st?.videoFrame ?? '—'} {compare !== 'off' ? (st?.videoExact ? '（已精确对齐）' : st?.playing ? '（播放中，近似对齐）' : '（定位中…）') : ''}
            {videoOk === false && ' 参考视频加载失败'}
          </span>
          <span className="mono">本帧推进 {st?.steps ?? 0} 步</span>
        </div>
      )}
      <footer>
        {credits.map((c) => (
          <div key={c}>{c}</div>
        ))}
        <div className="muted small">渲染：Live2D Cubism SDK for Web（Core / Framework），合成：PixiJS v8</div>
      </footer>
    </div>
  );
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
