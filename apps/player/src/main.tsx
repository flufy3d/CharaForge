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
 *   prewarm=0              不做新图层预热（缺省加载后与播放中让 Worker 提前做物理预热，见 PlayerEngine.startPrewarm；诊断用）
 * 快捷键：空格 播放/暂停；←/→ ±1 帧；Shift+←/→ ±10 帧；Home/End；[ / ] 上/下一个镜头标记；
 *         D 调试面板；调试面板打开时 1～4 = 对照模式 关/并排/叠加/差异图；M 静音
 */
import { asciiUrl } from '@charaforge/runtime';
import type { Scene } from '@charaforge/scene';
import { StrictMode, useEffect, useRef, useState, type ReactNode } from 'react';
import { createRoot } from 'react-dom/client';
import { demoScene } from './demo.ts';
import { PlayerEngine, type CompareMode, type EngineStatus } from './engine.ts';
import { pageUrl, timecode } from './util.ts';

const params = new URLSearchParams(location.search);
const DEFAULT_SCENE = 'scenes/deepseek-chan/scene.json';
const FALLBACK_SCENE = 'packages/scene/examples/minimal.scene.json';
const DEFAULT_VIDEO = 'assets/video/DeepSeek-chan.mp4';
const COMPARE_LABEL: Record<CompareMode, string> = { off: '关', side: '并排', overlay: '叠加', diff: '差异图' };
const COMPARE_MODES: CompareMode[] = ['off', 'side', 'overlay', 'diff'];
/** 模型原作者的发布页（模型禁止再分发，请从这里获取） */
const MODEL_PAGES: { match: RegExp; name: string; url: string }[] = [
  {
    match: /models\/deepseek-chan\//,
    name: 'DeepSeek-chan 免费 Live2D 模型（VGen）',
    url: 'https://vgen.co/KAMIYA/product/deepseek-chan-deepseek-free-live2d-model-layered-psd-psd/1c55a71a-2747-4616-819d-2dc0a934089d',
  },
];

async function resolveScene(): Promise<{ label: string; scene: string | object }> {
  const p = params.get('scene');
  if (p?.startsWith('demo:')) return { label: p, scene: demoScene(Math.max(1, Number(p.slice(5)) || 4)) };
  // 场景路径是仓库相对路径，相对页面地址解析（页面可能在反向代理的路径前缀下，不能用 / 开头）
  if (p) return { label: p, scene: pageUrl(p) };
  const head = await fetch(asciiUrl(pageUrl(DEFAULT_SCENE)), { method: 'HEAD' }).catch(() => null);
  if (head?.ok) return { label: DEFAULT_SCENE, scene: pageUrl(DEFAULT_SCENE) };
  return { label: `${FALLBACK_SCENE}（${DEFAULT_SCENE} 不存在，已回落）`, scene: pageUrl(FALLBACK_SCENE) };
}

/* ---------- 启动（模块级单例） ----------
 * 引擎是命令式的（WebGL 上下文、Worker 池、AudioContext），只能建一次。React StrictMode 在开发模式下会把 effect
 * 跑两遍：以前每遍各建一个引擎，两个运行时共用同一个 canvas，`window.__player` 指向哪个取决于谁先初始化完，
 * 测试脚本偶尔会等到那个被丢弃的引擎（永远不加载场景）。现在启动只发生一次，effect 只负责订阅。 */
interface BootState {
  stage: string;
  t0: number;
  error: { title: string; hint?: string; detail: string } | null;
  scene: Scene | null;
  label: string;
  videoOk: boolean | null;
  notices: string[];
}
let bootState: BootState = { stage: '初始化', t0: performance.now(), error: null, scene: null, label: '', videoOk: null, notices: [] };
const bootListeners = new Set<(s: BootState) => void>();
const setBoot = (patch: Partial<BootState>) => {
  bootState = { ...bootState, ...patch };
  for (const l of bootListeners) l(bootState);
};
let engine: PlayerEngine | null = null;

function explain(e: unknown): BootState['error'] {
  const detail = String((e as Error)?.stack ?? e);
  const msg = String((e as Error)?.message ?? e);
  if (/场景加载失败 404/.test(msg))
    return { title: '场景文件不存在', hint: '?scene= 是仓库相对路径，只能访问 scenes/、packages/scene/examples/ 下的文件。', detail };
  if (/加载失败 404：.*\/assets\//.test(msg) || /moc3|model3/.test(msg))
    return { title: '模型文件缺失或损坏', hint: '模型不在仓库里，请先运行 pnpm run setup（或 pnpm setup:assets）下载并校验。', detail };
  if (/WebGL2/.test(msg)) return { title: '浏览器不支持 WebGL2', hint: '请换用新版 Chrome / Edge，并确认没有禁用硬件加速。', detail };
  if (/^\[/.test(msg) || /schema|校验|must|expected/i.test(msg)) return { title: '场景格式校验失败', hint: '场景需符合 charaforge.scene v1（packages/scene/schema）。', detail };
  return { title: '播放器启动失败', detail };
}

function boot(canvas: HTMLCanvasElement): void {
  if (engine) return;
  const eng = new PlayerEngine();
  engine = eng;
  (window as unknown as { __player: PlayerEngine }).__player = eng;
  (async () => {
    if (typeof (globalThis as { Live2DCubismCore?: unknown }).Live2DCubismCore === 'undefined') {
      setBoot({ error: { title: '没有找到 Live2D Cubism Core', hint: 'Cubism Core 不进仓库，请先运行 pnpm run setup（或 pnpm setup:cubism），它会从 Live2D 官方地址下载到 vendor/cubism/core/。', detail: `${pageUrl('vendor/cubism/core/live2dcubismcore.min.js')} 未加载` } });
      return;
    }
    // 缺省 Worker 数：逻辑核数的一半减一，限制在 [2, 6]（12 核 → 5，实测 3 实例 60fps 所需）
    const nWorkers = params.has('workers') ? Number(params.get('workers')) : Math.max(2, Math.min(6, Math.floor((navigator.hardwareConcurrency || 4) / 2) - 1));
    setBoot({ stage: `初始化 WebGL 与 Cubism Core（${nWorkers} 个 Worker）` });
    await eng.init(canvas, nWorkers);
    setBoot({ stage: '读取场景' });
    const { label, scene } = await resolveScene();
    setBoot({ label });
    const sc = await eng.load(scene, (stage) => setBoot({ stage }));
    if (eng.audioError) setBoot({ notices: [...bootState.notices, `音频加载失败，将无声播放（时钟退化为系统时钟）：${eng.audioError}`] });
    if (params.has('frame')) eng.seek(Number(params.get('frame')));
    setBoot({ stage: '渲染第一帧' });
    eng.start();
    // 让 Worker 提前为各图层做物理预热（见 PlayerEngine.startPrewarm），避免播放到新图层时卡顿；?prewarm=0 关闭（诊断用）
    if (params.get('prewarm') !== '0') eng.startPrewarm();
    setBoot({ scene: sc });
    const vid = params.get('video') ?? DEFAULT_VIDEO;
    eng.loadVideo(vid).then(
      () => setBoot({ videoOk: true }),
      () => setBoot({ videoOk: false, notices: [...bootState.notices, `参考视频加载失败（${vid}），调试模式的原视频对照不可用。`] }),
    );
  })().catch((e) => setBoot({ error: explain(e) }));
}

function linkify(text: string): ReactNode[] {
  return text.split(/(https?:\/\/[^\s,，)）]+)/g).map((part, i) =>
    /^https?:\/\//.test(part) ? (
      <a key={i} href={part} target="_blank" rel="noreferrer">
        {part}
      </a>
    ) : (
      part
    ),
  );
}

function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoBoxRef = useRef<HTMLDivElement>(null);
  const [bs, setBs] = useState<BootState>(bootState);
  const [st, setSt] = useState<EngineStatus | null>(null);
  const [debug, setDebug] = useState(params.get('debug') === '1');
  const [compare, setCompare] = useState<CompareMode>('off');
  const [mix, setMix] = useState(0.5);
  const [gain, setGain] = useState(4);
  const [gotoText, setGotoText] = useState('');
  const [copied, setCopied] = useState(false);
  const [now, setNow] = useState(performance.now());
  const scene = bs.scene;
  const eng = engine;
  // 第一帧上屏之前都算加载中（第一帧要预热物理，可能要一两秒）
  const loading = !bs.error && !st?.ready;

  useEffect(() => {
    bootListeners.add(setBs);
    boot(canvasRef.current!);
    setBs(bootState);
    return () => void bootListeners.delete(setBs);
  }, []);

  // 引擎状态订阅：播放时状态栏 4Hz 刷新即可（React 重渲染会占用主线程）
  useEffect(() => {
    if (!eng) return;
    let last = 0;
    eng.onStatus = (s) => {
      const t = performance.now();
      if (t - last > 250 || !s.playing) {
        last = t;
        setSt({ ...s });
      }
    };
  }, [eng]);

  // 加载中显示已用时间
  useEffect(() => {
    if (!loading) return;
    const id = setInterval(() => setNow(performance.now()), 200);
    return () => clearInterval(id);
  }, [loading]);

  // 参考视频就绪：放进并排框；应用 URL 里的初始对照模式
  useEffect(() => {
    if (!bs.videoOk || !eng?.video || !videoBoxRef.current) return;
    eng.video.className = 'ref-video';
    videoBoxRef.current.appendChild(eng.video);
    const init = params.get('compare') as CompareMode | null;
    if (init && COMPARE_MODES.includes(init)) applyCompare(init);
  }, [bs.videoOk]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!eng?.scene || e.ctrlKey || e.metaKey || e.altKey) return;
      const t = e.target as HTMLElement;
      // 文字输入框、下拉框里按键归它们；滑块、单选框、按钮获得焦点时快捷键照常可用（并阻止它们自己的默认行为）
      if ((t instanceof HTMLInputElement && !['range', 'radio', 'checkbox', 'button', 'submit'].includes(t.type)) || t instanceof HTMLSelectElement || t instanceof HTMLTextAreaElement) return;
      const n = e.shiftKey ? 10 : 1;
      let handled = true;
      if (e.key === ' ') void eng.togglePlay();
      else if (e.key === 'ArrowRight') eng.step(n);
      else if (e.key === 'ArrowLeft') eng.step(-n);
      else if (e.key === 'Home') eng.seek(0);
      else if (e.key === 'End') eng.seek(eng.duration - 1);
      else if (e.key === 'd' || e.key === 'D') setDebug((d) => !d);
      else if (e.key === 'm' || e.key === 'M') eng.setMuted(!eng.clock.muted);
      else if (e.key === '[' || e.key === ']') jumpMarker(e.key === ']' ? 1 : -1);
      else if (debug && ['1', '2', '3', '4'].includes(e.key)) {
        const m = COMPARE_MODES[Number(e.key) - 1]!;
        if (m === 'off' || bs.videoOk) applyCompare(m);
      } else handled = false;
      if (handled) e.preventDefault();
    };
    // 按钮有焦点时空格会在 keyup 触发 click，这里一并拦掉，避免「播放/暂停」之外又点了一次按钮
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.key === ' ' && (e.target as HTMLElement).tagName === 'BUTTON') e.preventDefault();
    };
    window.addEventListener('keydown', onKey);
    window.addEventListener('keyup', onKeyUp);
    return () => {
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('keyup', onKeyUp);
    };
  });

  const fps = scene?.timebase.fps ?? 60;
  const frame = st?.frame ?? 0;
  const last = Math.max(0, (scene?.duration ?? 1) - 1);
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

  async function copyLink() {
    const q = new URLSearchParams(location.search);
    q.set('frame', String(eng?.frame ?? frame));
    if (debug) q.set('debug', '1');
    else q.delete('debug');
    if (debug && compare !== 'off') q.set('compare', compare);
    else q.delete('compare');
    // 在当前完整地址上替换查询串（保留代理的路径前缀）
    const u = new URL(location.href);
    u.search = q.toString();
    u.hash = '';
    const url = u.href;
    history.replaceState(null, '', url);
    try {
      await navigator.clipboard.writeText(url);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* 剪贴板不可用时至少地址栏已更新 */
    }
  }

  // 署名：场景 meta.credits + 各模型 credits，去重；附上模型原作者的发布页
  const credits = scene ? [...new Set([...(scene.meta.credits ?? []), ...scene.assets.models.flatMap((m) => (m.credits ? [m.credits] : []))])] : [];
  const pages = scene ? MODEL_PAGES.filter((p) => scene.assets.models.some((m) => p.match.test(m.uri))) : [];
  const waiting = (st?.waitMs ?? 0) > 0;

  return (
    <div className={debug ? 'app debug-open' : 'app'}>
      <header>
        <strong>CharaForge 播放器</strong>
        <span className="muted">{scene?.meta.title}</span>
        <span className="muted small">{bs.label}</span>
      </header>
      {bs.error && (
        <div className="error-box" role="alert">
          <strong>{bs.error.title}</strong>
          {bs.error.hint && <div>{bs.error.hint}</div>}
          <details>
            <summary>详细信息</summary>
            <pre>{bs.error.detail}</pre>
          </details>
        </div>
      )}
      {bs.notices.map((n) => (
        <div key={n} className="notice">
          {n}
        </div>
      ))}
      <div className={`stage compare-${debug ? compare : 'off'}`}>
        <div className="canvas-box">
          <canvas ref={canvasRef} />
          {loading && (
            <div className="overlay">
              <div className="spinner" />
              <div>{bs.stage}…</div>
              <div className="muted small">{((now - bs.t0) / 1000).toFixed(1)} s</div>
            </div>
          )}
          {!loading && waiting &&<div className="badge">定位中（补算物理）… {((st?.waitMs ?? 0) / 1000).toFixed(1)} s</div>}
        </div>
        <div className="video-box" ref={videoBoxRef} style={{ display: debug && compare === 'side' ? undefined : 'none' }} />
      </div>
      <div className="transport">
        <button className="primary" disabled={!scene} onClick={() => eng && void eng.togglePlay()} title="空格">
          {st?.playing ? '❚❚ 暂停' : '▶ 播放'}
        </button>
        <button disabled={!scene} onClick={() => eng?.step(-10)} title="Shift+←">−10</button>
        <button disabled={!scene} onClick={() => eng?.step(-1)} title="←">−1</button>
        <button disabled={!scene} onClick={() => eng?.step(1)} title="→">+1</button>
        <button disabled={!scene} onClick={() => eng?.step(10)} title="Shift+→">+10</button>
        <input className="scrub" type="range" min={0} max={last} value={frame} disabled={!scene} onChange={(e) => eng?.seek(Number(e.target.value))} />
        <span className="mono">
          帧 {frame} / {last}　{timecode(frame, fps)}
        </span>
        <button disabled={!scene} onClick={() => eng?.setMuted(!eng.clock.muted)} title="M">
          {st?.muted ? '🔇 已静音' : '🔊 声音'}
        </button>
      </div>
      <div className="info">
        <span>镜头：{marker ? `${marker.label ?? marker.id}（${marker.start}–${marker.end ?? ''}）` : '—'}</span>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            const m = markers.find((m) => m.id === gotoText || m.label === gotoText);
            const n = m ? m.start : Number(gotoText);
            if (Number.isFinite(n)) eng?.seek(n);
          }}
        >
          <input value={gotoText} onChange={(e) => setGotoText(e.target.value)} placeholder="帧号或镜头（如 s24）" />
          <button type="submit">跳转</button>
        </form>
        <select value="" onChange={(e) => e.target.value && eng?.seek(Number(e.target.value))}>
          <option value="">镜头列表（{markers.length}）</option>
          {markers.map((m) => (
            <option key={m.id} value={m.start}>
              {m.start} · {m.label ?? m.id}
            </option>
          ))}
        </select>
        <button onClick={() => setDebug((d) => !d)} title="D">{debug ? '关闭调试' : '调试模式'}</button>
        <span className="mono muted">
          {st?.playing ? `实测 ${st.fps.toFixed(1)} fps` : '已暂停'} · 每帧渲染 {st ? st.renderMs.toFixed(1) : '—'} ms · workers {eng?.workers ?? '—'}
        </span>
        {st && st.prewarm.done < st.prewarm.total && (
          <span className="small muted" title="Worker 提前为各图层做物理预热（不阻塞播放、不改变画面）。">
            预热图层 {st.prewarm.done}/{st.prewarm.total}
          </span>
        )}
      </div>
      {debug && (
        <div className="debug">
          <div className="row">
            <span>原视频对照：</span>
            {COMPARE_MODES.map((m, i) => (
              <label key={m} title={`快捷键 ${i + 1}`}>
                <input type="radio" checked={compare === m} onChange={() => applyCompare(m)} disabled={m !== 'off' && !bs.videoOk} />
                {COMPARE_LABEL[m]} <kbd>{i + 1}</kbd>
              </label>
            ))}
            {compare === 'overlay' && (
              <label>
                视频不透明度 {mix.toFixed(2)}
                <input type="range" min={0} max={1} step={0.01} value={mix} onChange={(e) => { setMix(Number(e.target.value)); eng?.setOverlay(Number(e.target.value), gain); }} />
              </label>
            )}
            {compare === 'diff' && (
              <label>
                差异放大 ×{gain}
                <input type="range" min={1} max={32} step={1} value={gain} onChange={(e) => { setGain(Number(e.target.value)); eng?.setOverlay(mix, Number(e.target.value)); }} />
              </label>
            )}
            <button onClick={() => void copyLink()} title="把当前帧、调试与对照模式写进地址栏并复制，打开即复现这一帧">
              {copied ? '已复制' : '复制本帧链接'}
            </button>
          </div>
          <div className="row mono">
            <span>
              渲染帧 {frame} · 视频帧 {st?.videoFrame ?? '—'}{' '}
              {compare !== 'off'
                ? st?.videoExact
                  ? <span className="ok">已逐帧对齐</span>
                  : st?.playing
                    ? <span className="warn">播放中，视频近似对齐（误差 ≤ 2 帧；叠加/差异图请暂停后看）</span>
                    : <span className="warn">视频定位中…</span>
                : ''}
              {bs.videoOk === false && <span className="warn"> 参考视频加载失败</span>}
              {bs.videoOk === null && <span className="muted"> 参考视频加载中…</span>}
            </span>
            <span>本帧推进物理 {st?.steps ?? 0} 步</span>
            <span>
              镜头标记 <kbd>[</kbd> <kbd>]</kbd>
            </span>
          </div>
          {compare === 'diff' && <div className="row small muted">差异图 = |渲染合成到原视频上 − 原视频| × 放大倍数（与验收指标的 C_f 同一口径；没画到角色的地方为 0）。叠加 = 渲染合成到原视频上，再与原视频按不透明度混合。</div>}
        </div>
      )}
      <div className="keys small muted">
        快捷键：<kbd>空格</kbd> 播放/暂停　<kbd>←</kbd>/<kbd>→</kbd> ±1 帧（<kbd>Shift</kbd> ±10）　<kbd>Home</kbd>/<kbd>End</kbd>　<kbd>[</kbd>/<kbd>]</kbd> 上/下一个镜头　<kbd>D</kbd> 调试　<kbd>1</kbd>–<kbd>4</kbd> 对照模式　<kbd>M</kbd> 静音
      </div>
      <footer>
        {credits.map((c) => (
          <div key={c}>模型：{linkify(c)}</div>
        ))}
        {pages.map((p) => (
          <div key={p.url}>
            模型发布页（请从原作者处获取并支持原作者）：<a href={p.url} target="_blank" rel="noreferrer">{p.name}</a>
          </div>
        ))}
        {pages.length > 0 && <div className="muted small">模型文件按原作者条款禁止再分发；本页面只供本地查看，不公开部署。</div>}
        <div className="muted small">渲染：Live2D Cubism SDK for Web（Core / Framework），合成：PixiJS v8；CharaForge 代码 MIT 许可。</div>
      </footer>
    </div>
  );
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
