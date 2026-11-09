/*
 * 编辑器入口：创建单例 store / 引擎，挂载 React 根。
 * 引擎初始化在 effect 里做（canvas 已挂载），StrictMode 的重复 effect 由 store.init 的幂等保护。
 */
import { StrictMode, useEffect, useRef, useState, useSyncExternalStore } from 'react';
import { createRoot } from 'react-dom/client';
import { EditorShell } from './shell.tsx';
import { createEditorStore } from './store.ts';

const store = createEditorStore();

function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const state = useSyncExternalStore(
    (listener) => store.subscribe(listener),
    () => store.getState(),
    () => store.getState(),
  );
  const [now, setNow] = useState(performance.now());
  const t0Ref = useRef(performance.now());

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    void store.init(canvas).then(() => {
      (window as unknown as { __player?: unknown }).__player = store.engine;
    });
  }, []);

  useEffect(() => {
    if (state.phase !== 'booting') return;
    t0Ref.current = performance.now();
    setNow(performance.now());
    const timer = window.setInterval(() => setNow(performance.now()), 200);
    return () => window.clearInterval(timer);
  }, [state.phase]);

  return (
    <>
      <EditorShell store={store} canvasRef={canvasRef} />
      {state.phase !== 'ready' && (
        <div className="boot-overlay" role={state.phase === 'error' ? 'alert' : 'status'}>
          <div className="boot-card">
            {state.phase === 'error' ? (
              <>
                <h2>编辑器启动失败</h2>
                <pre className="boot-error">{state.error}</pre>
                <div className="small muted">
                  常见原因：没有运行 pnpm run setup（Cubism Core / 模型素材）、场景路径不存在，或浏览器不支持 WebGL2。
                </div>
                <button type="button" className="primary" onClick={() => location.reload()}>
                  重新加载
                </button>
              </>
            ) : (
              <>
                <div className="spinner" />
                <h2>CharaForge 场景编辑器</h2>
                <div>{state.stage}…</div>
                <div className="small muted">
                  已用 {((now - t0Ref.current) / 1000).toFixed(1)} s · 大型场景首次加载需要数秒
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </>
  );
}

const container = document.getElementById('root');
if (!container) throw new Error('缺少 #root');
createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
