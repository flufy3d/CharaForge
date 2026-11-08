/*
 * 等待 Cubism Core 就绪。
 *
 * Core 6（live2dcubismcore.min.js）的 WASM 用 WebAssembly.instantiateStreaming 异步实例化：脚本执行完、全局
 * Live2DCubismCore 已经存在时，底层函数可能还没装好，这时调用会抛 "Cannot read properties of undefined (reading 'apply')"。
 * 开发服务器下模块是逐个请求的，等到 Framework 启动时 Core 早已就绪；pnpm build 之后 Worker 的 workerMain 被打进同一个
 * chunk，import 立刻完成，就会撞上。所以启动 Framework 之前先探测一次 Version.csmGetVersion()，失败就稍后重试。
 * 本模块不引用 Framework（Framework 有模块在加载时就读全局 Live2DCubismCore）。
 */
export async function waitForCubismCore(timeoutMs = 20_000): Promise<void> {
  const t0 = Date.now();
  for (;;) {
    const core = (globalThis as { Live2DCubismCore?: { Version?: { csmGetVersion(): number } } }).Live2DCubismCore;
    if (!core) throw new Error('未找到 Live2DCubismCore：请先加载 vendor/cubism/core/live2dcubismcore.min.js（pnpm setup:cubism）');
    try {
      core.Version!.csmGetVersion();
      return;
    } catch (e) {
      if (Date.now() - t0 > timeoutMs) throw new Error(`Cubism Core 初始化超时（${timeoutMs} ms）：${String(e)}`);
    }
    await new Promise((r) => setTimeout(r, 10));
  }
}
