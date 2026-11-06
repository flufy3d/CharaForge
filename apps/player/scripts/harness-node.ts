/*
 * Node 端：启动 Vite 开发服务器（与 pnpm dev 同一份配置）+ 启用 GPU 的无头 Chromium，打开 harness.html。
 * render CLI、gallery、runtime 的确定性测试、拟合器都通过这里拿到 window.__charaforge。
 *
 * GPU 启动参数（RTX 2060 / Linux 实测，记录见 docs/runtime-eval.md）：
 *   - 去掉 DISPLAY：本机 X 显示 :1 无响应时，Chromium 的 GPU 进程会卡在连接 X 上，导致 WebGL 永远挂起；
 *     无头模式不需要 X。
 *   - 缺省 ANGLE/Vulkan：--use-angle=vulkan --enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan
 *     --use-vulkan=native，并用 VK_ICD_FILENAMES 指向 NVIDIA ICD（否则可能选到核显/lavapipe）。
 *     实测 ANGLE/GL-EGL 后端的 MSAA resolve 不确定（同一输入两次绘制有 ±1 的像素差），Vulkan 后端逐位稳定。
 *   - CHARAFORGE_GPU=egl：--use-angle=gl-egl + __EGL_VENDOR_LIBRARY_FILENAMES 指向 NVIDIA 的 glvnd 描述文件
 *     （否则 EGL 默认选 Mesa / Intel 核显）。此后端需配合 CHARAFORGE_MSAA=0 才能逐位确定。
 *   - --ignore-gpu-blocklist
 *   - WSL2（RTX 5080 Laptop / Windows 驱动 615.71 / Mesa d3d12，2026-11 实测，记录见 docs/runtime-eval.md）：
 *     WSL 里没有 NVIDIA 的 Vulkan / EGL ICD，GPU 只能经 Mesa 的 d3d12 Gallium 驱动（/usr/lib/wsl/lib/libd3d12.so）走 D3D12。
 *     缺省 vulkan 和 --use-angle=gl 都落到 llvmpipe / SwiftShader；gl-egl + GALLIUM_DRIVER=d3d12
 *     + MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA 得到 "ANGLE (Microsoft Corporation, D3D12 (NVIDIA GeForce RTX 5080 Laptop GPU),
 *     OpenGL ES 3.1)"（不设 adapter 名会选中 Intel 核显）。此后端 MSAA=4 时确定性测试 10/13 失败（不确定），
 *     MSAA=0 时只剩第 8 项（它要求开着 MSAA 才成立，msaa=0 时跳过）。
 *     所以检测到 WSL（有 libd3d12.so 且没有 NVIDIA Vulkan ICD）时，未显式设置的项缺省为
 *     CHARAFORGE_GPU=egl、GALLIUM_DRIVER=d3d12、MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA、MSAA 0；显式设置的环境变量优先。
 * 启动后检查 WebGL renderer 字符串，不含 CHARAFORGE_EXPECT_GPU（缺省 "NVIDIA"）就报错退出，防止悄悄落到 SwiftShader。
 */
import { existsSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium, type Browser, type Page } from 'playwright';
import { createServer, type ViteDevServer } from 'vite';
import type { CharaForgeHarness } from '../src/harness-api.ts';

const HERE = dirname(fileURLToPath(import.meta.url));
export const PLAYER_DIR = resolve(HERE, '..');
export const REPO = resolve(PLAYER_DIR, '..', '..');

const NVIDIA_EGL = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json';
const NVIDIA_ICD = '/usr/share/vulkan/icd.d/nvidia_icd.json';
const WSL_D3D12 = '/usr/lib/wsl/lib/libd3d12.so';

/** WSL2 且只能经 Mesa d3d12 用 GPU（没有 NVIDIA 的 Vulkan ICD）。原生 Linux 恒为 false。 */
export const WSL_D3D12_ONLY =
  existsSync(WSL_D3D12) && !existsSync(NVIDIA_ICD) && !existsSync('/etc/vulkan/icd.d/nvidia_icd.json');

function gpuMode(): string {
  return process.env.CHARAFORGE_GPU ?? (WSL_D3D12_ONLY ? 'egl' : 'vulkan');
}

/** harness 的 MSAA 缺省：CHARAFORGE_MSAA 优先；WSL d3d12 走 egl 时为 0；否则 undefined（用 harness.ts 的缺省 4）。 */
export function defaultMsaa(): number | undefined {
  if (process.env.CHARAFORGE_MSAA !== undefined) return Number(process.env.CHARAFORGE_MSAA);
  return WSL_D3D12_ONLY && gpuMode() === 'egl' ? 0 : undefined;
}

export function gpuLaunchConfig(): { args: string[]; env: Record<string, string> } {
  const mode = gpuMode();
  const env: Record<string, string> = {};
  for (const [k, v] of Object.entries(process.env)) if (v !== undefined && k !== 'DISPLAY' && k !== 'WAYLAND_DISPLAY') env[k] = v;
  const args = ['--ignore-gpu-blocklist', '--enable-gpu-rasterization', '--disable-gpu-sandbox'];
  if (mode === 'vulkan') {
    args.push('--use-angle=vulkan', '--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan', '--use-vulkan=native');
    if (existsSync(NVIDIA_ICD)) env.VK_ICD_FILENAMES = NVIDIA_ICD;
  } else if (mode === 'egl') {
    args.push('--use-angle=gl-egl');
    if (existsSync(NVIDIA_EGL)) env.__EGL_VENDOR_LIBRARY_FILENAMES = NVIDIA_EGL;
    if (WSL_D3D12_ONLY) {
      env.GALLIUM_DRIVER ??= 'd3d12';
      env.MESA_D3D12_DEFAULT_ADAPTER_NAME ??= 'NVIDIA';
    }
  } else if (mode !== 'default') {
    throw new Error(`未知 CHARAFORGE_GPU=${mode}（egl | vulkan | default）`);
  }
  if (process.env.CHARAFORGE_CHROME_ARGS) args.push(...process.env.CHARAFORGE_CHROME_ARGS.split(/\s+/).filter(Boolean));
  return { args, env };
}

export interface Harness {
  server: ViteDevServer;
  browser: Browser;
  page: Page;
  baseUrl: string;
  info: { vendor: string; renderer: string; msaa: number; workers: number };
  close(): Promise<void>;
}

export async function startServer(): Promise<{ server: ViteDevServer; baseUrl: string }> {
  const server = await createServer({
    configFile: join(PLAYER_DIR, 'vite.config.ts'),
    root: PLAYER_DIR,
    logLevel: 'error',
    server: { port: 0, host: '127.0.0.1', hmr: false },
  });
  await server.listen();
  const addr = server.httpServer?.address();
  if (!addr || typeof addr === 'string') throw new Error('Vite 服务器没有监听端口');
  return { server, baseUrl: `http://127.0.0.1:${addr.port}` };
}

export async function launchBrowser(): Promise<Browser> {
  const { args, env } = gpuLaunchConfig();
  return chromium.launch({ args, env, headless: true, timeout: 60_000 });
}

/** 打开 harness.html，等待就绪并校验 GPU。 */
export async function openHarness(opts: { msaa?: number; workers?: number; expectGpu?: string | null } = {}): Promise<Harness> {
  const { server, baseUrl } = await startServer();
  const browser = await launchBrowser();
  const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
  page.on('console', (m) => {
    if ((m.type() === 'error' || m.type() === 'warning') && !m.text().includes('GPU stall due to ReadPixels')) process.stderr.write(`[page ${m.type()}] ${m.text()}\n`);
  });
  page.on('pageerror', (e) => process.stderr.write(`[page error] ${e.message}\n`));
  const msaa = opts.msaa ?? defaultMsaa();
  const qs = new URLSearchParams();
  if (msaa !== undefined) qs.set('msaa', String(msaa));
  if (opts.workers) qs.set('workers', String(opts.workers));
  const q = qs.size ? `?${qs}` : '';
  await page.goto(`${baseUrl}/harness.html${q}`);
  await page.waitForFunction(() => !!window.__charaforge, null, { timeout: 60_000 });
  await page.evaluate(() => window.__charaforge.ready);
  const info = await page.evaluate(() => window.__charaforge.info());
  const expect = opts.expectGpu === undefined ? process.env.CHARAFORGE_EXPECT_GPU ?? 'NVIDIA' : opts.expectGpu;
  const close = async () => {
    await browser.close().catch(() => {});
    await server.close().catch(() => {});
  };
  if (expect && !info.renderer.includes(expect)) {
    await close();
    throw new Error(`WebGL renderer 不是 GPU（期望包含 "${expect}"）：${info.renderer}\n可设置 CHARAFORGE_GPU=vulkan 或 CHARAFORGE_EXPECT_GPU 调整`);
  }
  return { server, browser, page, baseUrl, info, close };
}

export type { CharaForgeHarness };
