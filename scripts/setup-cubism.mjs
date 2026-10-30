// 下载 Live2D Cubism SDK for Web 5-r.5，校验后把 Core 与 Framework 安装到 vendor/cubism/。
// Core 受 Live2D Proprietary Software License 约束，Framework 受 Live2D Open Software License 约束，
// 两者都不进入本仓库。
import { join } from 'node:path';
import { ROOT, download, extractZip } from './lib.mjs';

const SDK = {
  url: 'https://cubism.live2d.com/sdk-web/bin/CubismSdkForWeb-5-r.5.zip',
  name: 'CubismSdkForWeb-5-r.5.zip',
  sha256: '67064a7fb1812cf502f5c4a03bfe12cc638c75a621bb4acf06bb28763df06ba0',
};
const PREFIX = 'CubismSdkForWeb-5-r.5/';

console.log(`
Live2D Cubism SDK 授权提示：
  - Cubism Core：Live2D Proprietary Software License
    https://www.live2d.com/eula/live2d-proprietary-software-license-agreement_en.html
  - Cubism Framework：Live2D Open Software License
    https://www.live2d.com/eula/live2d-open-software-license-agreement_en.html
  继续安装即表示你已阅读并同意上述条款。
`);

const zip = await download(SDK);
const out = join(ROOT, 'vendor', 'cubism');
const n = extractZip(zip, out, (name) => {
  if (!name.startsWith(PREFIX)) return null;
  const rel = name.slice(PREFIX.length);
  if (rel.startsWith('Core/')) return rel.replace(/^Core\//, 'core/');
  if (rel.startsWith('Framework/src/')) return rel.replace(/^Framework\//, 'framework/');
  if (rel === 'Framework/LICENSE.md') return 'framework/LICENSE.md';
  if (rel === 'LICENSE.md' || rel === 'NOTICE.md') return rel;
  return null;
});
console.log(`已安装 ${n} 个文件到 vendor/cubism/（core/、framework/src/）`);
