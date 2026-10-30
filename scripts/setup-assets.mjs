// 下载并整理第一期用到的外部资源（均不进仓库）：
//   assets/models/deepseek-chan/   DeepSeek-chan Live2D 模型（排除 131MB 分层 PSD）
//   assets/video/DeepSeek-chan.mp4 参考视频
//   assets/video/DeepSeek-chan.wav 参考音频（ffmpeg 解码为 PCM，避免 AAC 编码延迟导致的音画偏移）
import { copyFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { execFileSync } from 'node:child_process';
import { ROOT, download, extractZip } from './lib.mjs';

const MODEL = {
  url: 'https://file.360.world/James/DeepSeek/DeepSeek-chan%20%20DeepSeekFree%20Live2D%20Model%20%2B%20Layered%20PSD%20%20PSD.zip',
  name: 'DeepSeek-chan-model.zip',
  sha256: 'f3341cda80c678bf8ee7e88e3b6f0153c95a327de1a99294ae0d1bcb48f0d777',
};
const VIDEO = {
  url: 'https://file.360.world/James/DeepSeek/DeepSeek-chan.mp4',
  name: 'DeepSeek-chan.mp4',
  sha256: '7832e91305585f6fe787d59036bb73bec83153051f9bd43b67554130c2e49afa',
};

const [modelZip, video] = await Promise.all([download(MODEL), download(VIDEO)]);

// 压缩包顶层目录名含空格和全角字符，这里去掉顶层目录；模型文件保持原名（模型条款要求不改名）。
const modelDir = join(ROOT, 'assets', 'models', 'deepseek-chan');
const n = extractZip(modelZip, modelDir, (name) => {
  const rel = name.split('/').slice(1).join('/');
  if (!rel || rel.startsWith('psd/')) return null;
  if (rel.startsWith('DeepSeek/')) return rel;
  if (rel.endsWith('.txt')) return `terms/${rel}`; // 使用条款
  return null;
});
console.log(`模型：${n} 个文件 → assets/models/deepseek-chan/`);

const videoDir = join(ROOT, 'assets', 'video');
mkdirSync(videoDir, { recursive: true });
copyFileSync(video, join(videoDir, 'DeepSeek-chan.mp4'));
execFileSync('ffmpeg', [
  '-v', 'error', '-y', '-i', join(videoDir, 'DeepSeek-chan.mp4'),
  '-vn', '-c:a', 'pcm_s16le', join(videoDir, 'DeepSeek-chan.wav'),
], { stdio: 'inherit' });
console.log('视频与音频 → assets/video/');
