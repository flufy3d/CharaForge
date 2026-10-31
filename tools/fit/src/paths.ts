import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export const REPO = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
export const MODEL_URI = 'models/deepseek-chan/DeepSeek/大肥鱼.model3.json';
export const FIT_DIR = resolve(REPO, '.data/fit');
/** 结果目录后缀（上限实验用 FIT_TAG=_synth，避免覆盖正式结果） */
export const FIT_TAG = process.env.FIT_TAG ?? '';
