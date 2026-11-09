/*
 * 场景编辑 CLI（AI / HTTP 后端共用 packages/scene 的 edit 命令层）。
 *
 *   node --experimental-strip-types --disable-warning=ExperimentalWarning \
 *     apps/player/scripts/editor-command.ts --scene <path> [--input <json>] [--out <path>] [--dry-run] [--frame <n>]
 *
 * 不传 --input 时从 stdin 读 JSON。输入可以是命令数组、{ commands: [...] } 或单个命令对象。
 * 不传 --out 时只返回 diff，不写文件；--dry-run 时即使给了 --out 也不写。
 * 输出：{ ok, scene, out, dryRun, commandCount, changed, effects, diff, preview }
 * 出错：stderr 输出 { ok:false, error } 并 exit 1。
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { applyEditCommands, parseScene, type EditCommand } from '@charaforge/scene';

function usage(): void {
  console.error(`用法：editor-command --scene <scene.json> [--input <commands.json>] [--out <scene.json>] [--dry-run] [--frame <n>]

不传 --input 时从 stdin 读 JSON。输入格式见 apps/player/scripts/editor-command.ts 文件头。`);
}

function fail(message: string): never {
  throw new Error(message);
}

function readInput(path: string | undefined): unknown {
  const text =
    !path || path === '-'
      ? readFileSync(0, 'utf8')
      : readFileSync(resolve(process.env.INIT_CWD ?? process.cwd(), path), 'utf8');
  try {
    return JSON.parse(text);
  } catch (error) {
    fail(`命令 JSON 解析失败：${String((error as Error)?.message ?? error)}`);
  }
}

function commandsOf(value: unknown): EditCommand[] {
  if (Array.isArray(value)) return value as EditCommand[];
  if (value && typeof value === 'object') {
    const record = value as Record<string, unknown>;
    if (Array.isArray(record.commands)) return record.commands as EditCommand[];
    if (typeof record.op === 'string') return [record as unknown as EditCommand];
  }
  fail('输入必须是命令数组、{ commands: [...] } 或单个命令对象');
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2).filter((arg) => arg !== '--');
  const { values } = parseArgs({
    args: argv,
    options: {
      scene: { type: 'string' },
      input: { type: 'string' },
      out: { type: 'string' },
      'dry-run': { type: 'boolean' },
      frame: { type: 'string' },
      help: { type: 'boolean' },
    },
  });
  if (values.help) {
    usage();
    return;
  }
  if (!values.scene) fail('缺少 --scene');

  const cwd = process.env.INIT_CWD ?? process.cwd();
  const scenePath = resolve(cwd, values.scene);
  const rawScene: unknown = JSON.parse(readFileSync(scenePath, 'utf8'));
  const scene = parseScene(rawScene);
  const commands = commandsOf(readInput(values.input));

  let frame: number | undefined;
  if (values.frame !== undefined) {
    frame = Number(values.frame);
    if (!Number.isFinite(frame) || frame < 0) fail('--frame 必须是非负数字');
  }

  const result = applyEditCommands(scene, commands, { frame });
  const outPath = values.out ? resolve(cwd, values.out) : null;
  if (outPath && !values['dry-run']) writeFileSync(outPath, JSON.stringify(scene, null, 2) + '\n');

  console.log(
    JSON.stringify(
      {
        ok: true,
        scene: scenePath,
        out: outPath,
        dryRun: !!values['dry-run'],
        commandCount: commands.length,
        changed: result.changes.length > 0,
        effects: result.effects,
        diff: result.changes,
        preview: result.preview ?? null,
      },
      null,
      2,
    ),
  );
}

main().catch((error) => {
  console.error(JSON.stringify({ ok: false, error: String((error as Error)?.message ?? error) }, null, 2));
  process.exit(1);
});
