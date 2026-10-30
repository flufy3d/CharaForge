// 把 TypeBox 定义导出为 JSON Schema（draft-07），供 Python 工具和外部编辑器使用。
import { writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { Scene } from '../src/scene.ts';
import { Shots } from '../src/shots.ts';

const dir = join(import.meta.dirname, '..', 'schema');
for (const [name, schema] of [['scene.v1.json', Scene], ['shots.v1.json', Shots]] as const) {
  const json = { $schema: 'http://json-schema.org/draft-07/schema#', ...JSON.parse(JSON.stringify(schema)) };
  writeFileSync(join(dir, name), JSON.stringify(json, null, 2) + '\n');
  console.log(`schema/${name}`);
}
