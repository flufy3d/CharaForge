/**
 * 前向模型的可选变体（诊断实验用，FIT_MODEL='{"decouple":true,...}'）。确认有效的变体会改成缺省开启。
 *   decouple  身体（PositionX/Y/Z）由独立的面捕输入 BodyX/BodyY/BodyZ 驱动，而不是 vtube.json 里的 FaceAngleX/Y/Z
 *             （PV 作者录制时的 VTS 配置可能与随模型发布的不同；VTS 也有 FacePositionX/Y/Z 等输入）
 *   bodyJac   身体链的次级物理输出（肩、盆骨、BodyAngleRX/X2/Y2、前倾 Param50/53/54……）进雅可比
 *   keyEvery  图层变换关键帧间隔（帧）；小间隔 ≈ 模型随头部在画面上移动（ModelPositionMovement）
 *   consts    镜头常量（前倾、体型）作为连续未知量进 GN（--fixed 给的是初值）
 *   clamp     GN 步长逐分量截断（缺省整步等比缩小）
 *   reg       变换初值粗配准（逐关键帧、低分辨率梯度 NCC 的多起点 CMA）
 *   anneal    鲁棒核（Cauchy）尺度退火的起点（px），每次 GN ×0.85
 *   perframe  逐帧精修迭代数（缺省 12；0 关闭）：每个节点帧放开全部可见参数（postPhysics add）与变换偏移，光流 GN，节点间插值后导出
 *   upper     诊断：拟合结束后每个节点帧独立放开全部可见参数与变换，做 N 次光流 GN，测表达能力上限
 */
export interface ModelX { decouple?: boolean; bodyJac?: boolean; keyEvery?: number; consts?: boolean; upper?: number; clamp?: boolean; reg?: boolean; anneal?: number; perframe?: number }
export const MODELX: ModelX = JSON.parse(process.env.FIT_MODEL ?? '{}');
