# Marketing Creative Agent

基于 LangGraph 的营销图像生成与编辑 Agent。它不是固定 workflow：Agent 会根据真实视觉评估结果选择需要优先修复的 2–3 个维度，迭代生成候选图，并保留全局最佳版本。

## 核心能力

- Qwen-Image 文生图
- Grounding DINO + SAM2 目标定位与分割
- PowerPaint 局部编辑
- Qwen2.5-VL/VQAScore 十维美学评估
- Qwen2.5-VL OCR 文字准确性与重复检测
- Top-3 低分维度自动修复
- 动态且可复现的随机种子
- 最佳候选保留与质量平台期早停
- FastAPI 异步任务接口

## 目录结构

```text
marketing_agent/       Agent、工具、模型适配器和 API
tests/                 单元测试与接口测试
scripts/               启动、验收、模型检查和实验入口
requirements/          主环境及隔离模型环境依赖
docs/                  模型与验收说明
runtime/               本地模型、环境、日志和输出（Git 忽略）
```

## Agent 闭环

```text
理解目标 → 生成/编辑 → VQA + OCR 评估
                         ↓
                 选择最低 2–3 个维度
                         ↓
                生成具体中文修复指令
                         ↓
                 新候选 → 再评估
                         ↓
             通过 / 平台期早停 / 达到预算
```

## 环境要求

- Python 3.10
- NVIDIA GPU，推荐 80GB 显存
- CUDA 12.x
- 模型权重需放入 `runtime/models/`

期望的模型目录：

```text
runtime/models/Qwen-Image/
runtime/models/Qwen2.5-VL-3B-Instruct/
runtime/models/PowerPaint-v2/
runtime/models/grounding-dino-base/
runtime/models/sam2/
```

模型权重、第三方仓库、虚拟环境和实验输出不进入 Git。

## 安装

```bash
python -m venv runtime/venv/gpu
source runtime/venv/gpu/bin/activate
pip install -r requirements/base.txt

# 按需创建 PowerPaint 与 VQAScore/OCR 隔离环境
bash scripts/setup_model_envs.sh all
```

复制环境变量模板：

```bash
cp .env.example .env
```

所有路径默认相对于项目根目录，也可以通过环境变量覆盖。

## 运行测试

```bash
PYTHONPATH="$PWD" runtime/venv/gpu/bin/python -m pytest -q tests
```

当前测试基线：88 个测试通过。

## 命令行运行

```bash
export MARKETING_AGENT_ROOT="$PWD"
export PYTHONPATH="$PWD"

runtime/venv/gpu/bin/python scripts/run_qwen_agent.py \
  "生成一张高端护肤精华液商业海报" \
  --production \
  --quality-threshold 0.87 \
  --max-iterations 8
```

## 启动 API

```bash
bash scripts/run_api.sh
```

默认监听 `0.0.0.0:8000`。健康检查：

```bash
curl http://127.0.0.1:8000/health
```

### 品牌规范与结构化文案

任务可直接携带品牌档案，不需要预先创建品牌数据库。Agent 会把品牌色、调性、
必显/禁用文案、视觉规则和素材逻辑 ID 注入每轮生成，并在结果中返回结构化文案。

```json
{
  "prompt": "生成高端精华活动海报，标题《奢润新生》",
  "generate_copy": true,
  "brand_profile": {
    "brand_id": "lumiere",
    "name": "LUMIÈRE",
    "tone": ["高端", "克制"],
    "primary_colors": ["象牙白", "香槟金"],
    "required_phrases": ["焕亮新生"],
    "forbidden_phrases": ["全网最低"],
    "visual_rules": ["品牌名只能出现一次"]
  }
}
```

不需要自动生成文案时传入 `"generate_copy": false`。

### Top-K 多候选生成

通过 `candidate_count` 请求 2～8 个独立候选。每个候选使用可复现的独立种子，
最终按“文字合规优先、营销评分其次”自动选择；结果中的 `candidate_summaries`
保留所有候选的分数、合规状态和选中标记。

```json
{
  "prompt": "生成高端精华活动海报",
  "candidate_count": 3,
  "seed": 42,
  "parallel_candidates": false
}
```

GPU 模式建议保持 `parallel_candidates=false`，避免多个扩散模型同时争用显存。

### 多尺寸渠道适配

`output_formats` 支持 `1:1`、`4:5`、`9:16` 和 `16:9`。Agent 会针对每种
画幅注入独立安全区规则并分别选出最佳候选，结果通过 `format_summaries` 返回。

```json
{
  "prompt": "生成全渠道香水活动海报",
  "output_formats": ["1:1", "4:5", "9:16", "16:9"],
  "candidate_count": 2
}
```

以上请求会运行 4 个画幅 × 2 个候选。当前多尺寸仅支持文生图任务；图像编辑仍保持
原图尺寸，防止 PowerPaint 在未重排版时产生拉伸结果。

### 人工审核与恢复执行

提交任务时设置 `"review_required": true`，Agent 完成候选选择后会进入
`waiting_for_review`。预览结果已经持久化，但批准前不会写入案例记忆。

批准交付：

```bash
curl -X POST http://127.0.0.1:8000/tasks/TASK_ID/review \
  -H 'Content-Type: application/json' \
  -d '{"decision":"approve","reviewer":"creative_lead"}'
```

带反馈修订：

```bash
curl -X POST http://127.0.0.1:8000/tasks/TASK_ID/review \
  -H 'Content-Type: application/json' \
  -d '{"decision":"revise","feedback":"产品放大到画面高度55%，标题上移"}'
```

修订会归档上一轮结果、使用新种子重新执行，并再次进入审核点。默认最多修订 3 轮，
可通过 `max_review_rounds` 调整。审核历史保存在结果的 `review_history` 字段中。

### 实验与决策看板

API 启动后访问：

```text
http://127.0.0.1:8000/dashboard
```

看板直接读取任务目录，提供任务状态、平均最佳分、待审核队列、各画幅最佳图、
Top-K 候选对比和 Agent 决策轨迹。机器可读汇总接口为：

```text
GET /dashboard/api/summary
```

任务详情页：

```text
GET /dashboard/tasks/{task_id}
```

生成图通过受限资产接口展示，接口只允许读取对应任务目录内且已登记在结果中的文件。

### 常驻模型 Worker 与 GPU 调度

生产模式默认使用 JSONL 长驻 Worker。PowerPaint、VQAScore 和 OCR 首次调用加载模型，
后续请求复用同一进程和模型对象；Qwen-Image pipeline 在 API 进程内复用。

```bash
MODEL_WORKER_MODE=persistent
MODEL_WORKER_TIMEOUT_SECONDS=900
GPU_MAX_CONCURRENT=1
```

所有真实 GPU 工具共享调度器，默认串行运行。`parallel_candidates=true` 只会并行准备
候选，GPU 调用仍受 `GPU_MAX_CONCURRENT` 限制。调度状态可通过 `/health` 的
`runtime.gpu_schedulers` 查看，包括当前活跃数、历史峰值和完成计数；每次 Observation
的 `gpu_queue_seconds` 记录该次调用的排队时间。

显存不足或需要逐次释放模型时，可以回退到旧模式：

```bash
MODEL_WORKER_MODE=oneshot
```

### 历史经验学习与策略记忆

策略记忆记录“低分维度 → 修复动作 → 分数变化”，区别于保存成品参考的案例 RAG。
只有带来有效提升的动作会进入后续推荐；失败动作只用于统计。人工审核任务在批准前
不会学习。

默认关闭且不会创建文件。需要跨任务持久化时启用：

```bash
EXPERIENCE_MEMORY_ENABLED=true
EXPERIENCE_MEMORY_PATH=runtime/memory/experience.jsonl
```

查看当前累计经验和推荐策略：

```text
GET /experience/strategies
```

任务结果中的 `experience_used` 表示本次是否采用历史策略，
`learned_experience_count` 表示本次新增了多少条可统计经验。

## 当前限制

- Qwen-Image 与 Qwen2.5-VL 每次任务重新加载，首轮延迟较高。
- OCR 与 VQA 目前顺序执行，后续应改为常驻模型 worker。
- 扩散模型无法保证每次生成的文字完全正确，OCR 负责检测但不保证修复成功。
- PowerPaint 只应用于图生图/局部编辑任务，文生图失败时默认重新生成。

## 安全与发布

- 不要提交 `.env`、模型权重、数据集、日志、生成图片或凭据。
- 上传前运行 `git status --short` 和测试。
- 本项目不包含模型权重；使用模型时请遵守各模型许可证。
