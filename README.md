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

当前测试基线：46 个测试通过。

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

## 当前限制

- Qwen-Image 与 Qwen2.5-VL 每次任务重新加载，首轮延迟较高。
- OCR 与 VQA 目前顺序执行，后续应改为常驻模型 worker。
- 扩散模型无法保证每次生成的文字完全正确，OCR 负责检测但不保证修复成功。
- PowerPaint 只应用于图生图/局部编辑任务，文生图失败时默认重新生成。

## 安全与发布

- 不要提交 `.env`、模型权重、数据集、日志、生成图片或凭据。
- 上传前运行 `git status --short` 和测试。
- 本项目不包含模型权重；使用模型时请遵守各模型许可证。
