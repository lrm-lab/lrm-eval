# 天禧合成题交付格式 v1

交付可运行的 Harbor 题目。每个任务是一个目录，包含给 Agent 的说明、可复现的执行环境，以及用来验证最终产物的评分脚本。训练集和留出集分别交付。

```text
datasets/<数据集>-v1/
├── train/
│   └── invoice-reconcile-001/
│       ├── task.toml
│       ├── instruction.md
│       ├── environment/
│       │   ├── Dockerfile
│       │   └── inputs/
│       ├── tests/
│       │   └── test.sh
│       └── solution/
│           └── solve.sh
└── holdout/
    └── <采用相同结构的独立题目>/
```

- **题目 ID**用小写字母、数字和连字符。目录名就是运行入口的 `--task` 值。版本冻结后保留任务 ID 和内容。同一题的改版放入新的数据集版本。
- **题目说明**写明可用输入、工作目录、交付文件路径和完成条件。公开输入放在 `environment/inputs/`，由 Dockerfile 拷贝进工作目录。也可在 `task.toml` 指定固定版本或 digest 的预构建镜像。
- **评分**由 `tests/test.sh` 写入 `/logs/verifier/reward.txt`。首版采用 0/1，满足所有条件写 1，有效作答但未通过写 0。评分器本身故障时让脚本报错，保留日志。答案、私有测试和参考解放在 verifier 或 oracle 使用的目录，构建 Agent 镜像时避免加入这些文件。
- **参考解**放在 `solution/solve.sh`，用于交付验收。参考解应通过，空跑应得 0。再检查至少一种典型错误产物，确认评分器能区分。依赖安装和网络权限随环境定义一起锁定。
- **留出题**用独立题目来源或生成种子，供固定回归评测。反馈给合成同事的是训练题的失败类型、难度和脱敏行为轨迹。

## task.toml 示例

```toml
schema_version = "1.1"

[task]
name = "tianxi-synthetic/invoice-reconcile-001"
description = "Reconcile two invoice files and produce a summary."

[metadata]
split = "train"
source = "invoice-generator-v1"
difficulty = "easy"

[agent]
timeout_sec = 300.0

[verifier]
timeout_sec = 60.0

[environment]
build_timeout_sec = 300.0
cpus = 1
memory_mb = 1024
storage_mb = 4096
gpus = 0
```

## 在统一入口登记

在 `framework/harbor.toml` 为两个 split 分别登记。相对路径以该配置文件所在位置为准。

```toml
[datasets.synthetic-v1-train]
name = "synthetic-v1-train"
path = "../datasets/synthetic-v1/train"

[datasets.synthetic-v1-holdout]
name = "synthetic-v1-holdout"
path = "../datasets/synthetic-v1/holdout"
```

先用统一入口的 `--plan` 检查题目路径，再在有 Docker 的执行机做参考解、空跑和真实 Agent 验收。数据集通过后，同一题可切换已配置的 harness 和模型。运行结果进统一的 `episodes.jsonl`，关联评分、ATIF 和可用的训练轨迹。题目交付只需要上述任务文件，轨迹由实际运行生成。

订单对账和离线日志汇总可用 `synthesize.py` 生成。生成器同时写出题包本地的 `harbor.toml`，可直接传给统一入口的 `--config`。先运行 `synthesize.py check <题包目录>` 验证参考解、独立产物、空跑、典型错误和评分器异常。扩展方法及可选教师配置见 [README.md](../README.md#合成和训练)。

## 纯模型题的轻量格式

只需把问题交给模型、拿答案评分的题可交 `cases.jsonl`，无须容器。每行一题，至少有唯一 `id` 和字符串 `prompt`。可选 `images` 是相对于 JSONL 的本地图片路径数组。评分数据留在 `answer` 或其他自定义字段，只交给评分器。

```jsonl
{"id":"choice-001","prompt":"水的化学式是哪一个？只回复选项字母。A. H2O B. CO2","answer":"A"}
{"id":"diagram-001","prompt":"图中有几个圆？只输出数字。","images":["media/diagram-001.png"],"answer":"3"}
```

在 `harbor.toml` 登记该数据集。工具、代码执行等需要考场的题仍用上面的 Harbor 格式。

```toml
[datasets.simple-qa-v1]
name = "simple-qa-v1"
runner = "direct"
path = "../datasets/simple-qa-v1/cases.jsonl"
# 复杂评分可设置已审查的本地 Python 脚本路径：
# scorer = "../datasets/simple-qa-v1/score.py"
```

默认将模型回复和 `answer` 去除首尾空白后做精确匹配，只用于明确要求短答案的题。接历史 Benchmark 时，按原有协议提供评分脚本，避免更换统计口径。自定义评分脚本从 stdin 读单个 JSON 对象 `{"case": {...}, "prediction": "..."}`，在 stdout 输出 `{"reward": 1}` 或其他有限数值。脚本在运行机执行，请使用已审查的脚本。

用 `python3 -B run.py simple-qa-v1 --task choice-001 --model "$QWEN_MODEL" --batch qa-smoke` 运行。地址使用 `OPENAI_BASE_URL`，认证使用 `OPENAI_API_KEY`，多模态题还需要模型网关支持图像输入。直接回复、评分和状态也会写到同一批次的 `episodes.jsonl` 及对应 `result.json`。直接运行标为 `runner=direct`，其行为记录是模型回复。
