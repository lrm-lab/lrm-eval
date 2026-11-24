# LRM Eval

![LRM Eval architecture — unified benchmarking, Harbor evaluation, reporting, and training feedback](assets/lrm-eval-architecture.svg)

统一的 Harbor 评测入口，支持自定义任务、多个模型和 harness、批次报告、训练采集验收及 OpenEnv + TRL 训练。使用 Harbor 自带的 Viewer 浏览原始作业。

| 入口 | 用途 |
|---|---|
| `framework/run.py` | 在 Harbor 执行评测、导出 ATIF、暂停派发及续跑 |
| `framework/report.py` | 重算批次报告，检查冒烟结果和训练采集 |
| `framework/synthesize.py` | 生成、验收合成题，按真实分数筛选训练候选 |
| `framework/train.py` | 调用 TRL 的 OpenEnv Harbor 异步 GRPO 示例 |

## 运行一条自定义 Harbor 题

Linux 执行机需要 Python 3.12+、Docker、Compose、Buildx 和 `uvx`。默认使用 Harbor 0.26.0 和 LiteLLM 1.110.1。数据集配置及任务文件见 [DATASET_FORMAT.md](framework/DATASET_FORMAT.md)。

```bash
git clone https://github.com/lrm-lab/lrm-eval.git
cd lrm-eval/framework

python3 -B synthesize.py generate demo-v1 --method log-rollup \
  --train-per-method 1 --holdout-per-method 0
python3 -B synthesize.py check ../datasets/demo-v1

python3 -B run.py demo-v1-train --config ../datasets/demo-v1/harbor.toml \
  --task log-rollup-train-001 --agent oracle --batch demo-oracle
python3 -B report.py ../harbor-runs/demo-oracle --format markdown
```

实际模型评测从环境读取网关配置和凭据。设置 `OPENAI_BASE_URL` 与 `OPENAI_API_KEY` 后，选用有该模型协议支持的 harness。例如，接 Chat Completions 接口时设置 `MODEL_ROUTE`，用一个短题确认工具执行、评分和轨迹：

```bash
python3 -B run.py demo-v1-train --config ../datasets/demo-v1/harbor.toml \
  --task log-rollup-train-001 --agent opencode --model "$MODEL_ROUTE" \
  --api-format openai_chat_completions --batch demo-smoke
python3 -B report.py ../harbor-runs/demo-smoke --check-smoke --format markdown
```

默认 OpenCode 版本为 1.19.3。执行机已有离线交接包时，设置 `TIANXI_BENCH_ASSETS` 后运行 `bash setup.sh --opencode`，以只读方式把固定二进制挂入任务容器。运行其他题库时可用 `--config` 登记其本地 Harbor 格式路径，也可用 `--bundle` 指向已有交接包。自定义容器代理可用 `--agent-env` 转发，信任的 CA 用 `--agent-ca-bundle` 传入。

切换模型、harness、网关或工具环境时，先完成一次真实冒烟。正式批次可重复传 `--task`，或用 `--all`；`--attempts` 定义每题采样次数，`--output` 指定导出目录。每个组合用独立批次名。

## 计分和运行记录

每批有 `episodes.jsonl`、`batch-report.json`、`batch-status.json` 及按 trial UUID 索引的评分和可用 ATIF。报告包含计划、已结束、有效评分、有效样本均分、通过数、覆盖率和故障分类。计划 8 次、5 次有效且全通过时，报告分别显示 **5/5 通过** 和 **5/8 覆盖率**。历史批次缺少计划时可传 `--planned`，逐题补分母可用 `--attempts-per-task`。

在官方评分器完成计分后，用 `report.py --official-result <official.json>` 记录其正式分数及分母。文件字段为 `dataset`、`metric`、`score`、`denominator` 和 `source`。有效均分用于分析本批有效样本。浏览原生会话时，运行 `harbor view <原始作业目录>`。

运行故障用固定摘要和执行机日志位置标识。默认连续三次同类故障会暂停派发，阈值用 `--pause-after` 调整。原配置下给原命令追加 `--resume`，由 Harbor 接着执行未完成的试验。框架会先核对题库、模型、harness、参数及原生配置锁。可复用 `--export-job` 导入已有 Harbor 作业。

## 合成和训练

`synthesize.py generate` 默认生成订单对账和离线日志汇总各 20 道训练候选、6 道留出题。参考解、独立正确产物、空跑、典型错误及评分器故障用 `synthesize.py check` 验收。生成的题包含可直接运行的 `harbor.toml`。在 `[synthesis.methods]` 注册额外的 `module:function`，在 `[synthesis.auditors]` 登记对应的独立验算函数。

可选 `--teacher` 按环境变量 `TIANXI_SYNTH_TEACHER_API_KEY` 调用配置的 Chat Completions 教师，为各模板生成背景、边界条件建议和示范思路。默认教师为 DeepSeek V4.1 Flash，可通过独立配置文件更换端点或模型。

训练候选用同一模型和同一 OpenCode 版本每题采样 8 次，随后运行 `synthesize.py select <episodes.jsonl>`。混合成败组进入 GRPO，0/8 交给强教师实跑，8/8 用于回归，异常或不完整样本进入复核。首轮门槛可用 `--min-grpo` 配置。

训练侧使用 OpenEnv 采集模型实际返回的 token ID、logprob 与 loss mask，Harbor 给奖励，TRL 更新模型并同步权重到 vLLM。先跑 `report.py --check-training-captures <batch>` 核对评分和 trial 级采集文件，再核查一次真实更新和 vLLM 的权重版本。将 OpenEnv 和 vLLM 启动后，运行：

```bash
python3 -B train.py --trl-root "$TRL_ROOT" --plan -- \
  --server "$OPENENV_URL" --vllm-url "$VLLM_URL" --model "$TRAIN_MODEL" \
  --split "$TRAIN_SPLIT" --harness opencode --num-generations 8 --max-steps 1
```

SFT 需要源 trial、工具调用到结果的配对关系和对应的实际模型请求。发生上下文压缩时按请求窗口分段，记录目标起始位置。训练入口、采集协议及分段思路见以下来源。[TRL Harbor 示例](https://github.com/huggingface/trl/tree/main/examples/async_grpo_harbor) · [OpenEnv Harbor](https://github.com/openenv/OpenEnv/blob/main/docs/source/environments/harbor.md) · [Roller 计分和轨迹产物](https://github.com/Valen-Team/Roller/blob/main/README_zh-CN.md)
