# LRM Eval

LRM Eval 是一套可扩展的 Agent Infra。**合成数据方法、Agent Harness、模型与网关、训练后端**均可按需扩展。平台通过 Harbor、Direct 和 VLMEvalKit 三种评测入口执行任务，统一记录评分、轨迹与批次报告；可用不同 Harness、模型和网关进行对比。合成任务通过验收后可进入在线训练，更新后的模型再用留出题和基准题评测。

![LRM Eval 可扩展架构：合成数据、Agent Harness、模型与网关、训练后端，以及三种评测执行路径与反馈闭环](assets/lrm-eval-architecture.svg)

| 模块 | 接入方式 |
|---|---|
| Harness | `--agent` 选择 Harbor 原生 harness 或 Python 适配器。数据集可绑定专用适配器。 |
| 模型 | `--model` 选择模型路由。OPD 可以为不同环境和任务指定专家，把能力蒸馏到同一个学生。 |
| 模型网关 | `--base-url` 和 `--api-format` 接入模型服务。`--gateway-capture` 用 OpenEnv 记录原生 harness 的请求窗口和工具往返。 |
| 数据集 | 在 `harbor.toml` 登记题库。工具和沙箱任务走 Harbor，自定义文本及图文题可走 `direct`，原生图文题库可走 `vlmevalkit`。各入口按 trial 导出结果。 |
| 数据合成 | `synthesize.py` 提供五种生成方法，支持注册生成器、题包写入器、验收器和独立验算器。 |
| 训练 | `train.py` 支持 TRL GRPO 和多专家 OPD。通过配置接入其他训练后端，并替换路由、采样、评分和优化等组件。 |

## 数据集

| Agent 题库 | 数据集 ID | 题包及验证状态 |
|---|---|---|
| DeepSWE 1.1 | `deepswe` | 113 道 Harbor 题，已有 Pi、OpenClaw 历史实跑，本轮跳过复测 |
| Terminal-Bench 2.1 | `terminal` | 89 道 Harbor 题，Pi × Qwen3.8-27B 单题已通过评分及 ATIF 检查 |
| GAIA validation | `gaia` | 165 道 Harbor 题，Pi × Qwen3.8-27B 单题已通过评分及 ATIF 检查，也有 Pi、OpenClaw 历史实跑 |
| WildClawBench | `wildclaw` | 60 道 Harbor 题，Pi × Qwen3.8-27B 已完成单题评分和 ATIF 检查，保留原评分器和交接包适配器 |

这四套题库由交接包提供。安装、数据位置和单题命令见 [部署与运行](framework/SETUP.md)。单题冒烟用于验证当前执行链路，完整跑分使用各题库的官方评分器、题库版本和分母。

| 计划接入 | 数据集 | 当前状态 |
|---|---|---|
| 通用能力 | MMLU-Pro、HLE、IFBench、BFCL-v4、Multi-LCB、Global-MMLU-Lite、AA-LCR v1.1 | 待登记题库版本和适配评分。BFCL-v4 还需工具调用交互，Multi-LCB 需代码执行环境及测试 |

多模态入口 `vlmevalkit` 复用原生图文题面和评分，直连支持图像的模型 API。逐题图文问答与 Harbor Agent 轨迹共用 `ATIF-v1.7`，由 `episodes.jsonl` 索引。CMMMU 可按原生题包执行，MMMU-Pro (10c) 和 MM-IFEval 的运行条件见安装指南。 [VLMEvalKit 图文评测](framework/VLMEVALKIT.md)

已有 `direct` 入口支持自定义文本和图文 JSONL、精确匹配及 Python 评分器。每个具名 benchmark 的加载、交互与计分仍需按其协议实现和实跑验证。

## 模型和 harness

`--model provider/model-id` 选择模型，`--base-url` 和 `--api-format` 选择网关协议。凭据由执行环境注入。每换一个模型、harness、网关或工具镜像，先跑一道会调用工具的短题，并检查评分和 ATIF。

| Harness | 入口及协议 |
|---|---|
| Pi | `--agent pi`，支持 Chat Completions。Terminal 和 WildClaw 使用交接包适配器 |
| OpenClaw | `--agent openclaw`，支持 Chat Completions。Qwen3.8-27B 合成短题已完成评分，当前版本的 ATIF 工具轨迹待补 |
| OpenCode | `--agent opencode`，已验收版本为 `1.19.3`，Qwen 的 Chat Completions 示例使用 `vllm/Qwen3_8-27B` |
| Codex | `--agent codex`，使用 OpenAI Responses，Qwen3.8-27B 合成短题已完成评分和 ATIF 冒烟 |
| Claude Code | `--agent claude-code`，指定版本并连接 Anthropic Messages。Qwen3.8-27B 的合成短题已通过临时兼容层冒烟 |
| 自定义 | `--agent` 接受 Harbor 原生名称或 Python 导入路径，也可在数据集配置中绑定适配器 |

## 自定义数据集和快速开始

手写题按 [任务格式](framework/DATASET_FORMAT.md) 提供说明、环境、参考解和评分器，在自己的 `harbor.toml` 注册。合成器可按固定种子生成订单对账、离线日志汇总、图工具、自提问和多跳题，支持注册新方法以及可选教师。新题可单题运行、重复指定 `--task`，也可用 `--all` 跑整个数据集。

Linux 执行机准备好 Python 3.12+、Docker、Compose、Buildx 和 `uvx` 后，可以从空目录生成一题并用参考解检查 Docker 和评分。镜像拉取、共享盘运行、网关和代理配置见 [部署与运行](framework/SETUP.md)。这段命令只需要代码和基础 Docker 镜像。

```bash
git clone https://github.com/lrm-lab/lrm-eval.git
cd lrm-eval/framework
python3 -B synthesize.py generate demo-v1 --method log-rollup \
  --train-per-method 1 --holdout-per-method 0
python3 -B synthesize.py check ../datasets/demo-v1
python3 -B run.py demo-v1-train --config ../datasets/demo-v1/harbor.toml \
  --task log-rollup-train-001 --agent oracle --batch demo-oracle \
  --output ../harbor-runs
python3 -B report.py ../harbor-runs/demo-oracle --format markdown
```

设置 `OPENAI_BASE_URL`、`OPENAI_API_KEY` 和与网关匹配的 `MODEL_ROUTE` 后，同一题可换成真实模型。用 `--output` 指定任意可写的结果目录。

```bash
python3 -B run.py demo-v1-train --config ../datasets/demo-v1/harbor.toml \
  --task log-rollup-train-001 --agent opencode --model "$MODEL_ROUTE" \
  --api-format openai_chat_completions --batch demo-smoke --output ../harbor-runs
python3 -B report.py ../harbor-runs/demo-smoke --check-smoke --format markdown
```

## 网关采集

给 Harbor 评测加 `--gateway-capture`，OpenEnv 会用同一个代理采集原生 harness 的模型交互。`episodes.jsonl` 将每次试验的网关请求窗口、评分和可用 ATIF 关联到同一个 trial。用 `report.py --check-gateway-capture` 核对评分和实际工具往返。 [安装和短测命令](framework/SETUP.md#网关采集短测)

## 计分和运行记录

每批有 `episodes.jsonl`、`batch-report.json`、`batch-status.json` 及按 trial UUID 索引的评分和可用 ATIF。报告包含计划、已结束、有效评分、有效样本均分、通过数、覆盖率和故障分类。计划 8 次、5 次有效且全通过时，报告分别显示 **5/5 通过** 和 **5/8 覆盖率**。历史批次缺少计划时可传 `--planned`，逐题补分母可用 `--attempts-per-task`。

在官方评分器完成计分后，用 `report.py --official-result <official.json>` 记录其正式分数及分母。文件字段为 `dataset`、`metric`、`score`、`denominator` 和 `source`。有效均分用于分析本批有效样本。浏览原生会话时，运行 `harbor view <原始作业目录>`。

运行故障用固定摘要和执行机日志位置标识。默认连续三次同类故障会暂停派发，阈值用 `--pause-after` 调整。原配置下给原命令追加 `--resume`，由 Harbor 接着执行未完成的试验。框架会先核对题库、模型、harness、参数及原生配置锁。可复用 `--export-job` 导入已有 Harbor 作业。

## 合成和训练

`synthesize.py` 注册了 5 种生成方法。默认每种生成 20 道训练候选题和 6 道留出题，共 130 题。已有的 `tianxi-synth-v1` 题包包含前两种方法的 52 道题。每种方法可单独调用，多次传入 `--method` 会将选中的方法写入同一题包。离线生成使用固定种子，训练和留出分别采样。题目遵循 [任务格式](framework/DATASET_FORMAT.md)。

| `--method` | 生成方式 |
| --- | --- |
| `order-reconcile` | 订单对账，覆盖事件过滤、去重和金额结算。 |
| `log-rollup` | 离线日志汇总，覆盖分片、去重、时区和分位数。 |
| `toolspace` | 从工具图生成查找、交接、版本过期后恢复三类任务。目录记录保存在独立的 Docker 服务，CLI 逐条查询。评分同时核对证据路径、服务端提交状态和回执。 [ToolSpace 项目](https://toolspace-agent.github.io/) |
| `self-questioning` | 从材料提出唯一匹配、最早、最晚三类问题。每个条件都要影响结果，独立 SQL 验算从公开题面重算答案。支持离线出题、模型出题和已接受提案回放。 |
| `multi-hop` | 生成直链、双条件汇合和时点归属三类证据题。每对只改一个事实，正确联系人随之变化。评分核对答案和来源记录，独立验算逐条移除关键事实检查必要性。 |

```bash
# 不依赖外部模型的可复现版本
python3 -B synthesize.py generate my-synth-v1
python3 -B synthesize.py check ../datasets/my-synth-v1

# 可选教师。先在运行环境注入 TIANXI_SYNTH_TEACHER_API_KEY。
python3 -B synthesize.py generate my-synth-teacher-v1 --teacher

# 生成单题。也可重复传 --method，或调整每类题量和随机种子。
python3 -B synthesize.py generate my-one-task-v1 --method log-rollup \
  --train-per-method 1 --holdout-per-method 0 --seed 42

python3 -B synthesize.py generate my-reasoning-v1 \
  --method toolspace --method self-questioning --method multi-hop
python3 -B synthesize.py check ../datasets/my-reasoning-v1
```

教师默认走智增增的 `deepseek-v4.1-flash`，每个方法和 split 生成一份题面背景、边界提示及示范思路。教师只收到合成素材的少量样例和评分约定。背景用于题面，边界提示和示范保存在题包外层的 `synthesis.json`，供人工审阅。程序负责标准产物、验收和评分。教师简报与后续强模型在 Harbor 里完成的 SFT 候选轨迹分别记录。

新方法在 `harbor.toml` 的 `[synthesis.methods]` 登记为 `module:function`。函数接收 `(split, index, seed)`，返回 `synthesis_methods.Case`，包含输入文件、题目规则、CSV 或 JSON 预期结果、典型错误、Python 参考解、题面背景和可选的 `environment` 标签。参考解默认在 `/app` 工作，本地验收时接收等价工作目录作为第一个参数。[synthesis.auditors] 接收输入目录并独立重算答案。需要自定义镜像、工具服务或评分格式时，按方法在 `[synthesis.writers]` 注册 `(folder, method, split, case, scenario)`，在 `[synthesis.checkers]` 注册 `(task_dir, scratch_dir, auditor)`，由这两个插件负责题包写入和验收。生成器写出的配置可直接传给 `run.py --config <题包>/harbor.toml`。手写 Harbor 题也可按 [任务格式](framework/DATASET_FORMAT.md) 登记。

`[synthesis.options.<方法名>]` 可给生成函数传关键字参数。`[synthesis.options.toolspace]` 的 `graph_file` 可指向自定义 JSON，以 `train` 和 `holdout` 为顶层键。每个 split 下按 `工具名 -> {关系名: 下一个工具名}` 定义有向无环图，第一个工具为入口，末端工具用空对象表示。可采样的末端至少经过三条边，且从入口只有一条路径。路径参数可用绝对路径或相对于命令运行目录的路径。

团队可复制随仓库提供的 `framework/harbor.toml`，在已有表内添加注册项。例如，给 ToolSpace 的新工具图注册名称 `warehouse-handoff`，在 `synthesis.methods`、`synthesis.writers`、`synthesis.checkers` 和 `synthesis.auditors` 中分别沿用 `toolspace` 的对应函数，再在 `[synthesis.options.warehouse-handoff]` 设置 `graph_file`。用 `--config team.toml --method warehouse-handoff` 可单独生成，验题时也传入同一份配置。全新方法可以放在自己的可导入模块里；返回标准 `Case` 的文件题仅需在 `synthesis.methods` 添加一个名称和函数入口。

分别试一种方法时，可在 `framework/` 运行下面的命令。训练和留出各生成 6 道，覆盖三种图结构及其配对版本。相同方法、配置和种子单独生成或混合生成会得到相同的离线题目。

```bash
python3 -B synthesize.py generate graph-pilot --method multi-hop --seed 42 \
  --train-per-method 6 --holdout-per-method 6
python3 -B synthesize.py check ../datasets/graph-pilot
```

自提问默认自带材料。自己的 `materials_file` 使用 JSON 对象，`train` 和 `holdout` 对应文档数组，每行有唯一 `id`、`owner`、`team`、`topic`、`stage` 和整数时间序号 `ordinal`。可将某类错例转成 `focus = "earliest"` 或 `"latest"`。配置模型出题时，在自有配置文件中加入以下条目，模型基于材料给出条件和证据 ID，框架检查证据、唯一性及每个条件的必要性，并由校验过的条件生成题面。最多重新提案两次。此选项独立于生成背景介绍的 `--teacher`。

```toml
[synthesis.options.self-questioning.proposer]
base_url = "https://your-model-service.example/v1"
model = "your-model"
api_key_env = "QUESTION_PROPOSER_KEY"
```

已接受的结构化提案写入题包根目录的 `synthesis.json`。使用相同方法名称、材料、种子和题目数量，设置 `[synthesis.options.self-questioning]` 的 `replay_manifest` 指向此文件，可以离线重放。自定义方法若要生成配对题，可在 `[synthesis.seed_groups]` 将方法名配置为 `2`。同一 split 中相邻两题共享随机种子，生成函数用 `index` 决定每题的变化。训练和留出的种子独立。

训练候选用同一模型和同一 OpenCode 版本每题采样 8 次，随后运行 `synthesize.py select <episodes.jsonl>`。混合成败组进入 GRPO，0/8 交给强教师实跑，8/8 用于回归，异常或不完整样本进入复核。首轮门槛可用 `--min-grpo` 配置。

GRPO 使用 OpenEnv 采集实际 token ID、logprob 和 loss mask。项目默认固定 TRL 提交 `b2686bf5c300756cdf878b84f83a48f794e2fb94`，按 verifier 原始正确性训练。先在独立的 TRL checkout 安装该版本及其 Harbor 示例依赖，启动 OpenEnv 和 vLLM。用 `report.py --check-training-captures <batch>` 可核查已有 trial 的训练采集。

```bash
python3 -B train.py --trl-root "$TRL_ROOT" --plan -- \
  --server "$OPENENV_URL" --vllm-url "$VLLM_URL" --model "$TRAIN_MODEL" \
  --split "$TRAIN_SPLIT" --harness opencode --num-generations 8 \
  --max-staleness 0 --save-steps 1 --max-steps 2 --output-dir "$EXPERT_OUTPUT"
```

去掉 `--plan` 启动训练。做全权重 GRPO 的一步发布验收时，设 `--max-steps 1` 并在 `--` 前加 `--verify-grpo-update <logprob容差>`。训练返回后，入口通过同一 vLLM 实例和 harness 再跑一次所选 split 中的题，将真实采样的 token 在 CPU 上分别用初始权重和最终权重复算。验收要求所有受训 token 与最终权重吻合，至少一个足以区分初始权重，并将结果写入 `grpo-policy-*.json`。用小模型并按相同权重跨引擎的误差校准容差。

`$EXPERT_OUTPUT` 使用绝对路径。要分别训练工具和推理专家，用各自的训练 split 和输出目录各跑一次，再将通过本域留出题评测的权重填入下面的 `opd.experts`。按领域保留底座、对应专家和最终学生在相同留出题上的通过率。两步用于检查训练和发布链路，正式训练可调整步数。

每次运行的 `grpo-rewards-*.jsonl` 按 TRL rollout ID 和组 ID 记录到达评分回调的样本，包括 verifier 原始分、工具调用次数、效率分、实际加分及最终训练分。未评分的样本保留空值。需要沿用上游满分后的效率加分时，在 `--` 前传 `--tool-efficiency-weight 0.3`。使用其他兼容 TRL 版本时，在同一位置传 `--trl-revision <完整提交 SHA>`。启动器检查 checkout 并从中加载 TRL。[固定版本的 Harbor 示例](https://github.com/huggingface/trl/blob/b2686bf5c300756cdf878b84f83a48f794e2fb94/examples/async_grpo_harbor/async_grpo_harbor.py)

SFT 需要源 trial、工具调用到结果的配对关系和对应的实际模型请求。发生上下文压缩时按请求窗口分段，记录目标起始位置。训练入口、采集协议及分段思路见以下来源。[TRL Harbor 示例](https://github.com/huggingface/trl/tree/main/examples/async_grpo_harbor) · [OpenEnv Harbor](https://github.com/openenv/OpenEnv/blob/main/docs/source/environments/harbor.md) · [Roller 计分和轨迹产物](https://github.com/Valen-Team/Roller/blob/main/README_zh-CN.md)

### 多专家 OPD

`train.py --backend opd` 给每个训练任务选定一个专家。每步先发布当前学生 checkpoint，由学生在 OpenEnv Harbor 里实跑，再用对应专家给学生实际采到的 token 评分。不同专家的分数进入同一次学生更新。更新使用被 loss mask 选中的 token 估计 reverse KL。Harbor reward 随轨迹记录，答错的有效轨迹也可以参与蒸馏。下一步从更新后的 checkpoint 重新采样。 [多专家 OPD 参考](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/distillation/)

如果所有专家都由同一个底座微调而来，并且有完整的本地 Hugging Face 权重，可以用专家权重初始化学生，再开始 OPD。在 `framework/` 下执行：

```bash
python3 -B prepare_opd_student.py --base /models/base \
  --expert /models/tools-expert --expert /models/reasoning-expert \
  --output /models/student-init
```

把生成的目录传给训练入口的 `--student /models/student-init`。初始参数为 50% 底座加 50% 专家平均值，两个专家各占 25%。准备过程在 CPU 上逐个读取专家，用 FP32 累加后恢复底座参数的数据类型，内存需容纳一份 FP32 底座及一个专家。输出保留底座的配置、tokenizer 和缓冲区，`opd-init.json` 保存输入路径和权重占比。可以用同一批任务和多个种子对比从底座和从混合权重开始训练的每域成绩。

本地专家还可以在各自的 `[opd.experts.<名称>]` 下设置 `scorer = "training_opd_models:FullDistributionExpert"`。学生仍然自己采样。对于 loss mask 选中的每个位置，训练器使用专家对整个词表的概率计算条件 KL(student ‖ expert)，并在 `opd.json` 中记录 `exact_reverse_kl`。默认 scorer 只取实际采到的 token 的分数。完整概率需要额外内存，规模随该轮受训 token 数和词表大小增长。远端 vLLM 专家仍可使用现有的单 token scorer。

在主配置的 `[training.opd]` 下配置，或通过 `--opd-config` 提供顶层 `[opd]` 配置。下面的模型路径和 OpenEnv split 名需按实际部署填写。`[opd.splits]` 的值取自 OpenEnv `/harbor_env/splits`。同一批次可重复传 `--dataset` 混合环境，或用 `--task` 选部分任务。

```toml
[opd]
temperature = 1.0

[opd.splits]
my-reasoning-v1-train = "/srv/tianxi/datasets/my-reasoning-v1/train"

[opd.experts.tools]
model = "/models/tools-expert"
device = "cuda:1"

[opd.experts.reasoning]
model = "/models/reasoning-expert"
device = "cuda:1"

[[opd.routes]]
environment = "tool-catalog"
expert = "tools"

[[opd.routes]]
environment = "record-catalog"
expert = "reasoning"

[[opd.routes]]
environment = "fact-shards"
expert = "reasoning"

[[opd.routes]]
dataset = "my-reasoning-v1-train"
task = "multi-hop-train-001"
expert = "tools"

[opd.openenv]
url = "http://127.0.0.1:8200"
harness = "opencode"
sandbox = "docker"

[opd.student]
device = "cuda:0"

[opd.vllm]
cuda_visible_devices = "2"
args = ["--enable-auto-tool-choice", "--tool-call-parser", "hermes"]
```

规则可按 `dataset`、`environment`、`source` 和 `task` 精确匹配。先按 `priority` 选优先级，再按匹配字段数选择更具体的规则。缺少匹配或最高等级指向多个专家时，计划会报错。任务必须通过任务元数据或数据集配置标记 `split = "train"`。生成的训练题已带此字段；旧版订单和日志题会归入 `offline-files`。

```bash
python3 -B train.py --config ../datasets/my-reasoning-v1/harbor.toml \
  --backend opd --plan -- --dataset my-reasoning-v1-train --opd-config ./opd.toml

python3 -B train.py --config ../datasets/my-reasoning-v1/harbor.toml \
  --backend opd -- --dataset my-reasoning-v1-train --opd-config ./opd.toml \
  --student /models/student --output ../training-runs/opd-smoke --steps 2 --verify-update
```

默认后端使用单设备 Transformers 学生训练器和本地 Hugging Face 专家。专家需与学生使用相同的 token ID 映射和输出词表。远端 vLLM 专家可配置 `scorer = "training_opd_models:VLLMExpert"`，并提供 `url`（含 `/v1`）、`model`、`tokenizer` 和可选的 `api_key_env`。每步默认按专家轮转采样，每位被选中的专家各贡献一条轨迹。`--rollouts-per-step` 可改变批量大小。输出目录保存 `routing.json`、学生 checkpoint 及每步的 `opd.json`，记录该步的采样 checkpoint、任务、专家、奖励和 token 数。

两步验收可加 `--verify-update`。第 2 轮起，内置优化器在同一设备额外加载上一版学生，检查至少一个参数确实变化，并用两版权重在每条真实采样轨迹上重算相同、未屏蔽 token 的 logprob。当前版的所有误差必须在 `[opd.student].logprob_tolerance` 内，每条轨迹至少一个 token 的旧版误差必须超过该容差。同一权重跨引擎的数值误差可先用第 1 轮校准。两版都能解释采样时验收会报告证据不足。每步 `opd.json` 保存本轮请求的模型 ID、OpenEnv trial 和 session、按轨迹计算的误差，以及通过验收时对比的 checkpoint 和变化参数名。普通训练可以省略此选项以节省显存。

训练环境需有 PyTorch、Transformers、支持 `purpose="train"` 和 `to_training_trace` 的 OpenEnv Harbor，以及启用 processed logprobs 的 vLLM。默认 publisher 每步重启一次 vLLM 并使用新的模型 ID，适合小规模验收。OpenEnv 必须能访问 publisher 提供的 URL，可通过 `[opd.vllm]` 的 `host`、`port` 和 `url` 调整。`hermes` 是示例解析器，实际配置需匹配学生模型。 [vLLM 服务参数](https://docs.vllm.ai/en/stable/cli/serve/)

`[opd.components]` 可用 `module:function` 替换 `router(task, settings)` 和 `scheduler(routed, seed)`，以及用返回组件实例的 `(args, settings)` 工厂替换 `publisher`、`rollout` 和 `optimizer`。Publisher 提供 `serve(checkpoint)` 上下文，rollout 提供 `sample(task, checkpoint, endpoint)`，优化器提供 `save`、`begin`、`accumulate` 和 `step`。每位专家可单独配置 scorer，工厂接收 `(definition, args, settings, optimizer)`，返回对象通过 `score(turn)` 给未屏蔽的学生 token 打分。远程权重服务可用 `training_opd_openenv:CommandPublisher`，其命令从 `[opd.publisher]` 读取，末尾收到 checkpoint 路径，stdout 返回包含 `checkpoint`、`model`、`url` 的 JSON。新的训练后端可在 `[training.backends]` 注册，接受 `(config, config_dir, trainer_args, plan=...)`。

本地验收覆盖全部 130 道合成题。两个小型专家在 CPU 上完成两次学生权重更新，采样适配器每步读取最新 checkpoint。GPU、Docker 和 vLLM 的端到端训练仍待验证。
