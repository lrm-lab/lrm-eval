# VLMEvalKit 图文评测

在 `harbor.toml` 用 `runner = "vlmevalkit"` 登记题库。LRM Eval 调用 VLMEvalKit 的题面构建、图像编码和评分器，通过 OpenAI Chat Completions 请求视觉模型。执行机需要 Python 和网关访问，无需 Docker 或 Agent CLI。工具型图文任务可用 Harbor 题包与 harness，成绩按各自的题面和评分协议比较。

## 准备运行环境

已有 VLMEvalKit 环境时，设置 `VLMEVALKIT_ROOT` 为代码目录，`VLMEVALKIT_PYTHON` 为该环境的 Python。也可单独安装，正式跑分时将题库版本和下面的提交一并固定。 [VLMEvalKit 安装说明](https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/Quickstart.md)

```bash
git clone https://github.com/open-compass/VLMEvalKit.git
export VLMEVALKIT_ROOT="$PWD/VLMEvalKit"
git -C "$VLMEVALKIT_ROOT" checkout c247c4d
python3.12 -m venv "$VLMEVALKIT_ROOT/.venv"
export VLMEVALKIT_PYTHON="$VLMEVALKIT_ROOT/.venv/bin/python"
"$VLMEVALKIT_PYTHON" -m pip install -e "$VLMEVALKIT_ROOT"
```

在执行环境注入 `OPENAI_BASE_URL` 和 `OPENAI_API_KEY`，并将 `MODEL_ID` 设为网关上的模型名，例如 `Qwen3_8-27B`。有组织 CA 时设置 `REQUESTS_CA_BUNDLE`。网关地址可带完整的 `/chat/completions` 路径。程序从环境读取凭据，运行时单独准备工具包和缓存目录。

## 注册和运行 CMMMU

将下列内容保存为题库目录下的 `harbor.toml`，原生 TSV 放在同目录。TSV 中保留 VLMEvalKit 的 `index`、图片、题面和评分字段。CMMMU 的短测题包需覆盖原评分器使用的六个学科大类。

```toml
[datasets.cmmmu-val]
name = "CMMMU validation"
runner = "vlmevalkit"
path = "CMMMU_VAL.tsv"

[datasets.cmmmu-val.vlmevalkit]
dataset = "CMMMU_VAL"
dataset_class = "CMMMU"
metric_key = "总准确率"
metric = "CMMMU overall accuracy"
```

```bash
python3.12 framework/run.py cmmmu-val --config "$DATA_DIR/harbor.toml" \
  --all --model "$MODEL_ID" --batch cmmmu-001 --concurrent 3 \
  --work "$PWD/.work" --output "$PWD/harbor-runs"
python3.12 framework/report.py "$PWD/harbor-runs/cmmmu-001" --format markdown
```

`--plan` 检查入口、数据路径和模式。单题联通可用 `--task <样本 index>`，可重复指定。指定样本时只做推理。整批跑分用 `--all`，在题库配置设 `mode = "infer"` 可对整批只做推理。需要 Judge 的数据集可通过 `[datasets.<id>.vlmevalkit.judge]` 指定原生 `model` 及评测参数，Judge 的地址和凭据分别用 `VLMEVALKIT_JUDGE_BASE_URL` 和 `VLMEVALKIT_JUDGE_API_KEY`。

可用 `[datasets.cmmmu-val.vlmevalkit.model_options]` 固定 `max_tokens`、`temperature`、`img_detail` 和网关参数。Qwen 在支持该字段的 vLLM 网关上可设 `chat_template_kwargs = { enable_thinking = false }`。比较跑分时使用相同的生成参数。

本地图片 TSV 按原生数据集类加载，类名可查该版本的 `vlmeval/dataset/__init__.py`。也可以将 `path` 指向准备好的 LMUData 目录，由工具包按 `dataset` 名称构建数据集。目录会复制到作业目录，请为媒体文件留足空间。数据集自有依赖、视频抽帧和需 Judge 的评分应按题库分别验收。

## 产物

| 路径 | 内容 |
|---|---|
| `episodes.jsonl` | 每题一条记录，包含 trial ID、模型、运行状态、原生逐题分可用时的奖励 |
| `trials/<UUID>/result.json` | `vlmeval_result` 中的样本 ID、发送给模型的消息、回答、HTTP 状态 |
| `batch-status.json`、`batch-report.json` | 作业位置、计划数、完成数、失败数和批次得分 |
| `vlmevalkit/metrics.json` | VLMEvalKit 原生评分输出 |
| `official-result.json` | 选择 `metric_key` 时，批次报告读取的原生指标与本批分母 |

原始预测表和执行日志位于 `--work/jobs/<批次>/`。题面含图像时，逐题记录也会保存发送的图像内容。CMMMU 的原生评分只提供汇总，逐题 reward 留空。对原生评分器有明细的题库，可配置 `detail_file`（预测文件同目录下的明细文件名）和 `detail_score_key`，按原生 `index` 导入逐题成绩。有推理失败的批次只发布已完成的记录。多模态模型的问答证据用于复核评测。Agent 工具 ATIF 和在线训练 token 采集走各自的 Harbor 与 OpenEnv 入口。
