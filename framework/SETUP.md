# 部署与运行

## 1. 准备执行机

图文 API 评测可以单独安装 VLMEvalKit。 [图文评测安装和短测](VLMEVALKIT.md)

使用 Linux x86_64、Python 3.12+、可访问 Docker 的运行账号，以及 Docker Compose 和 Buildx。安装交接包还需 `tar`、`zstd`、`flock` 和可写的本地缓存。验证当前账号能访问 Docker 并检查磁盘空间。 [Docker Engine 安装](https://docs.docker.com/engine/install/) · [uv 安装](https://docs.astral.sh/uv/getting-started/installation/)

```bash
python3 --version
docker info
docker compose version
docker buildx version
```

## 2. 选择代码和题包位置

在团队共享盘上可以直接进入 `framework/`。脚本默认从同级的 `../bundles/` 读交接包，在运行账号自己的 `~/.cache/tianxi-harbor/bundle` 解包。运行结果默认写到 `../harbor-runs/`，可用 `--output` 指定个人目录。共享目录的站点路径及历史结果位置见该目录的 `README.md`。

从 GitHub 克隆后，进入 `lrm-eval/framework`。如果只想用自定义 Harbor 或 `direct` 题库，按项目首页生成一题即可。未安装交接包时，Harbor 运行依赖 `uvx` 拉取固定版本 Harbor 0.26.0 和 LiteLLM 1.110.1。`direct` 只需 Python、题库及网关。 [仓库快速开始](../README.md#自定义数据集和快速开始) · [题库交付格式](DATASET_FORMAT.md)

四套预置 Agent 题库需要单独分发的交接包。将 `TIANXI_BENCH_ASSETS` 设为可读的 `bundles/` 绝对路径。若执行机没有共享盘，复制下面所需文件到执行机，再将该变量指向复制后的目录。Docker 镜像可从批准的镜像源拉取，或用 `docker image load -i <镜像归档>` 导入。

| 文件 | 用途 |
|---|---|
| `pi-openclaw-20261120-core.tar` | 固定 Harbor、Terminal-Bench 2.1、DeepSWE 1.1 和 Pi/OpenClaw 安装缓存 |
| `gaia-validation-20261124.tar` | GAIA validation 的 165 道原始 Harbor 题 |
| `wildclawbench-20261120-tasks.tar.zst` | 可选，WildClawBench 60 道题 |
| `opencode-linux-x64-1.19.3.tgz` | 可选，固定 OpenCode 二进制 |
| `wildclawbench-ubuntu_v1.3.tar` | 可选，另行分发的 WildClaw 官方任务镜像 |

```bash
bash setup.sh                    # Terminal、GAIA、DeepSWE
bash setup.sh --opencode         # 再准备 OpenCode 固定二进制
bash setup.sh --wildclaw         # 再解包 WildClaw 题库
# 若镜像还未导入，使用站点提供的镜像归档绝对路径
# docker image load -i "$WILDCLAW_IMAGE_TAR"
```

也可设置 `TIANXI_BENCH_BUNDLE` 指定自有缓存，或用 `run.py --bundle` 复用已解包并安装的目录。WildClaw 题库解包和行尾规范化约需 16 GB 本地空间，另预留官方镜像归档、Docker 镜像层和试验目录的空间。WildClaw 需要搜索或 Judge 的题目按原评分协议配置相应服务。

## 3. 配置网关并跑一题

模型名用网关可识别的实际 ID。Pi、OpenClaw 和 OpenCode 可接 OpenAI Chat Completions，Codex 需要 OpenAI Responses。Claude Code 需要 Anthropic Messages 和固定的 CLI 版本。前两种协议通常用 `OPENAI_BASE_URL` 和 `OPENAI_API_KEY`，Messages 用 `ANTHROPIC_BASE_URL` 和 `ANTHROPIC_AUTH_TOKEN` 或 `ANTHROPIC_API_KEY`。在授权的运行环境注入凭据。下例假设已完成交接包安装并配置了 Qwen3.8-27B 网关。

```bash
export EVAL_OUT="$PWD/../harbor-runs"
export QWEN_MODEL=openai/Qwen3_8-27B

python3 -B run.py terminal --task nginx-request-logging \
  --agent pi --agent-version 0.87.1 --model "$QWEN_MODEL" \
  --api-format openai_chat_completions --batch terminal-pi-smoke --output "$EVAL_OUT"
python3 -B report.py "$EVAL_OUT/terminal-pi-smoke" --check-smoke --format markdown
```

| 题库 | 单题 `--task` | 提示 |
|---|---|---|
| Terminal-Bench 2.1 | `nginx-request-logging` | 评分步骤需要下载 `uv` 和 Python 依赖 |
| GAIA validation | `076c8171-9b3b-49b9-a477-244d2a532826` | 读取随题表格，用 `--api-format openai_chat_completions` |
| WildClawBench | `01_Productivity_Flow_task_6_calendar_scheduling` | 依赖 `wildclawbench-ubuntu:v1.3`，评分使用本地文件 |
| DeepSWE 1.1 | `clack-async-autocomplete-options` | 构建及测试耗时，先用 `--plan` 检查配置 |

将命令中的数据集 ID、任务 ID 和批次名一起替换。`--plan` 可检查生成的 Harbor 命令。`--work` 指定存放原始日志的本地目录。真实短测通过后可以改为多个 `--task` 或 `--all`，再按需指定 `--attempts` 和 `--concurrent`。OpenClaw 2026.11.5 搭配 Qwen3.8-27B 时可用 `--thinking off`，默认的 `medium` 会被该 CLI 拒绝。OpenCode 的 Qwen Chat Completions 路由用 `vllm/Qwen3_8-27B`。

### 同一道题切换 harness

先按项目首页生成 `demo-v1`。下表的路由以 Qwen3.8-27B 为例。跨网关换模型时改用实际注册的 ID，并新建批次。

| Harness | 传给 `run.py` 的参数 |
|---|---|
| Pi | `--agent pi --agent-version 0.87.1 --model openai/Qwen3_8-27B --api-format openai_chat_completions` |
| OpenClaw | `--agent openclaw --agent-version 2026.11.5 --thinking off --model openai/Qwen3_8-27B --api-format openai_chat_completions` |
| OpenCode | `--agent opencode --agent-version 1.19.3 --model vllm/Qwen3_8-27B --api-format openai_chat_completions` |
| Codex | `--agent codex --agent-version 0.171.0 --model openai/Qwen3_8-27B --api-format openai_responses` |
| Claude Code | `--agent claude-code --agent-version 2.1.91 --model anthropic/Qwen3_8-27B --api-format anthropic_messages`，需提供能转发该模型的 Messages 地址 |

```bash
python3 -B run.py demo-v1-train --config ../datasets/demo-v1/harbor.toml \
  --task log-rollup-train-001 --agent codex --agent-version 0.171.0 \
  --model openai/Qwen3_8-27B --api-format openai_responses \
  --batch demo-codex --output "$EVAL_OUT"
python3 -B report.py "$EVAL_OUT/demo-codex" --check-smoke --format markdown
```

预装 CLI 的执行机也可用 `--mounts` 将已核对版本的 Linux x86_64 可执行文件只读挂载进沙箱。例如将 Codex 挂到 `/usr/local/bin/codex`，`--agent-version` 使用该二进制报告的版本。Docker 内 CLI 的 ABI 与镜像应兼容。

## 4. 排查网络和结果

镜像拉取、Dockerfile 构建、Agent 请求和 verifier 下载使用各自的网络设置。执行机能访问网关时，再核对容器内访问。安装依赖需要代理时，将站点批准的代理注入对应进程。Docker 镜像可先经批准的镜像源获取并预导入。`docker build` 中的 apt/pip 网络使用 Docker 客户端的构建代理配置。参考 Docker 文档的构建代理和 daemon 代理说明。 [Docker 客户端和构建代理](https://docs.docker.com/engine/cli/proxy/) · [Docker daemon 代理](https://docs.docker.com/engine/daemon/proxy/)

以下参数加到实际的 `run.py` 命令中，引用已配置的环境变量。`NO_PROXY` 保留容器内回环地址。若网关使用组织 CA，另加 `--agent-ca-bundle "$ORG_CA_PEM"`。评分器如果也需要组织 CA，可按任务环境配置证书，由该评分脚本实际使用。

```bash
--agent-env 'HTTPS_PROXY=${HTTPS_PROXY}' --agent-env 'https_proxy=${https_proxy}' \
--agent-env 'NO_PROXY=${NO_PROXY}' --agent-env 'no_proxy=${no_proxy}' \
--agent-env NODE_USE_ENV_PROXY=1 \
--verifier-env 'HTTPS_PROXY=${HTTPS_PROXY}' --verifier-env 'https_proxy=${https_proxy}' \
--verifier-env 'NO_PROXY=${NO_PROXY}' --verifier-env 'no_proxy=${no_proxy}'
```

运行成功后，`batch-report.json` 给出有效得分和覆盖率，`episodes.jsonl` 索引每次 trial，`trials/<UUID>/trajectory.json` 保留可用 ATIF。`report.py --check-smoke` 同时检查评分和配对的工具调用。`failure.raw_log` 指向执行机原始日志，`harbor view <原始作业目录>` 可浏览 Harbor 原生作业。


## 网关采集短测

`--gateway-capture` 启动本机 OpenEnv 采集代理。Harbor 启动原生 harness，代理接收它的模型请求，再转发给 `OPENAI_BASE_URL`。上游凭据由 `OPENAI_API_KEY` 注入，沙箱使用独立采集会话。当前入口使用 OpenEnv 0.14.0 和 Harbor 0.26.0。

在具备 Docker 权限的执行账号中准备 Python 3.12+ 环境。若已有安装这些包的环境，可直接运行评测入口。

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install 'openenv==0.14.0' 'harbor==0.26.0' 'litellm==1.110.1'
```

配置 `OPENAI_BASE_URL`、`OPENAI_API_KEY`，以及执行机访问上游所需的代理和信任证书。将 `--capture-host` 设为运行机上 Docker 容器可直连的 IPv4 地址，示例使用 Linux 的 `172.17.0.1`。先按项目首页生成一个可在 Docker 中运行的日志短题。任务镜像可以预装已固定版本的 harness，或让 Harbor 按其默认安装流程准备。

```bash
python -B run.py demo-v1-train --config ../datasets/demo-v1/harbor.toml \
  --task log-rollup-train-001 --agent opencode --agent-version 1.19.3 \
  --model vllm/Qwen3_8-27B --gateway-capture --capture-host 172.17.0.1 \
  --batch demo-gateway --output ../harbor-runs

python -B report.py ../harbor-runs/demo-gateway --check-gateway-capture --format markdown
```

输出的 `episodes.jsonl` 用同一 `trial_id` 关联 `result.json`、可用 ATIF `trajectory.json` 和 `model-gateway.json`。网关文件存有模型请求窗口、工具定义、响应、采样参数和分支标记。真实工具往返通过匹配模型调用 ID 与后续请求中的工具返回来验证。选择适合存放完整请求内容的输出目录。

该模式使用 OpenEnv 的协议及 harness 配置，支持 `--task`、`--all`、`--attempts`、`--concurrent`。可用 `--pause-after` 在连续出现同类故障时暂停派发。原生 Harbor 批次的 `--resume`、Agent 和 verifier 环境覆盖及绑定文件参数仍在原入口使用。网关评测产物标记为 `purpose=eval`。在线训练使用 OpenEnv 训练采集、vLLM 的实际 token 和 logprob，再由 TRL 消费经过验证的 mask。 [OpenEnv Harbor 采集](https://github.com/openenv/OpenEnv/blob/main/docs/source/environments/harbor.md#the-intercept)
