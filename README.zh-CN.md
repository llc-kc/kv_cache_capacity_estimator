# KV Cache 容量仿真工具

## 功能

### 1. 离线 token IDs 回放

这是最推荐的使用方式，这个方式具备最小的依赖，不依赖于sglang或者vllm等引擎。而下面的基于openai请求的replay方式则容易在tokenize过程遇到一些问题。

该模式只依赖 Python 标准库，不加载 tokenizer、模型或 chat template。输入是 JSONL，每行可以是以下任一种形式：

```json
[1, 2, 3, 4]
{"input_ids": [1, 2, 3, 4]}
{"token_ids": [1, 2, 3, 4]}
```

运行方式：

```bash
python src/token_ids_replay.py token_ids.jsonl \
  --kv-bytes-per-token 61505 \
  --page-size 64
```

两种回放都会为每个请求直接计算连续前缀命中的容量需求。若无限容量下该请求可连续命中 `H` 个 prefix page，目标系数为 `R`，则要求前 `ceil(H × R)` 个 prefix page 全部命中；所需容量为这些 page 的最大 LRU depth 加一，再乘以每个 page 的字节数。

默认目标为该请求无限容量理论连续 prefix page 命中率的 99%。结果会输出精确容量需求的 mean、p50、p90、p95、p99 和 p999。

```bash
--target-hit-rate-percent 99
```

`--capacities` 各种容量的命中率预估，默认值：`100GiB,200GiB,400GiB,800GiB,1TiB,2TiB,4TiB,6TiB,8TiB,12TiB,16TiB,24TiB,32TiB,64TiB`。显式传入 `--capacities` 可覆盖默认值。

`--warm-up-percent PERCENT` 接受 0 到 100（也可写成 `20%`）。输入开头`floor(请求总数 × PERCENT / 100)` 条请求只用于建立 LRU/cache 状态，不会计入请求数、page 访问数、任何命中率或每请求容量分布。例如：

```bash
--warm-up-percent 20
```

### 2. OpenAI 兼容请求回放

输入为 OpenAI Chat/Completions 兼容的 JSONL。既支持直接请求对象，也支持 OpenAI Batch API 的 `body` 包装格式。工具支持三种 tokenize 后端，得到的 token IDs 都会送入与功能 1 完全相同的分析流程。

`request_replay.py` 默认使用 4 个线程并发 tokenize。使用 `--tokenize-workers N`可以调整线程数，设为 `1` 可恢复串行处理。请求即使在线程中乱序完成，也会严格按JSONL 输入顺序将 token IDs 送入仿真，因而不会改变 LRU 访问轨迹。

#### 本地 SGLang 后端（无需部署，推荐）

`sglang` 后端直接使用 SGLang 的 OpenAI 请求类型、tool-call parser、内容格式转换、chat template 和 tokenizer。它只读取模型 config、tokenizer 和 chat template，既不加载模型权重，也不启动推理服务。

Note: 需要在SGLang 的镜像环境中执行。

以下命令可直接处理仓库中的 GLM-5.2 示例。`--tool-call-parser glm47` 会按 SGLang 的 parser 注册表校验，并在带 tools 的请求中初始化对应 parser：

```bash
python src/request_replay.py \
  test_data/LongBench-v2/data_short.openai.jsonl \
  --tokenize-backend sglang \
  --model test_model/GLM-5.2-FP8 \
  --tool-call-parser glm47 \
  --kv-bytes-per-token 61505 \
  --page-size 64 \
  --capacities 40GiB,80GiB,160GiB,200GiB,250GiB,300GiB,512GiB
```

该后端对 completion prompt 使用 SGLang 加载的 tokenizer；对 chat messages 则复用SGLang 的内容格式规范化和 `apply_chat_template` 路径，包括历史 assistant tool call参数从 JSON 字符串到对象的转换，以及 `chat_template_kwargs`（例如`enable_thinking`）的传递。

**测试过的模型**

- GLM 5.2, token ids与sglang引擎生成的一致。在SGLang v0.5.19版本和fp8_e4m3 kv cache数据类型，不开启MTP时，每个token的kv cache容量为日志打印为60 KiB.

### 3. OpenAI 请求转换为 token IDs

`openai_to_token_ids.py` 复用上述三种 tokenize 后端，将 OpenAI Chat/Completions兼容 JSONL 转为离线 token-ID JSONL。输出每行是一个 token ID 数组，可直接作为`token_ids_replay.py` 的输入。未指定 `--output` 时写到标准输出。

例如使用本地 SGLang源码的tokenizer：

```bash
python src/openai_to_token_ids.py \
  test_data/LongBench-v2/data_short.openai.jsonl \
  --tokenize-backend sglang \
  --model test_model/GLM-5.2-FP8 \
  --tool-call-parser glm47 \
  --output token_ids.jsonl
```



### 4. online KV cache容量分析

第一步：

使用sglang镜像docker pull lmsysorg/sglang:v0.5.20-cu130或者

在sglang镜像中构建和安装whl包，参考下面安装环节。

第二步：

获取sglang适配并安装覆盖原有版本

```
git clone -b kv_capacity_estimator https://github.com/llc-kc/sglang.git
cd sglang
pip uninstall -y sglang
pip install -e "python" 
```



## 指标

```text
Page hit   = 命中的 page 访问数 / 全部 page 访问数
Reuse hit  = 命中的可复用 page 数 / 所有非冷启动 page 访问数
Prefix hit = 各请求从开头连续命中的 page 数之和 / 全部 page 访问数
```



## 安装与构建 wheel

项目使用标准的 `pyproject.toml` 构建配置，要求 Python 3.10 或更高版本。
从源码安装：

```bash
pip install -e ./
```

构建 wheel：

```bash
python -m pip install build
python -m build --wheel
```

构建产物位于 `dist/` 目录，可通过下面的命令安装：

```bash
python -m pip install dist/kv_cache_capacity_simulator-0.1.1-py3-none-any.whl
```

安装后提供以下命令行入口，它们分别对应原先 `src/` 下的三个脚本：

```text
kv-cache-token-replay
kv-cache-request-replay
kv-cache-openai-to-token-ids
```

例如，离线 token IDs 回放也可以写成：

```bash
kv-cache-token-replay requests.jsonl \
  --kv-bytes-per-token 61505 \
  --page-size 64 \
  --capacities 10GiB,100GiB,1TiB
```

基础 wheel 只依赖 Python 标准库。使用本地 `sglang` 或 `vllm` tokenize 后端时，仍需在该引擎的docker运行环境中执行。
