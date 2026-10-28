# TracePack

**中文** | [English](README.en.md)

TracePack 会在 `/compact` 后恢复用户原话和工具输出。

Claude Code 压缩长会话（compaction）时，会在摘要里保留计划，但可能省掉一些细节：报错原文、创建过的路径、基准测试的数字，或是你两小时前提到的端口。TracePack 会在每次压缩后，按预算选取用户原话和工具输出放回上下文，并标明出处。整个过程在本地完成，不调用模型，也不向外发送数据。

<img src="assets/demo.gif" alt="两个会话并排同步播放：左边是原版 Claude Code，右边安装了 TracePack。两边运行同一个脚本，自动压缩阈值均设为 100k token。先在聊天中提供一个事实，再打印 120 个随机 key，随后读取两份较大的服务器日志，触发四次自动压缩。左边无法根据当前上下文答出 key087，需要搜索会话记录才能找到 key042；右边两题都立即答出，没有调用工具。结尾的卡片引用了左边的摘要和 TracePack 恢复的内容，最后一张卡片汇总了四次运行的结果。" width="100%">

<sub>录制环境为 Claude Code 2.1.302（Sonnet 5.5）。同一个脚本分别在原版和安装了 TracePack 的会话中运行，自动压缩阈值均为 100k token。视频剪去了等待模型响应的时间，两边同步以 1.3 倍速播放，展示的是 4 次运行中的第 3 次。原版只有一次能根据当前上下文答出 key087，因为那次摘要恰好保留了这些 key；其余三次都需要搜索会话记录才能找到 key042，分别耗时 24 秒、22 秒和 5 秒。安装了 TracePack 的会话在 4 次运行中都立即答对了两题，没有调用工具：每次压缩后，完整的 key 列表都被放回了上下文。结尾引用的是实际生成的摘要和恢复内容，答案均与原始输出一致。点击可查看大图。</sub>

<picture><source media="(prefers-color-scheme: dark)" srcset="assets/results-zh-dark.png"><img src="assets/results-zh.png" alt="测试结果。LongMemEval-S 对话历史每到 32k token 就压缩：Claude Code 准确率为 32.1%，安装 TracePack 后为 75.8%，Codex 压缩方法为 68.1%。CodeMemo 编程会话中，在 2,000 token 内召回答案所在轮次的比例：TracePack 34.6%，BM25 34.0%，grep 13.7%，只看最新上下文为 3.9%。研究测试框架中的 SWE-bench Verified，32k 窗口：达到步数上限的比例从 19.2% 降至 8.4%，问题解决率从 46.7% 变为 50.0%，统计上持平。" width="100%"></picture>

- `32% → 76%`：LongMemEval-S 上，历史每到 32k token 就压缩时，Claude Code 安装插件前后的准确率。
- `2.5×`：真实 Claude Code 编程会话中，证据召回率相对 grep 的倍数；与 BM25 持平，无需模型。
- `−10.8 pp`：SWE-bench Verified 上，达到步数上限的比例下降幅度；问题解决率持平。

```
/plugin marketplace add oklen/TracePack
/plugin install tracepack@tracepack
```

Agent memory 系列：[VeraKV](https://github.com/oklen/VeraKV) 讨论该把什么交给模型；[Compute Globally, Materialize Locally](https://github.com/oklen/Compute-Globally-Materialize-Locally) 讨论数据被驱逐后还保留什么；[TracePack](https://github.com/oklen/TracePack) 讨论会话压缩后还保留什么。

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) ![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg) ![Claude Code plugin](https://img.shields.io/badge/Claude%20Code-plugin-d97757.svg) ![No dependencies](https://img.shields.io/badge/dependencies-none-brightgreen.svg)

## 测试结果

### Claude Code：LongMemEval 准确率提高 43.7 个百分点

我们用真实的 Claude Code（2.1.294）测试了同一批 480 道 LongMemEval-S 题，分别运行原版和安装 TracePack 的版本。每道题的流程如下：

- 将历史消息（约 50 个会话、~115k token）回放到一个 Claude Code 会话中；
- 每超过 32k token，就运行一次 `/compact`，几乎每道题都会压缩三次；
- 最后根据压缩后保留的内容作答，不使用工具。

各实验组使用同一个模型 DeepSeek-V4-Pro，通过内置桥接组件接入，压缩时机也相同。答案按 LongMemEval 官方提示词评分。

| 480 题，同一个模型 | 准确率 |
|---|---|
| **Claude Code + TracePack** | **75.8%** |
| Claude Code + TracePack，只用笔记 | 70.2% |
| Claude Code + TracePack，只恢复用户原话 | 53.8% |
| Claude Code | 32.1% |
| *参考：Codex 压缩方法* | *68.1%* |

- 准确率提高 43.7 个百分点，95% CI [+39.0, +48.5]。在 219 道题上由错变对，9 道题上由对变错，六种题型的准确率均有提高。其中，跨会话题从 21% 提高到 73%，时间推理题从 19% 提高到 71%，用户事实题从 42% 提高到 94%。
- 单独使用带日期的笔记，准确率提高 38.1 个百分点；单独恢复用户原话，提高 21.7 个百分点。
- 原版 Claude Code 的压缩提示词面向编程任务，主要保留需求、文件、报错和下一步操作。三次压缩后，大部分个人信息已被省略。480 道题中，原版有 271 道回答「I don't have that information」，安装 TracePack 后为 80 道。原版也不像 Codex 那样原样保留用户消息。
- 启用插件后，摘要长度增加约 45%，中位数为 5.5k token，原版为 3.8k token；每次压缩后还会恢复约 2.2k token 的内容。
- 研究方案在同一测试框架、同一模型下的准确率为 81.9%。该方案由模型挑选并原样保留 6.5k token，约为插件的三倍；插件则在本地挑选约 2k token。

这是一组压力测试：用 32k 窗口处理 115k token 的历史，需要压缩三次。Claude Code 实际窗口为 200k，日常使用中要到更长的会话才会遇到这种情况。另外还有两处差异：

- 测试模型为 DeepSeek-V4-Pro，而非 Claude；
- Claude Code 会把机器上的日期告诉模型，而每道题有自己的日期。

运行 `tracepack bench lme run --arms claude-code,claude-code+tracepack` 可复现，具体步骤见[测试方法](docs/BENCH_LME.md)。

### 研究方案：LongMemEval 准确率比 Codex 压缩方法高 17.5 个百分点

上下文满后，Codex 会用一份交接摘要记录任务进展，细节留待 agent 之后重新读取。文件还在时，这种做法可行；用户只在聊天里说过的话却没有其他来源。在所需事实仍位于当前上下文的 305 个案例中，Codex 的交接摘要只保留了 4 个。

TracePack 的研究方案用笔记式摘要记录信息，再按内容挑选用户消息，原样保留。在 LongMemEval-S 上，对约 115k token 的长对话使用 32k 窗口，每道题的历史大约压缩三次：

| 480 题，相同的时间窗口和答题模型 | 准确率 |
|---|---|
| **TracePack 压缩方法**（笔记式摘要 + 用户原话） | **86.0%** |
| Codex 压缩方法（按 Codex 源码重新实现） | 68.5% |

- 准确率高 17.5 个百分点，95% CI [+13.3, +21.7]。六种题型均有提高，在 280 道新题上高 19.3 个百分点。
- 与把相同预算全部用于笔记式摘要相比，准确率持平，差值为 +1.0 [−2.3, +4.2]；TracePack 少生成了 30% 的 token。
- 在 SWE-bench Verified 的 32k 窗口测试中，每次压缩后恢复被移除的文件查看内容，相比 OpenHands 自带的压缩方法，agent 达到步数上限的比例降低了 10.8 个百分点（−10.8 [−18.6, −3.0]），问题解决率持平。

这些结果来自研究测试框架中预注册的配对实验。LongMemEval 部分已收录在仓库的 `tracepack bench lme` 中，提示词与研究版本逐字节一致；使用研究中的评分结果生成报告，可以得到上表。换成 DeepSeek-V4-Pro 重跑，同一方法比 Codex 压缩方法高 +13.7 [+8.8, +18.5] 个百分点。实验细节及其他基准测试见[研究说明](docs/RESEARCH.md)。

### 压缩后检索：证据召回率为 grep 的 2.5 倍

CodeMemo 是一个公开基准测试，包含来自 66 个真实 Claude Code 会话的 153 道题，每个项目的历史最多为 430 MB。下表比较在相同的 2,000 token 预算内，原样召回答案所在轮次和确切值的比例：

| 2,000 token 以内 | 召回答案所在的轮次 | 召回确切的值 |
|---|---|---|
| **TracePack** | **34.6%** | **47%** |
| grep 会话记录（未安装插件时 Claude 的做法） | 13.7% | 17% |
| 当前上下文（最新的 2,000 token） | 3.9% | 12% |
| *参考：基于消息的 BM25 搜索引擎* | *34.0%* | *47%* |

- 证据召回率为 grep 会话记录的 2.5 倍，确切值召回率为 2.8 倍；证据召回率约为只看当前上下文的 9 倍。
- 调试类问题的证据召回率为 grep 的 4.7 倍。例如「当时怎么定位和修好 X」，TracePack 为 48%，grep 为 10%。
- 整体表现与 BM25 搜索引擎相当。返回内容按记录组织，注明工具、时间和会话记录行号，工具调用附带简短输出，总长度受预算限制。
- 检索不调用模型、不联网。在最多 52,000 条记录的历史中，每题约耗时 0.4 秒。
- 在 Claude Code 中，压缩前打印的随机 key 可以在 `/compact` 后的一轮对话内找回，无需重新运行命令。

这项测试不使用模型，结果可确定复现：CodeMemo 标注为证据的轮次需要在预算内被召回。运行 `tracepack bench codememo --data <CodeMemo 目录>` 即可复现。默认设置在奇数编号题目上选定，偶数编号题目留作验证。在偶数题上，TracePack 的证据召回率与 BM25 持平（34.7% vs 34.7%），确切值召回率更高（43% vs 40%）。完整数据见 [Benchmark](#benchmark)。

## 安装

在 Claude Code 中运行：

```
/plugin marketplace add oklen/TracePack
/plugin install tracepack@tracepack
```

Claude Code 2.1.275 及以上版本可以用一条命令完成安装。2.1.292 及以上版本也支持在 shell 中运行 `claude plugin install`，参数相同：

```
/plugin install tracepack --marketplace oklen/TracePack
```

安装后重启 Claude Code，或运行 `/reload-plugins`，再用 `/tracepack:status` 检查状态。

需要 Python 3.9+，且 `PATH` 中能找到 `python3`。macOS 和大多数 Linux 发行版已自带 Python，无需安装其他依赖。

## 60 秒上手

1. 让 Claude 运行以下命令，然后回复「done」：
   ```
   python3 -c "import secrets; [print('key%02d' % i, secrets.token_hex(8)) for i in range(30)]"
   ```
2. 运行 `/compact`。
3. 问：*「key17 是多少？什么都别运行。」*

这些 key 是随机生成的，重新运行会得到另一组值，答案只能从原始输出中找。压缩结束后会显示 *TracePack restored N exact records*，用 `/tracepack:last` 可以查看恢复了哪些内容。Claude 随后会根据恢复的输出，在一轮对话内作答，无需重新运行命令。

短会话中，Claude Code 自己的摘要也可能保留这些 key。会话变长、摘要省略大部分输出后，TracePack 的作用才更明显。没有 TracePack 时，Claude 可能重新运行命令，得到另一组 key，或用 grep 搜索自己的会话记录文件。

## 功能

- 每次手动或自动压缩时，Claude Code 都会额外写一个带日期的「Notes」小节。用户提供的每个事实（需求、约束、偏好、名字、数字、路径、版本）、Claude 的每个回答，以及每处改动的旧值和新值，各占一行。具体值按原话照抄，笔记会延续到后续压缩。
- 每次压缩后，TracePack 会在约 2,000 token 的预算内恢复以下内容：
  - 从用户全部消息中按内容挑选的事实陈述，保留原文和日期；
  - 最新的工具输出；
  - `/compact` 指令和最近请求所指向的记录。

  这些内容按时间排列，每条都有来源标签，例如 `Bash output of \`pytest -q\` · 10-08 13:14 · L22`。
- `recall` 工具用于查找压缩前的其他内容。Claude 可以直接提问，例如「上一次 pytest 的确切报错」或「用户给的数据库端口」，并在预算内拿到原样记录。来自工具的记录会标明对应的工具调用。
- `expand` 工具用于分页查看长记录的全文。
- 可用命令：
  - `/tracepack:recall <问题>`
  - `/tracepack:status`
  - `/tracepack:last`，查看上一次压缩后恢复的内容。

下面是压缩后 Claude 看到的内容，来自内置示例会话，可运行 `tracepack demo` 查看：

```
TracePack: the user's own words from before the compaction, verbatim. These sentences were picked
because they state facts (names, numbers, dates, preferences, decisions, constraints); "…" marks
where a message was shortened. The summary above may have paraphrased them.

[u1] user said · 2026-10-08 17:00 · L1
… Profile it and keep the numbers; the staging DB is at port 6543.

[u2] user said · 2026-10-08 17:10 · L11
… Also the migration id we must not touch is 2026_09_30_add_ledger_index.

TracePack: 7 verbatim records from this session before the compaction, in the order they happened.
The summary above may have shortened them. Other records can be looked up with the TracePack recall tool.

[2] Bash output of `python bench.py --rows 200000` · 10-08 17:02 · L3
rows=200000 elapsed=41.7s p95_latency=812ms
hot path: parse_rows 63% of time
...
[5] Edit call · 10-08 17:06 · L7
Edit {"file_path": "/work/demo/bench/parse.py", "new_string": "for line in buf.splitlines():", ...}
...
[7] Bash output of `python bench.py --rows 200000` · 10-08 17:09 · L10
rows=200000 elapsed=29.3s p95_latency=530ms
hot path: parse_rows 41% of time
```

## 工作原理

Claude Code 会将每条消息和工具输出写入 `~/.claude/projects/<project>/<session>.jsonl`，TracePack 直接读取这份文件，不另行记录会话。

TracePack 按以下步骤查找和恢复内容：

1. 按 id 配对工具调用及其输出，识别会话压缩和压缩时原样保留的消息。

2. 只搜索已移出上下文的内容，避免用预算重复返回 Claude 已有的信息。
3. 用 BM25 计算记录与问题的匹配分数，不调用模型。
4. 按排名将完整记录放入预算，再按发生时间展示。工具输出注明对应的调用，工具调用附带简短输出。若高排名记录过长，剩余预算无法容纳，则返回匹配行的带标签摘录，不做未标明的截断。

所有返回内容都会做脱敏处理，遮盖看起来像凭据的字符串，包括 API key、token 和密码。

用户原话也在本地筛选。每个句子按事实密度评分：数字、日期、名字、标识符、第一人称陈述、偏好、计划、决定，以及「never」「must」这类约束都会计入。提问、道谢和粘贴材料的得分较低。筛选优先保留单位 token 内事实较多的句子，因此可以从长消息中单独取出一句事实陈述。

插件通过两个开销很小的 hook 接入：

- PreCompact 记录 `/compact` 指令，输出记笔记的说明。Claude Code 会将这段输出接在用户的 `/compact` 指令后，一起放入压缩提示词。
- SessionStart 在压缩结束后，将用户原话和记录放回上下文。

插件没有逐次工具调用的 hook，不运行守护进程，也不占用端口。Claude Code 在这些 hook 运行后才写入压缩摘要，因此 TracePack 根据 `/compact` 指令和最近的工作构造查询，并优先恢复最新工具输出。

## 适用场景与限制

TracePack 适合保留难以重新获取的细节：

- 只打印过一次的值，如随机 id、耗时和基准测试数字；
- 耗时较长、需要远程执行或有副作用的命令输出；
- 你只在聊天里告诉 Claude 的事情；
- 文件修改前的旧内容。

如果文件还在，重新读取的成本也低，让 Claude 重读就能达到同样的效果。即使没有 TracePack，只要摘要提供了线索，Claude 有时也能用 grep 从原始会话记录中找到某个值，只是需要额外调用一次工具，搜索 JSON lines 文件。

我们的研究发现，压缩中丢失的细节大多可以低成本重读。在 SWE-bench 上，恢复这些记录并没有改变问题解决率。TracePack 主要用于难以重读的内容，详见[研究说明](docs/RESEARCH.md)。

## 隐私与开销

插件在本地运行，读取本机文件，并在插件 data 目录（`~/.claude/plugins/data/tracepack…`）写入几个小 JSON 文件。它不发送网络请求、不收集遥测数据，也不调用模型。

仓库中的基准测试和研究代码（`tracepack bench` 以及 `eval`、`pilot`、`profiler`、`baselines` 目录）只在手动运行时可能联网。`tracepack bench lme download` 会从 Hugging Face 下载 LongMemEval，使用模型的基准测试会调用配置的服务端点。插件的 hook 和 MCP server 不会运行这些代码。

额外的上下文开销来自压缩摘要中的笔记，以及每次压缩后恢复的最多约 2,000 token。`recall` 返回的工具输出大小由 Claude 请求的预算决定。

插件不会写入你的仓库或设置。

会话记录由 Claude Code 管理，超过 `cleanupPeriodDays`（默认 30 天）就会被删除。TracePack 只能检索尚未删除的内容。

## 设置

默认即可使用。需要调整时，设置相应环境变量，例如放在 `~/.claude/settings.json` 的 `"env"` 下：

| 变量 | 默认值 | 作用 |
|---|---|---|
| `TRACEPACK_NOTES` | `1` | 每次压缩时保留带日期的笔记。`0` 关闭。 |
| `TRACEPACK_INJECT_BUDGET` | `2000` | 每次压缩后恢复的 token 数，最多 8,000 个字符。`0` 关闭。 |
| `TRACEPACK_USER_WORDS` | `1` | 恢复内容中包含用户原话。`0` 只恢复记录。 |
| `TRACEPACK_RECALL_BUDGET` | `2000` | `recall` 返回内容的默认预算。 |
| `TRACEPACK_MAX_MB` | `64` | 会话记录过大时，只读取最新部分。 |
| `TRACEPACK_REDACT` | `1` | 遮盖疑似凭据的字符串。`0` 关闭。 |

## 命令行

检索引擎也可以在 Claude Code 之外使用：

```bash
uv tool install git+https://github.com/oklen/TracePack     # 或者：pipx install git+https://github.com/oklen/TracePack
tracepack recall "exact error from the last pytest run"     # 当前项目最新的会话
tracepack recall "the port I gave you" --session <id 或路径>
tracepack expand 417                                        # 一条 record 的全文
tracepack status | last | sessions | demo | doctor
tracepack bench codememo --data <CodeMemo 目录>            # recall benchmark（不用模型）
tracepack bench lme run --arms tracepack,codex --out runs   # 带 compaction 的 LongMemEval（需要模型）
```

## 接入其他 agent（MCP）

`tracepack mcp` 是无依赖的标准 stdio MCP server，提供 `recall`、`expand` 和 `status`。任何 MCP client 都可接入：

```json
{ "mcpServers": { "tracepack": { "command": "tracepack", "args": ["mcp"] } } }
```

Python API 还支持读取 pi 会话和 OpenHands trace（`tracepack.pack`），并附带 OpenHands condenser 和 LlamaIndex memory block，见[库参考](docs/RESEARCH.md)。

## Benchmark

[CodeMemo](https://github.com/laynepenney/codememo-benchmark)（MIT）基于三个软件项目的 66 个 Claude Code 会话构建了 158 道题，其中 153 道可以定位到证据所在轮次。测试将每个项目的会话按顺序拼接为长历史，模拟多次压缩后的会话。对每道题和每种预算，各方法返回一组记录，再检查两项指标：

- 答案所在轮次的召回率：CodeMemo 标注的证据轮次中，至少有一个出现在返回记录里。
- 确切值召回率（表中括号内的数字）：简短答案中的具体值原样出现在返回内容里。此项只统计答案包含数字、版本号、路径或标识符的题目。

所有方法使用相同的 token 估算方式：约 4 个字符计为一个 token。

| 方法 | 1,000 token | 2,000 token | 4,000 token |
|---|---|---|---|
| **TracePack** | 26.1% (37%) | 34.6% (47%) | 39.2% (56%) |
| 基于消息的 BM25 | 29.4% (36%) | 34.0% (47%) | 40.5% (55%) |
| grep 会话记录 | 10.5% (12%) | 13.7% (17%) | 17.0% (21%) |
| 只看最新上下文 | 3.3% (8%) | 3.9% (12%) | 5.2% (16%) |

- TracePack 使用默认设置的 `recall`：BM25 排序，按排名放入带标签的完整记录，附上工具调用的简短输出；高排名记录放不下时，返回摘录。
- BM25 使用相同排序，不加标签，也不折叠输出，是最强的简单基线。
- grep 从原始 JSON lines 中读取包含问题里两个最具体词语的行，按文件顺序返回，每行最多 2,000 个字符。
- 「只看最新上下文」保留预算能容纳的最后几条记录。

按题目类别统计，2,000 token 预算下答案所在轮次的召回率如下：

| 类别 | TracePack | BM25 | grep | 只看最新上下文 |
|---|---|---|---|---|
| Factual | 37% | 37% | 17% | 6% |
| Debug | 48% | 45% | 10% | 0% |
| Architecture | 32% | 32% | 7% | 4% |
| Temporal | 24% | 24% | 14% | 5% |
| Convention | 35% | 35% | 20% | 5% |
| Cross-session | 25% | 25% | 15% | 5% |

默认设置在奇数编号题目上选定，再用偶数编号题目验证，预算均为 2,000 token。选定的设置为：使用 BM25，不叠加 hashing embedding；不做 dependency closure；按排名打包。

| 题目分组 | 方法 | 答案所在的轮次 | 确切的值 |
|---|---|---|---|
| 奇数（n=78） | tracepack | 34.6% | 51% |
| 奇数（n=78） | bm25 | 33.3% | 53% |
| 偶数（n=75） | tracepack | 34.7% | 43% |
| 偶数（n=75） | bm25 | 34.7% | 40% |

这项测试没有测量答题准确率，因为准确率还取决于读取记录的模型。它也没有测量 `/compact` 后的自动恢复。自动恢复使用同一个打包器，但会优先放入最新工具输出；前面的 LongMemEval 测试对恢复内容和笔记进行了端到端评估。

早期研究结果，包括 dependency closure 的适用情况，见[研究说明](docs/RESEARCH.md)。

### LongMemEval 会话压缩测试

`tracepack bench lme` 是前述压缩实验使用的测试框架。每条 LongMemEval-S 历史约含 50 个会话、~115k token，逐步送入 32k 上下文，装满后按对应实验组的方法压缩。最后只提问一次，不使用工具，答案按 LongMemEval 官方提示词评分。各实验组如下：

- `tracepack`：研究方案，使用笔记式摘要，加上按内容挑选、原样保留的用户消息。
- `codex`：按 Codex CLI 源码重新实现的压缩方法。
- `notes`：只使用笔记式摘要，总预算相同。
- `full`：不压缩。
- `claude-code` 和 `claude-code+tracepack`：真实 Claude Code，分别不安装和安装插件，另有两个消融实验。

提示词、预算和统计方法均与研究版本一致，测试会逐字节核对提示词。可使用任何兼容 OpenAI 接口的模型：

```bash
pip install tiktoken
export TRACEPACK_LLM_BASE_URL=… TRACEPACK_LLM_API_KEY=… TRACEPACK_LLM_MODEL=… TRACEPACK_JUDGE_MODEL=…
tracepack bench lme run --arms tracepack,codex --out runs/lme      # 可断点续跑；--dry-run 不发出任何调用
tracepack bench lme report --out runs/lme
```

每道题的每个实验组约消耗 150k 输入 token。环境设置、两个 Claude Code 实验组及输出格式见 [docs/BENCH_LME.md](docs/BENCH_LME.md)。

## 故障排查

先运行 `/tracepack:status` 或 `tracepack doctor`。doctor 会检查 Python、查找会话记录，并用内置示例会话验证完整流程。

- 出现 `python3: command not found`：安装 Python 3.9+。
- `/compact` 后没有恢复内容：检查压缩前是否有工具输出或用户事实陈述，以及 `TRACEPACK_INJECT_BUDGET` 是否被设为 `0`。
- 手动 `/compact` 后出现记笔记的说明：Claude Code 会将各 PreCompact hook 的输出显示在 `/compact` 结果中。设置 `TRACEPACK_NOTES=0` 可以关闭笔记。
- hook 报错：错误不会中断 Claude，会写入 data 目录下的 `hook_errors.log`，`tracepack doctor` 也会显示。

## 卸载

```
/plugin uninstall tracepack@tracepack
/plugin marketplace remove tracepack
```

卸载会删除插件的 data 目录，原有会话记录不受影响。

## 开发

```bash
python3 tracepack/tests/run_all.py                 # 310 个测试，按 contract 逐项检查
claude plugin validate . --strict                  # manifest、marketplace、hook、MCP 配置
claude --plugin-dir .                              # 在 Claude Code 里试用工作副本
```

版本记录见 [CHANGELOG](CHANGELOG.md)。

## 许可

MIT © 2026 Zefeng Cai，见 [LICENSE](LICENSE)。
