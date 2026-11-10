# mini-agent —— 从零实现的最小可用 Agent

不依赖 LangGraph / OpenHands / OpenClaw 等任何 Agent 框架，手写 **ReAct 循环、工具注册、会话隔离、上下文压缩**。
129 条单元测试 2 秒内跑完且不需要 API Key，另有并发与真实 API 两个自测脚本。

```
仓库结构
└── mini-agent/          ← 项目本体，所有命令都在这个目录下执行
    ├── agent/           # 核心实现
    ├── tests/           # 129 条单元测试
    ├── scripts/         # 并发 / 真实 API 自测
    └── main.py          # CLI 入口
```

---

## 1. 快速开始

### 1.1 安装

```bash
git clone https://github.com/ying88618/Agent_Harness_test.git
cd Agent_Harness_test/mini-agent
pip install -r requirements.txt      # 只有 openai + pytest 两个依赖
```

Python 3.10+（用到了 `X | None` 语法）。

### 1.2 配置模型

复制 `.env.example` 为 `.env`，填入任意 **OpenAI 兼容** 服务的凭据：

```ini
LLM_API_KEY=sk-xxxx
LLM_BASE_URL=https://api.siliconflow.cn/v1
LLM_MODEL=deepseek-ai/DeepSeek-V4.1-Flash
```

DeepSeek / 通义 DashScope 兼容模式 / SiliconFlow / 本地 vLLM 都可以，只改这三个变量。
也可以用命令行覆盖：`--api-key` / `--base-url` / `--model`。

### 1.3 跑起来

```bash
python main.py                                  # 默认会话
python main.py --session window-1               # 指定会话（= 一个窗口）
python main.py --once "厦门今天天气怎么样"       # 单次执行后退出
python main.py --quiet                          # 不打印工具调用过程
```

交互模式下的命令：

| 命令 | 作用 |
|---|---|
| `/session <id>` | 切换会话（等价于换窗口） |
| `/sessions` | 列出已落盘的会话 |
| `/history` | 查看当前会话消息数 / 摘要长度 |
| `/reset` | 清空当前会话（消息 + 待办） |
| `/exit` | 退出 |

### 1.4 跑测试

```bash
python -m pytest                       # 129 passed，不需要 API Key
python -m pytest tests/test_loop.py -v # 单文件
python -m pytest -k "repair or compress" -v

python scripts/check_concurrency.py            # 并发自测（20 线程）
python scripts/check_concurrency.py --threads 200
python scripts/smoke_api.py                    # 真实 API 冒烟（需要 .env）
```

---

## 2. 演示：两个窗口互不影响

题面要求「用户 A 开窗口 1 查天气记待办，开窗口 2 写周报记待办，两个 session 独立」。终端里两条命令即可复现：

```bash
python main.py --once "查一下厦门天气，然后记一条待办：明天带伞" --session window-1
python main.py --once "记一条待办：写本周周报"                    --session window-2

# 关掉终端后再开，验证落盘与记忆
python main.py --once "我刚才让你记的待办是什么？" --session window-1
python main.py --once "我的待办有哪些？"           --session window-2
```

实测输出（节选）：

```
窗口1 → -> weather({"city": "厦门"})  -> todo({"action": "add", "text": "明天带伞"})
        助手 > ...待办 #1「明天带伞」
窗口2 → -> todo({"action": "add", "text": "写本周周报"})
窗口1 追问 → -> todo({"action": "list"}) → 你刚让我记的是 #1 明天带伞
窗口2 追问 → -> todo({"action": "list"}) → 你当前有 1 条待办：#1 写本周周报
```

注意最后一组：**追问时模型是去调 `todo(list)` 查的，而不是"回忆"**。这是有意的设计，见 §4.4。

---

## 3. 系统设计

### 3.1 目录与职责

```
mini-agent/
├── main.py              CLI：REPL、会话切换、参数解析
├── agent/
│   ├── types.py         跨模块数据结构（ToolCall / LLMResponse / ToolResult / ToolContext）
│   ├── errors.py        分层异常（LLMError / ToolError / ToolArgError / MaxTurnsExceeded）
│   ├── llm.py           唯一联网的地方：OpenAI 兼容客户端 + 重试 + 响应归一化
│   ├── parser.py        ★ 模型输出 → (思考, 工具调用, 最终答案)，双通道 + JSON 容错
│   ├── registry.py      ★ 工具注册、schema 生成、参数校验、执行兜底
│   ├── tools/           calculator / search(mock) / todo / weather(mock)
│   ├── session.py       ★ 会话状态、per-session 锁、JSON 落盘、消息配对修复
│   ├── context.py       ★ system prompt 组装 + 按轮次边界压缩
│   ├── trace.py         JSONL 事件流（ContextVar 绑定）
│   └── loop.py          ★ ReAct 主循环
├── tests/               129 条单元测试（含 fake_llm.py 脚本化假模型）
└── scripts/             check_concurrency.py / smoke_api.py
```

### 3.2 一次请求的完整链路

```
用户输入
 └─ SessionStore.load_or_create(session_id)      ← 会话隔离的边界
     └─ with session.lock:                        ← 同一窗口串行，防交叉写
         ├─ maybe_compress()                      ← 超预算先压，压完立刻生效
         ├─ build_messages()                      ← system + 历史消息
         ├─ llm.chat(tools=registry.openai_tools())
         ├─ parse_response()
         │   ├─ 有 tool_calls → 写回 assistant → registry.execute() → 写回 tool → 回到循环
         │   └─ 无 tool_calls → 写回 assistant → 结束
         └─ SessionStore.save()                   ← 原子写盘
```

### 3.3 与 `create_agent` 的对应关系

`create_react_agent(model, tools)` 内部是一张状态图。本实现把图拉平成 `for` 循环，保留图存在的理由：

| 框架里的东西 | 本实现 | 说明 |
|---|---|---|
| `bind_tools(model, tools)` | `registry.openai_tools()` | 自动从函数生成 JSON Schema |
| `[tools]` 节点 | `for call in parsed.tool_calls` | 支持一轮内并行调用 |
| `should_continue` 条件边 | `if not parsed.has_tool_call: break` | — |
| `MessagesState` reducer | `session.append(...)` | 唯一状态就是消息列表 + 工具 store |
| `checkpointer` + `thread_id` | `SessionStore` + `session_id` | 落盘，可跨进程续聊 |
| `handle_tool_errors` | `registry.execute()` 五分支兜底 | 异常转成 observation 回灌 |
| middleware 钩子 | 循环里的显式 `if` + `trace.event()` | 无框架的壳 |

### 3.4 核心模块

**parser：双通道 + 四级容错**

```
① 原生 tool_calls（结构化，主路径）
② content 里的文本协议：```json {"tool":...}``` / 裸 JSON / ReAct 的 Action: + Action Input:
③ 都没有 → 当作最终回答
```

工具参数解析四级降级，**永不抛异常**：

| 级别 | 处理 | 例子 |
|---|---|---|
| 1 | `json.loads` | 正常 |
| 2 | 扫描括号平衡的 `{...}` 再试 | `好的，参数是 {"city":"厦门"}` |
| 3 | `ast.literal_eval` + 修复 | 单引号 / 尾逗号 / `True` / 中文引号 |
| 4 | 二次编码解包 | `"{\"city\": \"厦门\"}"` |

都失败则返回 `{"__raw__": 原文}` 并记一条 warning，由 registry 报 `bad_arguments` 回灌给模型，让它自己改。

**registry：执行异常一律不抛出**

| 情况 | error 码 | 回灌给模型的内容 |
|---|---|---|
| 工具名不存在（幻觉） | `tool_not_found` | 可用工具清单 |
| 必填 / 类型 / 枚举不符 | `bad_arguments` | 错误原因 **+ 完整参数 schema** |
| 工具主动抛 `ToolError` | `tool_error` | 失败原因 |
| 工具内部 bug | `internal_error` | 异常类型与信息 |

这是 Agent 能「自我修正」的前提：模型看到「缺少必填参数 city，schema 长这样」，下一轮就会改对。

**loop：五个必须处理的边界**

| 边界 | 实现 |
|---|---|
| 轮次上限 | `for turn in range(1, max_turns+1)` + `for...else` 收口 |
| 重复调用 | 同一组 `(工具名 + 参数)` 连续出现即短路，但**仍写一条 tool 消息**保持配对 |
| 模型空输出 | 回灌一条提示，给一次改正机会 |
| 模型调用失败 | 只捕获 `LLMError`，转成 `llm_error` 结果 |
| 工具报错 | 不捕获，交给 registry 转成 observation |

### 3.5 关键设计决策

| 决策 | 理由 | 代价 |
|---|---|---|
| 用 `chat.completions` 而非 Responses API | 见 §7 | 拿不到服务端托管上下文（本题也不需要） |
| 工具签名是 `func(args, ctx)` 而非 `func(**kwargs)` | `ctx` 带 `session_id` 与 `store`，todo 才能按窗口隔离 | 比框架的 `@tool` 多一个参数 |
| 异常分层，工具错误不往上抛 | 让模型自我修正，而不是让对话崩掉 | 错误会被"吞"，靠 trace 兜底 |
| 压缩用抽取式摘要，不调 LLM | 少一次调用、可复现、可单测 | 摘要质量不如生成式 |
| 摘要存 `session.summary` 并拼进 system 末尾 | system 前缀稳定，利于 prompt cache | 摘要不参与消息结构 |
| calculator 用 AST 白名单，不用 `eval` | 模型可能被注入输出 `__import__("os")...` | 不支持变量与复杂表达式 |
| todo 清单渲染上限 30 条 | 每轮回灌整份清单是 O(N²)，50 条就能顶爆上下文 | 长清单需靠 `id` 定位 |

---

## 4. Context 与 Memory

### 4.1 什么进 context，什么不进

| 信息 | 是否进 context | 说明 |
|---|---|---|
| 用户输入 | ✅ | 必须 |
| assistant 的 `content` | ✅ | 最终回答要留在历史里 |
| assistant 的 `tool_calls` | ✅ | 与随后 `role=tool` 消息成对，缺一个就 400 |
| 工具执行结果 | ✅ 但截断 | 头尾各留一半，上限 4000 字符 |
| 模型的 `reasoning_content` | ❌ | 只进 trace。推理模型要求不回传，且对后续决策无增量 |
| trace 事件 | ❌ | 日志是给人看的，进 context 只会污染 |
| 更早轮次 | ⚠️ 折叠成摘要 | 见 4.3 |

### 4.2 为什么工具结果必须留在 context 里

用户说「把刚才那个待办改到明天」时，模型需要从**上一轮的工具结果**里看到 `#1` 这个 ID 才能改。
如果工具结果被压缩成一句「用户记了个待办」，ID 就永久丢失，追问必然失败。

所以压缩时的摘要**必须保留工具结果要点**（见 4.3），这不是锦上添花，是追问能力的底线。

### 4.3 压缩策略

触发条件：`session.messages` 字符数 + 摘要 > `MAX_CONTEXT_CHARS`（默认 12000）。

```
原始:  [turn1][turn2][turn3][turn4][turn5][turn6]
压缩后:                     [turn5][turn6]
        └─ turn1~4 折叠成 session.summary
```

三条硬规则：

1. **只在 turn 边界切。** 一个 turn = 一条 user 消息 + 其后所有 assistant/tool 消息。按 turn 整体裁剪，`tool_calls` 和 `tool` 消息永远同进同出。
2. **绝不在工具循环中间硬切。** 只剩最近 N 个 turn 仍超预算时，宁可让这次请求大一点——因为切了就会破坏配对，而配对断裂是**持久性**故障（存盘后每次加载都 400）。
3. **压缩发生在每轮组装 context 之前。** 这样无论本轮走工具分支还是纯对话分支都会触发，且压完立刻作用于本次请求。

摘要长这样（抽取式，不调 LLM）：

```
- 用户：记个待办：明天带伞
  工具：todo→已添加待办 #1：明天带伞
  助手：记好了
```

### 4.4 Memory 的召回时机与放置方式

**这个问题在这套设计里有个可能反直觉的答案：大部分"记忆"不在 prompt 里，在工具里。**

| 层 | 内容 | 放置位置 | 写入时机 | 召回时机 |
|---|---|---|---|---|
| 工作记忆 | 当前轮 user / assistant / tool 消息 | `session.messages` 尾部 | 每一步即时写入 | 每次调模型前全量带上 |
| 短期记忆 | 最近 `KEEP_RECENT_TURNS`（默认 2）个完整 turn 原文 | `session.messages` | 压缩时保留 | 每次调模型 |
| 长期记忆 | 更早 turn 的抽取式摘要（含工具要点） | `session.summary` → 拼进 **system prompt 末尾** | 压缩触发时追加 | 每次调模型 |
| 外部记忆 | 待办清单等结构化状态 | `session.store`（JSON 落盘） | 工具调用时写入 | **模型主动调 `todo(list)` 时** |

四条设计说明：

1. **摘要放在 system prompt 末尾，而不是塞进 messages。** 前文（工具清单、规则）保持稳定前缀，能命中 provider 的 prompt cache；同时不会污染对话结构（不会出现"system 消息夹在 user/assistant 中间"）。
2. **不需要向量检索。** 待办这类状态用工具读取比用 prompt 记忆更准：`todo(list)` 返回的是真相，而 prompt 里的"我记得你有三条待办"只是可能过期的印象。这也是题面「带工具的追问」的正确形态。
3. **跨进程召回靠落盘。** 关掉终端再开，`--session window-1` 会从 `sessions/*.json` 读回消息、摘要、工具 store 三样东西。只存消息不存 store，模型会说"待办 #1"但实际查不到。
4. **摘要不可能完全无损。** 因此压缩保留了工具要点这一层最小信息；更细的历史细节在超长对话里确实会丢，这是有意的取舍（题目要的是"基础的压缩"）。

---

## 5. 工具

| 工具 | 用途 | 失败形态 |
|---|---|---|
| `calculator` | AST 白名单求值，支持 `+ - * / // % **`、括号、`sqrt/log/sin` 等 | 拒绝属性访问、下标、lambda、非白名单函数、除零、指数炸弹 |
| `search` | 本地 mock 语料 + 关键词打分，返回 `{title, snippet, url, score}` | 无结果时返回友好提示 + 换词建议 |
| `todo` | `add / list / done / update / remove / clear`，**按 session 隔离** | 缺 text、id 不存在、id 非法 |
| `weather` | 7 个城市今日/明日天气（mock） | 空城市抛错；未收录城市返回可用城市清单 |

`calculator` 不用 `eval` 是被动防御的底线：模型完全可能被 prompt 注入诱导输出 `__import__("os").system(...)`。
测试里逐条验了 `().__class__.__bases__`、`[1,2][0]`、`[i for i in range(10)]`、`9**99999` 等 10 种输入。

工具注册示例：

```python
@tool(
    name="todo",
    description="管理当前会话的待办清单……",
    parameters={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["add", "list", "done", "update", "remove", "clear"]},
            "text": {"type": "string"},
            "id": {"type": "integer"},
        },
        "required": ["action"],
    },
)
def todo(args, ctx): ...
```

新增一个工具只需要写这个函数并在 `tools/__init__.py` 里注册，schema、校验、异常兜底、trace 全部自动生效。

---

## 6. 测试

### 6.1 分层

| 层 | 文件 | 条数 | 依赖 |
|---|---|---|---|
| 纯函数 | `test_parser.py` / `test_context.py` | 37 | 无 |
| 组件 | `test_tools.py` / `test_registry.py` | 56 | 无 |
| 状态 | `test_session.py` / `test_trace.py` | 25 | 临时目录 |
| 集成 | `test_loop.py` | 11 | 脚本化假模型 |

**129 条全部不需要 API Key**，2 秒内跑完。这是刻意的分层结果：网络只存在于 `llm.py`，`test_loop.py` 用 `tests/fake_llm.py` 注入模型。

### 6.2 假模型是集成测试的关键

```python
ScriptedLLM([tool_reply(call("weather", {"city": "厦门"})), reply("今天晴")])
```

它做三件事：
1. 接口与 `LLMClient.chat` 完全一致 → 被测代码不用为测试改一行；
2. 记录每次收到的 `messages` → 能断言"模型看到了什么"，这是测 context 的唯一手段；
3. 支持 callable 作脚本项 → 可以按轮次动态生成参数，模拟"一直调工具不停"。

真实 API 造不出这些场景，或者造出来不可复现且烧钱。

### 6.3 自测脚本

```bash
python scripts/check_concurrency.py --threads 200
```
- 场景 1：200 线程写同一 session → 断言 800 条消息**零交错**（每 4 条必须是连续的 `[user, assistant, tool, assistant]`）、200 条待办、无孤立 tool 消息。
- 场景 2：200 个 session 并发 → 断言每个 session 只有自己的待办。

```bash
python scripts/smoke_api.py
```
4 个真实场景：单工具计算 / 一轮多工具 / 同 session 追问 / 跨 session 隔离。断言尽量查**程序状态**而不是模型措辞（例如隔离性直接查 `store`，不问模型"你有没有待办"）。

### 6.4 未覆盖的部分

- `main.py` CLI 的 argparse 分支（手动测）
- 多进程写同一个 session 文件（需要文件锁，题目未要求）
- 流式输出（未实现）

---

## 7. 附：为什么用 `chat.completions` 而不是 Responses API

1. **题目要求自己管理 context。** Responses API 的核心卖点是 `previous_response_id`，把上下文托管在服务端——那"压缩""哪些信息进 context""memory 放置方式"就无从实现也无从演示。本实现必须持有完整 messages 数组。
2. **可移植性。** DeepSeek、通义 DashScope 兼容模式、SiliconFlow、vLLM、Ollama 的兼容层目前都只提供 `/v1/chat/completions`。用 Responses API 等于把项目锁死在 OpenAI 官方端点。
3. **解析层更可控。** Responses 的 `output` 是异构 item 列表（reasoning / message / function_call 混排），需要按 `type` 分支；chat completions 的 `tool_calls` 结构更收敛，可以把精力放在"模型把 JSON 写坏了怎么救回来"这个真正的难点上。
4. **可运行性。** 评审核对时改三个环境变量就能跑，成本几毛钱。

扩展点：若要增加 Responses 后端，只需替换 `agent/llm.py` 的 `_request()` 与 `_normalize()`，`chat()` 的入参/返回契约不变，上层无感。

---

## 8. AI 协作方式与问题解决记录

### 8.1 分工

| 环节 | 谁做 |
|---|---|
| 目录骨架、数据结构、工具模板 | AI 生成，人 review |
| parser 的容错分支、loop 的控制流 | AI 生成初稿，人逐行 review 边界条件 |
| 单元测试骨架 | AI 生成 |
| **逻辑漏洞** | **人发现的**（见 8.3 前三项） |
| 并发与真实 API 验证 | 脚本化后人工跑 |

结论：AI 能把"标准实现"写得又快又全，但**它倾向于覆盖 happy path 和显式需求，漏掉交叉场景**（例如"压缩"和"纯聊天"两条路径的交集）。这部分只能靠人读代码。

### 8.2 典型 prompt

- 「实现 OpenAI 兼容客户端封装，重试策略自己写，关掉 SDK 自带重试，异常统一收敛成 LLMError」
- 「parser 支持原生 tool_calls 和文本协议双通道，参数解析要四级降级，永不抛异常」
- 「写 loop 测试，用脚本化假模型，覆盖 max_turns / 重复调用 / 工具报错后自我修正」

共同点：**约束写得越死，AI 产出越可用**。只说"帮我写个 agent"，拿到的一定是玩具。

### 8.3 问题记录

| # | 现象 | 根因 | 修法 | 如何验证 |
|---|---|---|---|---|
| 1 | 长对话后上下文`tool_calls` 与 `tool` 消息配对断裂，请求 400 | 只删 assistant 或只删 tool | `repair_messages()`：按 turn 整体裁剪；缺结果补占位而不是删消息 | `test_repair_*` 8 条畸形输入 |
| 2 | **纯闲聊永远不触发压缩** | 压缩写在「工具分支」末尾，纯对话走 `break` 直接跳过了 | 挪到每轮开头、组装 context 之前 | `test_pure_chat_triggers_compression` |
| 3 | 轮次耗尽后这一轮"没有收口" | 兜底文案只返回给 CLI，没写回 session；末尾停在 `assistant(tool_calls)+tool` | 兜底回答写回 assistant，并列出已获取的信息 | `test_max_turns_keeps_history_and_closes_turn` |
| 4 | 压缩后追问"把刚才那个待办改到明天"失败 | 摘要只保留问答对，工具结果（含 ID）被整段丢弃 | 摘要增加「工具→结果要点」一行 | `test_summarize_keeps_tool_notes` |
| 5 | 50 个待办后上下文突然被压缩，agent"忘了前面的话" | `todo` 每次回灌整份清单，context 消耗 **O(N²)** | 渲染上限 30 条 + 从略提示 | `test_todo_render_is_capped` |
| 6 | 并发自测第一次 50 线程失败，待办翻倍 | **脚本没清理上轮数据**，`load_or_create` 把旧会话读回来了 | 每次运行前 `rmtree` | `check_concurrency.py` 干净起点 |

第 2、3、4 项是**人 review 发现的**，当时 129 条测试全绿——说明测试覆盖的是"我以为的路径"。第 5 项是被并发脚本顺带炸出来的，根因根本不在并发，而在数据规模。

### 8.4 两个反复踩的坑

**① 测试夹具必须和线上同构。**
假模型最初构造 `ToolCall(args={...})` 却没设 `raw_args`，而真实 API 传的是 JSON 字符串，导致测试走进了一个线上不存在的分支——还"通过"了。修完夹具立刻暴露了 parser 的一个缺口。**不同构的夹具比没有夹具更危险。**

**② 模型侧的错误不能抛异常。**
早期把工具异常直接往上抛，一次参数写错就整轮崩掉。后来统一改成"错误也是一条 observation"，模型自己会改对。这是 Agent 和普通程序的根本区别之一。

---

## 9. 已知限制

- **单轮超预算不压缩**：只剩最近 N 个 turn 仍超预算时返回 `False`，靠工具输出截断兜底。理由见 §4.3 第 2 条。
- **摘要是有损的**：只保留问答对 + 工具要点。
- **只有线程级并发保护**：多进程写同一个 session 文件没有文件锁，题目未要求。
- **未实现流式输出**：`llm.chat` 是一次性返回。
- **mock 工具**：`search` / `weather` 为本地数据，接口形态与真实 API 对齐，替换实现不影响上层。
