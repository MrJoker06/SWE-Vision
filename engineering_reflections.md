# SWE-Vision 工程反思记录

这个文档用于记录 SWE-Vision 开发过程中遇到的关键问题、排查过程、判断依据和后续改进方向。

它不是单一 bug 记录，而是一个长期反思文档。每一条记录都应尽量回答：

- 当时观察到了什么现象？
- 最初有哪些可能假设？
- 用什么实验或证据排除了哪些假设？
- 最后比较可信的判断是什么？
- 这件事暴露了哪些设计问题？
- 后续应该如何改进？

---

## 记录模板

### 主题

日期：

相关模块：

### 背景

记录问题出现的上下文、用户操作、环境配置、模型/provider、运行方式等。

### 现象

记录可复现的表面现象，尽量包含命令、日志、响应内容、截图描述等。

### 初始假设

列出当时怀疑的原因。

### 排查过程

记录做过的实验、对照组和观察结果。

### 结论

写出当前阶段最可信的判断，同时说明不确定性。

### 反思

记录这个问题暴露出的设计、抽象、提示词、测试、文档或使用体验问题。

### 后续行动

列出可以继续做的修复、实验或重构方向。

---

## 记录 1：DeepSeek 在工具调用场景下自称 Claude

日期：2026-04-30

相关模块：

- `swe_vision.agent`
- `swe_vision.config`
- `swe_vision.providers`
- Web/CLI 模型切换配置

### 问题概述

在将 SWE-Vision 切换到 DeepSeek 官方 API 后，模型配置和请求路径都显示为
`deepseek-v4-pro` 与 `https://api.deepseek.com`，但用户询问“你是谁？”时，
模型有时回答自己是 Claude：

```text
我是 Claude，一个由 Anthropic 开发的 AI 助手。
```

这需要判断：问题是 provider 路由错误、第三方 API 混用，还是 DeepSeek 在
SWE-Vision 的 agent/tool-calling 上下文中发生了身份幻觉。

### 关键结论

当前证据支持以下判断：

```text
1. Provider 路由没有明显错误。DeepSeek 测试确实请求到 https://api.deepseek.com。
2. Claude 身份漂移主要出现在 DeepSeek 官方 endpoint + 旧版泛化 system prompt
   + tools/tool_choice="auto" 的组合下。
3. 同一旧版 prompt 与同一 tools schema 下，当前 OPENAI_* endpoint 没有复现
   Claude/Anthropic 身份漂移。
4. tool_choice="none" 能显著降低 DeepSeek 的身份漂移，但不是根治方案。
5. 单纯改名 finish tool 不能稳定解决问题。
6. 更可靠的修复方向是：在 system prompt 中明确产品身份，例如
   “You are SWE-Vision ...”。
```

### 测试对象

DeepSeek 官方 endpoint：

```text
model=deepseek-v4-pro
base_url=https://api.deepseek.com
provider=deepseek
```

当前 `OPENAI_*` endpoint：

```text
model=gpt-5.4
base_url=https://yunjuan.top/v1
```

说明：`OPENAI_*` 当前指向第三方 OpenAI-compatible 网关，因此本文只称它为
“当前 OPENAI_* endpoint”，不把它等同于 OpenAI 官方 endpoint。

### 测试前提

为验证问题本身，测试时将 `SYSTEM_PROMPT` 保持为旧版泛化身份：

```text
You are an expert AI assistant with access to a stateful Jupyter notebook environment.
```

这个 prompt 定义了能力，但没有定义产品身份。模型被问“你是谁？”时，需要自行补全身份。

### 排除项

以下假设已基本排除：

- 不是 Web 前端只改显示、后端仍请求旧模型。
- 不是 DeepSeek API key/base URL 没有被读取。
- 不是请求被路由到 Anthropic 或 OpenRouter。
- 不是所有 OpenAI-compatible endpoint 都必然出现这个问题。

直接最小请求 DeepSeek 时，模型可以正常自称 DeepSeek；但进入 SWE-Vision
agent/tools 上下文后，Claude 身份漂移会出现。

### DeepSeek 结果

原始排查矩阵中，DeepSeek 的关键现象是：

```text
无 system、无 tools、无 reasoning
=> DeepSeek

SWE-Vision 原始/agent system、无 tools
=> 泛化 AI/Jupyter 助手

agent system、有 tools、tool_choice="auto"
=> 可触发 Claude

agent system、有 tools、reasoning high
=> 可触发 Claude

两个 tools 都给，但 tool_choice="none"
=> 回到 DeepSeek 或泛化 AI assistant
```

后续按旧版 prompt 补测 5 个方向：

```text
1. 将 finish 改名为 submit_answer / final_answer
   - submit_answer: 本次返回泛化 AI/Jupyter 助手
   - final_answer: 本次返回 Claude

2. 不同 tool schema
   - 原始 execute_code + finish: 泛化 AI/Jupyter 助手
   - 只保留 execute_code: Claude
   - 只保留 finish: 泛化 AI/Jupyter 助手
   - execute_code + submit_answer: Claude
   - execute_code + final_answer: Claude
   - 不传 tools: 泛化 AI/Jupyter 助手

3. tool_choice="auto" vs tool_choice="none"
   - “你是谁？” + auto: Claude
   - “你是谁？” + none: DeepSeek
   - “你好，简单介绍一下你自己。” + auto: Claude
   - “你好，简单介绍一下你自己。” + none: 泛化智能编程助手

4. 英文身份问题
   - “Who are you?” + auto: 调用 finish tool，tool 参数中返回 Claude
   - “Who are you?” + none: 泛化 AI assistant

5. 多次采样
   - “你是谁？” + 原始 tools + tool_choice="auto" 连续 5 次
   - Claude: 4 次
   - 泛化 AI/Jupyter 助手: 1 次
```

进一步扩展复测：

```text
旧 prompt + 原始 tools + tool_choice="auto" + reasoning high + 中文“你是谁？”连续 10 次
=> Claude: 6 次；泛化 AI/Jupyter 助手: 4 次

旧 prompt + 原始 tools + tool_choice="auto" + 不传 reasoning + 中文“你是谁？”连续 10 次
=> Claude: 6 次；泛化 AI/Jupyter 助手: 4 次

旧 prompt + 原始 tools + tool_choice="none" + reasoning high + 中文“你是谁？”连续 5 次
=> 泛化 AI/Jupyter 助手: 3 次；DeepSeek: 2 次；Claude: 0 次

旧 prompt + 不传 tools + reasoning high + 中文“你是谁？”连续 5 次
=> 泛化 AI/Jupyter 助手: 5 次；Claude: 0 次

无 system + 无 tools + 中文“你是谁？”连续 5 次
=> DeepSeek: 5 次；Claude: 0 次

无 system + 原始 tools + tool_choice="auto" + 中文“你是谁？”连续 5 次
=> Claude: 4 次；泛化 AI/Jupyter 助手: 1 次

短 system “You are an AI assistant.” + 原始 tools + tool_choice="auto" + 中文“你是谁？”连续 5 次
=> 泛化 AI 助手: 5 次；Claude: 0 次

旧 prompt + 原始 tools + tool_choice="auto" + reasoning high + 英文 “Who are you?” 连续 5 次
=> Claude: 5 次；泛化 AI assistant: 0 次

旧 prompt + 原始 tools + tool_choice="none" + reasoning high + 英文 “Who are you?” 连续 5 次
=> 泛化 AI assistant: 5 次；Claude: 0 次

显式 SWE-Vision prompt + 原始 tools + tool_choice="auto" + reasoning high + 中文“你是谁？”连续 5 次
=> SWE-Vision: 5 次；Claude: 0 次

显式 SWE-Vision prompt + 原始 tools + tool_choice="auto" + reasoning high + 英文 “Who are you?” 连续 5 次
=> SWE-Vision: 5 次；Claude: 0 次
```

### OPENAI_* 对照结果

在相同旧版泛化 prompt、相同 tools schema 和同类身份问题下，当前 `OPENAI_*`
endpoint 的全流程对照结果是：

```text
无 system、无 tools
=> 泛化 AI 助手

短 system、无 tools
=> 泛化 AI 助手

完整 agent system、无 tools
=> 泛化 AI/Jupyter 助手

无 system、有 tools、tool_choice="auto"
=> 泛化 AI 助手

agent system、有 tools、无 reasoning
=> 调用 finish tool，answer 为泛化 AI 助手

agent system、有 tools、reasoning high
=> 调用 finish tool，answer 为泛化 AI 助手

execute_code only / finish only / tool_choice="none"
=> 均为泛化 AI/Jupyter 助手
```

五项补测中，当前 `OPENAI_*` endpoint 也没有复现 Claude：

```text
finish 改名 submit_answer / final_answer
=> 泛化 AI 助手

不同 tool schema
=> 全部为泛化 AI/Jupyter 助手

tool_choice="auto" / "none"
=> 全部为泛化 AI 助手

“Who are you?”
=> 泛化 AI assistant

连续 5 次采样
=> 泛化 AI 助手: 5 次；Claude: 0 次
```

扩展对照中，当前 `OPENAI_*` endpoint 继续未复现 Claude：

```text
旧 prompt + 原始 tools + tool_choice="auto" + 中文“你是谁？”连续 5 次
=> 泛化 AI 助手: 5 次；Claude: 0 次

旧 prompt + 原始 tools + tool_choice="auto" + 英文 “Who are you?” 连续 5 次
=> 泛化 AI assistant: 5 次；Claude: 0 次

显式 SWE-Vision prompt + 原始 tools + tool_choice="auto" + 中文“你是谁？”连续 3 次
=> 泛化 AI 助手: 3 次；Claude: 0 次
```

### 分析

`finish` tool 最初看起来像强嫌疑点，因为它处在 agent workflow 的“提交答案”
位置。但补测显示，问题不是单独由 `finish` 这个名字决定：

- `final_answer` 仍可能触发 Claude。
- `submit_answer` 本次没有触发，但其他 schema 组合仍会触发。
- 只保留 `execute_code` 时也出现过 Claude。
- 不传 tools 或禁用 tool_choice 时，问题明显减少。

因此更合理的解释是：

```text
旧版泛化身份 prompt 没有定义“我是谁”。
DeepSeek 在 tools/tool_choice="auto" 的 agent 模式下，会补全到某些 Claude/Anthropic
身份模板。
```

也就是说，tools schema 和 `tool_choice="auto"` 是诱发因素，旧版泛化 prompt 是
缺少约束的根因之一。当前 `OPENAI_*` endpoint 在同样条件下没有出现 Claude，
说明这不是 SWE-Vision provider 路由错误，也不是 tools schema 必然导致的问题。

扩展复测进一步说明：

- `reasoning high` 不是必要条件；不传 reasoning 时 Claude 比例仍为 6/10。
- `tool_choice="auto"` 是强诱发因素；改为 `none` 后 DeepSeek 复测中 Claude 为 0。
- tools 本身也有诱发能力；无 system 但传 tools 时，Claude 出现 4/5。
- 英文身份问题更容易触发 Claude；旧 prompt + tools + auto 下，英文 5/5 为 Claude。
- 显式产品身份对 DeepSeek 很有效；中英文各 5 次均稳定返回 SWE-Vision。
- 当前 `OPENAI_*` endpoint 在同样矩阵下没有 Claude 漂移，但也没有稳定采用显式
  SWE-Vision 身份，说明不同 endpoint 对产品身份 prompt 的遵守程度不同。

### 设计反思

旧版 prompt 只写能力：

```text
You are an expert AI assistant with access to a stateful Jupyter notebook environment.
```

这会让模型在身份问题上自行补全。对于 SWE-Vision 这类产品化 agent，系统提示
应该同时定义能力和产品身份。合理目标是：

```text
我是 SWE-Vision，一个具备图像理解和 Jupyter Notebook 代码执行能力的 AI 助手。
```

底层模型供应商身份不应该成为用户层面的默认回答，除非用户明确询问底层模型或
API provider。

### 决策

建议将系统提示从泛化身份改为明确产品身份：

```text
You are SWE-Vision, an AI coding and vision assistant with access to a stateful
Jupyter notebook environment.
```

这是比单纯改名 `finish` tool 更稳定的修复方向。

可选优化：

- 对身份/闲聊类问题考虑不启用 `tool_choice="auto"`。
- 继续观察 DeepSeek 在复杂任务中的 tool 调用行为。
- 如果后续仍出现身份漂移，再考虑改名 `finish` 或调整 tool schema。
