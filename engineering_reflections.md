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

### 背景

在将 SWE-Vision 的 Web/CLI 调用切换到 DeepSeek 官方 API 后，前端右上角显示的模型为 `deepseek-v4-pro`，请求也实际发往 `https://api.deepseek.com`，但用户询问“你是谁？”时，模型回答：

> 我是 Claude，一个由 Anthropic 开发的 AI 助手。

这引发了一个问题：这是 provider 切换代码出错，还是 DeepSeek 模型在当前 agent 场景下出现身份幻觉？

### 现象

CLI 输出显示：

```text
Using model: deepseek-v4-pro
Using base URL: https://api.deepseek.com
Using provider: deepseek
HTTP Request: POST https://api.deepseek.com/chat/completions
```

说明请求确实发到了 DeepSeek 官方 endpoint，不是旧的第三方中转，也不是 OpenRouter 或 Anthropic。

直接用 `.env` 中的 `DEEPSEEK_API_KEY` 和 `DEEPSEEK_BASE_URL` 发最小请求时，模型正常回答：

```text
我是由深度求索公司创造的AI助手DeepSeek。
```

但使用 SWE-Vision agent 的系统提示和 tools 后，模型会回答 Claude。

### 初始假设

当时主要怀疑过：

- provider 自动识别错误，仍然请求了旧的 `OPENAI_BASE_URL`。
- Web 前端只改了右上角显示，但后端仍使用旧配置。
- DeepSeek API key/base URL 没有被项目读取。
- `reasoning_effort` 或 DeepSeek thinking 参数导致错误。
- SWE-Vision 的 system prompt 或 tools schema 诱发了模型身份幻觉。

### 排查过程

关键测试矩阵：

```text
无 system、无 tools、无 reasoning
=> DeepSeek

SWE-Vision 原始 system、无 tools
=> AI 编程助手 / 拥有 Jupyter Notebook 能力的助手

SWE-Vision agent system、无 tools
=> 拥有代码执行能力的 AI 助手

无 system、有 tools
=> 倾向调用工具，不直接回答身份

SWE-Vision agent system、有 tools、无 reasoning
=> Claude

SWE-Vision agent system、有 tools、reasoning high
=> Claude

只给 execute_code tool
=> 泛化 AI 助手，不是 Claude

只给 finish tool
=> Claude

两个 tools 都给，但 tool_choice="none"
=> DeepSeek
```

### 结论

这不是 provider 路由错误。provider、base URL 和 response model 都指向 DeepSeek。

更可能的问题是：

```text
DeepSeek v4-pro + Chat Completions tools + tool_choice="auto" + finish 工具
```

这个组合诱发了模型的身份幻觉。

其中 `finish` tool 是较强嫌疑点，因为只给 `finish` tool 时也会触发 Claude 身份，而只给 `execute_code` tool 时没有明显触发。

这个现象可以作为“DeepSeek 在某些 agent/tool-calling 场景中存在 Claude/Anthropic 身份模板残留”的弱佐证，但不能直接证明存在对 Claude 的蒸馏。

### 反思

SWE-Vision 的 `SYSTEM_PROMPT` 当前只写了：

```text
You are an expert AI assistant with access to a stateful Jupyter notebook environment.
```

这个身份定义太泛。模型被问“你是谁？”时，会自己补全身份。

以前模型返回：

```text
我是您的 AI 编程助手，拥有访问有状态 Jupyter Notebook 环境的能力。
```

这其实是对 system prompt 的正常复述。

DeepSeek 在 tools 模式下返回 Claude，则说明泛化身份提示不足以压住模型内部或训练数据中的身份模板。

从产品角度，用户问“你是谁？”时，合理答案应该是：

```text
我是 SWE-Vision，一个具备图像理解和 Jupyter Notebook 代码执行能力的 AI 助手。
```

而不是底层模型厂商身份，更不应该是 Claude。

### 后续行动

可以考虑：

- 在 system prompt 中显式定义产品身份。
- 将 `finish` tool 改名为更中性的 `final_answer` 或 `submit_answer`。
- 测试不同 tool schema 是否还触发 Claude 身份。
- 对 DeepSeek provider 单独考虑是否需要在简单身份/闲聊问题上避免 `tool_choice="auto"`。
- 对比 `deepseek-v4-pro`、`deepseek-v4-flash`、旧 `deepseek-reasoner`。
- 对比中文和英文身份问题。
- 对比不同 provider：Qwen、MiniMax、OpenRouter 上同样 tools schema。
- 多次采样统计 Claude 身份出现频率。

当前阶段最可靠的判断：

```text
Provider 切换和 DeepSeek API 调用路径没有明显错误。
Claude 回答来自 DeepSeek 在当前 SWE-Vision tools/agent 上下文中的身份幻觉。
问题需要从 system prompt 身份约束和 tool schema 设计两个方向继续处理。
```
