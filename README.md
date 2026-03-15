# Wisdom-Weasel

基于 [Rime 小狼毫（Weasel）](https://github.com/rime/weasel) 开源输入法，增加 **基于大语言模型（LLM）的智能预测** 功能：在保留 Rime 全套方案与词库的前提下，用 LLM 根据当前输入与历史上下文生成候选词，支持本地推理与云端 API 多种后端。

---
![demo](demo.gif)
## 功能特性

- **多后端 LLM 预测**
  - **OpenAI 兼容**（`provider_type: openai`）：任意 OpenAI 兼容 API（如 OpenAI、Ollama、本地 openai-api 等），通过 `llm/openai/` 配置。
  - **llama.cpp 本地**（`provider_type: llamacpp`）：本地 GGUF 模型，内置 llama.cpp 推理，通过 `llm/llamacpp/` 配置。
  - **HF Constraint**（`provider_type: hf_constraint`）：支持拼音约束生成Python后端，通过 `llm/hf_constraint/` 配置。
- **上下文历史**：维护用户最近输入词序列，作为 LLM 预测的上下文。
- **可选记忆压缩**：历史超过容量时，可异步调用单独配置的 LLM（`llm/memory/`）将旧词压缩为摘要，节省上下文长度。
- **与 Rime 并存**：预测候选与 Rime 方案候选一起展示，不改变原有方案、词库与部署流程。

---

## 系统要求

- **操作系统**：Windows 8.1 ~ Windows 11  
- **运行环境**：与官方小狼毫相同（需先安装/部署 Rime）  
- **LLM 可选**：  
  - 使用 `openai` 时需可访问的 API 或本地服务（如 Ollama）；  
  - 使用 `llamacpp` 时需本机加载 GGUF 模型（建议 4GB+ 显存或足够内存）；  
  - 使用 `hf_constraint` 需要创建Python环境，`hf_backend\requirements.txt`。

---

## 安装与构建

- **直接使用**：从 [Releases](https://github.com/scukeqi/Wisdom-Weasel/releases)下载安装包，安装后与官方小狼毫一样使用。
  - 配置初始拼音输入方案,如：[雾凇拼音](https://github.com/iDvel/rime-ice)等。
  - `weasel.yaml` 中启用并配置 LLM 。
- **从源码构建**：  
  - 运行 `build.bat x64`  
  - 依赖与官方 Weasel 一致（如 Boost、librime、yaml-cpp 等，见项目与 `weasel.props`）。  
  - 若使用 `llamacpp`，需要从[llamacpp](https://github.com/ggml-org/llama.cpp/releases)获取dll。

---

## LLM 配置说明

配置写在 **Rime 用户目录** 下的 `weasel.yaml`中。
### 总开关与提供者类型

```yaml
llm:
  enabled: true
  provider_type: openai   # 可选: openai | llamacpp | hf_constraint
```

### 1. OpenAI 兼容（`provider_type: openai`）

适用于 OpenAI、Ollama、本地 openai-api 等：

```yaml
llm:
  enabled: true
  provider_type: openai
  openai:
    api_url: "https://api.openai.com/v1/chat/completions"   # 或 Ollama 等地址
    api_key: "your-api-key"   # 本地服务可留空
    model: "gpt-3.5-turbo"
    max_tokens: 20
    temperature: "0.6"
```

### 2. llama.cpp 本地（`provider_type: llamacpp`）

本地 GGUF 模型，无需额外服务：

```yaml
llm:
  enabled: true
  provider_type: llamacpp
  llamacpp:
    model_path: "D:/models/your_model.gguf"
    n_ctx: 2048
    n_gpu_layers: -1
    max_tokens: 8
    temperature: "0.6"
    n_threads: 4
    model_type: "Instruct"   # 或 推荐"Base" 
```

### 3. HF Constraint（`provider_type: hf_constraint`）
[hf_backend 详细配置步骤](https://github.com/scukeqi/Wisdom-Weasel/blob/main/hf_backend/README.md)

拼音约束接口，请求体形如：`{"prompt": "历史上下文", "pinyin_constraints": ["当前输入"]}`：

```yaml
llm:
  enabled: true
  provider_type: hf_constraint
  hf_constraint:
    api_url: "http://localhost:8000/v1/generate/completions"
```
推荐使用Base模型
### 可选：记忆压缩（`llm/memory/`）

当上下文历史超过容量时，可用单独配置的 LLM 将旧词压缩为摘要（与预测用 LLM 分离）：

```yaml
llm:
  memory:
    enabled: true
    api_url: "https://api.openai.com/v1/chat/completions"
    api_key: "your-api-key"
    model: "gpt-3.5-turbo"
    max_tokens: 200
```

不配置或 `enabled: false` 时，不进行记忆压缩，仅使用固定长度的最近上下文。

### 启用调试日志终端输出

```yaml
dev_console:
  enabled: true
```
---

## 使用说明

- 安装/构建完成后，与官方小狼毫相同：在输入法指示器中选择【中】图标即可使用。  
- 通过右键托盘图标 **小狼毫输入法** 可访问「用户文件夹」「重新部署」等。  
- 修改 `weasel.yaml` 中 LLM 相关配置后，需要 **重新部署** 或重启 Weasel 服务后生效。  


---

## 致谢与许可证

- 本分支在 [Rime 小狼毫（Weasel）](https://github.com/rime/weasel) 基础上开发，致谢原项目作者与社区。  
- 输入方案与程序设计、美术、引用开源软件等说明见 [原仓库](https://github.com/rime/weasel)。  
- **许可证**：GPLv3（与 Weasel 一致）。  
- 项目主页：https://rime.im  

---

## 问题与反馈

- 本分支（LLM 预测功能、构建与配置）相关问题，请在本仓库 [Issues](https://github.com/scukeqi/Wisdom-Weasel/issues)（请将 `YOUR_USERNAME` 改为你的 GitHub 用户名） 中反馈。  
- Rime 输入法通用问题（方案、词库、部署等），请反馈至 [Rime 之家](https://github.com/rime/home/issues)。  
- 欢迎提交 Pull Request。

谢谢使用。
