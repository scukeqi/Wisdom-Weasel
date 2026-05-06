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

**直接使用（推荐）**

    从 [Releases](https://github.com/scukeqi/Wisdom-Weasel/releases) 下载适合设备的安装包，步骤如下：

1. 解压安装包后运行 `WeaselSetup.exe` 完成安装；
2. 配置初始拼音输入方案（如：[雾凇拼音](https://github.com/iDvel/rime-ice) 等）；
3. 在 `weasel.yaml` 中启用并配置 LLM（参考「LLM 配置说明」）；
4. 右键托盘「小狼毫输入法」图标，选择「重新部署」使配置生效。
- **安装包版本与显卡适配说明**
  
  如果选择使用llama.cpp 本地（`provider_type: llamacpp`）则须注意选择版本
  
  Releases 中提供 **CUDA / SYCL / Vulkan** 版本，适配不同品牌显卡：
  
  - **CUDA**：**NVIDIA**显卡(支持cuda的型号)；
  
  - **SYCL** ：适配 **Intel** 、**AMD** 显卡(支持的sycl型号)；
  
  - **Vulkan** ：通用，支持大部分集成/独立显卡，选择此版本兜底。

**从源码构建**

    若需自定义开发，可从源码构建：

1. 运行 `build.bat x64` 执行构建；
2. 依赖与官方 Weasel 一致（如 Boost、librime、yaml-cpp 等，见项目与 `weasel.props`）；
3. 若使用 `llamacpp` 后端，需从 [llamacpp](https://github.com/ggml-org/llama.cpp/releases) 获取对应 dll 文件并放置到指定目录。

---

## 使用说明

- 安装/构建完成后，在键盘布局中选择「小狼毫」即可正常使用；
- 右键托盘输入法指示器「中」或「A」图标，可访问「程序文件夹」「用户文件夹」「重新部署」「输入法设置」等功能；
- 修改 `weasel.yaml` 中 LLM 等相关配置后，需「重新部署」或重启 Weasel 服务才能生效。
- 生成异常时可以双击~键清空上下文

---

## LLM 配置说明

LLM配置写在 `Rime 程序文件夹\data\weasel.yaml`中。

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

## 致谢与许可证

- 本分支在 [Rime 小狼毫（Weasel）](https://github.com/rime/weasel) 基础上开发，致谢原项目作者与社区。  
- 输入方案与程序设计、美术、引用开源软件等说明见 [原仓库](https://github.com/rime/weasel)。  
- **许可证**：GPLv3（与 Weasel 一致）。  
- 项目主页：https://rime.im  

---

## 问题与反馈

- 本分支（LLM 预测功能、构建与配置）相关问题，请在本仓库 [Issues](https://github.com/scukeqi/Wisdom-Weasel/issues) 中反馈。  
- Rime 输入法通用问题（方案、词库、部署等），请反馈至 [Rime 之家](https://github.com/rime/home/issues)。  
- 欢迎提交 Pull Request。

谢谢使用。
