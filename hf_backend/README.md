# HF Constraint 后端

`/hf_backend` 现在不仅能做拼音约束推理，还提供了：

- 官方 `Qwen/Qwen3.5-4B-Base` 下载脚手架
- 面向输入法场景的拼音 SFT 数据准备
- 8GB 显存可跑的 QLoRA 微调入口
- 预测/反馈遥测与后续反馈再训练数据导出

---

## 1. 环境准备

```powershell
cd C:\Users\Felix\CLionProjects\Wisdom-Weasel\hf_backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

要求：

- Python 3.10+
- 建议 CUDA GPU；当前仓库默认面向 8GB RTX 4060 Ti

---

## 2. 当前配置说明

默认 `config.json` 已切到官方模型：

```json
{
  "model_id": "Qwen/Qwen3.5-4B-Base",
  "adapter_path": null,
  "model_kwargs": {
    "quantization_config": {
      "load_in_4bit": true,
      "bnb_4bit_compute_dtype": "bfloat16",
      "bnb_4bit_quant_type": "nf4",
      "bnb_4bit_use_double_quant": true
    }
  },
  "telemetry": {
    "enabled": true,
    "directory": "telemetry"
  }
}
```

新增关键字段：

- `adapter_path`：LoRA/PEFT 适配器目录；训练完后填这里即可热切
- `merge_adapter_on_load`：是否启动时把适配器并入基座
- `prompting`：统一训练/推理 prompt 模板
- `telemetry`：预测与反馈日志目录

---

## 3. 一键准备模型 + 语料 + 训练数据

推荐直接用 PowerShell 包装脚本：

```powershell
cd C:\Users\Felix\CLionProjects\Wisdom-Weasel\hf_backend
.\Bootstrap-Qwen35Pinyin.ps1
```

它会依次执行：

1. 安装 Python 依赖
2. 下载 `Qwen/Qwen3.5-4B-Base`
3. 下载公开拼音语料 `Duyu/Pinyin-Hanzi`
4. 生成输入法场景 SFT 数据
5. 导出反馈数据集（如果已有遥测）

可选参数：

```powershell
.\Bootstrap-Qwen35Pinyin.ps1 -ModelDir "D:\models\Qwen3.5-4B-Base" -MaxSeedRows 250000 -MaxExamples 300000
.\Bootstrap-Qwen35Pinyin.ps1 -Train
.\Bootstrap-Qwen35Pinyin.ps1 -SkipModelDownload
```

---

## 4. 单独执行各阶段

### 4.1 下载官方模型

```powershell
& .\.venv\Scripts\python.exe .\scripts\download_model.py --model-id Qwen/Qwen3.5-4B-Base --local-dir D:\models\Qwen3.5-4B-Base
```

### 4.2 下载公开拼音语料

```powershell
& .\.venv\Scripts\python.exe .\scripts\download_seed_data.py --dataset-id Duyu/Pinyin-Hanzi --max-rows 250000
```

### 4.3 生成输入法场景训练集

```powershell
& .\.venv\Scripts\python.exe .\scripts\prepare_training_data.py --max-examples 300000
```

训练集会混合三类数据：

- 公开拼音语料切出的“上下文 -> 当前拼音 -> 汉字补全”样本
- 仓库内 README / CHANGELOG / 预置短语挖出的项目领域短语
- 已记录的用户反馈 SFT 样本

### 4.4 从反馈日志导出增量数据

```powershell
& .\.venv\Scripts\python.exe .\scripts\export_feedback_dataset.py
```

会得到：

- `data/processed/feedback_sft.jsonl`
- `data/processed/feedback_pairs.jsonl`

---

## 5. QLoRA 微调

样例配置在：

- `C:\Users\Felix\CLionProjects\Wisdom-Weasel\hf_backend\training\train_config.sample.json`

启动训练：

```powershell
& .\.venv\Scripts\python.exe .\scripts\train_qlora.py
```

或覆盖输出目录：

```powershell
& .\.venv\Scripts\python.exe .\scripts\train_qlora.py --output-dir D:\models\WisdomWeasel-Qwen35-Lora
```

默认策略：

- 基座：`Qwen/Qwen3.5-4B-Base`
- 量化：4bit NF4
- 微调：QLoRA
- 目标模块：`q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj`
- 面向 8GB 卡的 batch/grad accumulation 预设

训练完成后，把适配器目录填回 `config.json`：

```json
{
  "adapter_path": "D:/models/WisdomWeasel-Qwen35-Lora"
}
```

---

## 6. 启动推理服务

```powershell
.\Start-HFConstraintBackend.ps1
```

停止：

```powershell
.\Stop-HFConstraintBackend.ps1
```

---

## 7. API

### `GET /health`

返回当前模型、适配器、遥测开关状态。

### `POST /v1/generate/completions`

请求：

```json
{
  "prompt": "最近在调输入法候选排序",
  "pinyin_constraints": ["pin", "yin"],
  "metadata": {
    "source": "weasel"
  }
}
```

响应：

```json
{
  "request_id": "7d6f...",
  "responses": "拼音 频音",
  "candidates": ["拼音", "频音"],
  "latency_ms": 38.214
}
```

`responses` 字段保留，兼容现有 C++ 侧解析逻辑。

### `POST /v1/feedback`

用于记录“用户是否接受了模型输出”：

```json
{
  "request_id": "7d6f...",
  "prompt": "最近在调输入法候选排序",
  "pinyin_constraints": ["pin", "yin"],
  "shown_candidates": ["拼音", "频音"],
  "selected_candidate": "拼音",
  "accepted": true,
  "metadata": {
    "client": "weasel"
  }
}
```

### `GET /v1/telemetry/summary`

返回：

- 接受率
- Top1 / Top3 / Top5 命中率
- MRR
- 用户期望结果是否落在候选列表里

这就是后续“模型有没有输出用户真正想要的结果”的基础遥测。

---

## 8. 反馈闭环建议

推荐流程：

1. 线上推理记录 `predictions.jsonl`
2. 用户确认/纠正时记录 `feedback.jsonl`
3. 定期运行：
   - `export_feedback_dataset.py`
   - `prepare_training_data.py`
   - `train_qlora.py`
4. 新适配器通过 `adapter_path` 热切到后端

如果后面要接入 C++ 端遥测，只需要把 `request_id + shown_candidates + selected_candidate` 回传到 `/v1/feedback` 即可。
