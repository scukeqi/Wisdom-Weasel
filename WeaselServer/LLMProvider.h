#pragma once

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

// LLM提供者抽象基类
class LLMProvider {
 public:
  virtual ~LLMProvider() = default;

  // 从配置文件加载配置
  virtual bool LoadConfig(const std::string& config_name) = 0;

  // 预测候选词
  // context: 历史上下文
  // current_input: 当前输入（可为空）
  // max_candidates: 最大候选词数量
  virtual std::vector<std::wstring> PredictCandidates(
      const std::wstring& context,
      const std::wstring& current_input,
      size_t max_candidates) = 0;

  // 检查LLM是否可用
  virtual bool IsAvailable() const = 0;

  // 获取提供者名称
  virtual std::string GetProviderName() const = 0;
};

// OpenAI兼容接口提供者
class OpenAICompatibleProvider : public LLMProvider {
 public:
  OpenAICompatibleProvider();
  ~OpenAICompatibleProvider() override;

  bool LoadConfig(const std::string& config_name) override;
  std::vector<std::wstring> PredictCandidates(
      const std::wstring& context,
      const std::wstring& current_input,
      size_t max_candidates) override;
  bool IsAvailable() const override;
  std::string GetProviderName() const override { return "OpenAI Compatible"; }

 private:
  // 执行HTTP请求
  bool ExecuteRequest(const std::string& url,
                      const std::string& request_body,
                      std::string& response_body);
  // 解析JSON响应
  std::vector<std::wstring> ParseResponse(const std::string& json_response);
  void CloseConnection();  // 关闭并清空复用的 HTTP 连接

  bool m_enabled;
  std::string m_api_url;
  std::string m_api_key;
  std::string m_model;
  int m_max_tokens;
  double m_temperature;
  double m_top_p;
  double m_presence_penalty;
  double m_frequency_penalty;
  bool m_has_seed;
  int m_seed;
  std::string m_extra_body_json;  // 额外透传 JSON（对象字符串）
  void* m_hSession;       // HINTERNET，复用的 WinHTTP 会话
  void* m_hConnect;       // HINTERNET，复用的连接
  std::string m_cached_url;  // 当前连接对应的 URL，变化时重建连接
};

// llama.cpp 本地推理提供者
class LlamaCppProvider : public LLMProvider {
 public:
  LlamaCppProvider();
  ~LlamaCppProvider() override;

  bool LoadConfig(const std::string& config_name) override;
  std::vector<std::wstring> PredictCandidates(
      const std::wstring& context,
      const std::wstring& current_input,
      size_t max_candidates) override;
  bool IsAvailable() const override;
  std::string GetProviderName() const override { return "llama.cpp Local"; }

 private:
  // 初始化模型
  bool InitializeModel();
  // 清理资源
  void Cleanup();
  // 生成文本
  std::string GenerateText(const std::string& prompt, size_t max_tokens);
  // 批量采样：n_parallel 条序列并行，每条只生成一个词（最多 max_new_tokens 个 token）；与单次生成一样复用 system 的 KV cache
  std::vector<std::string> GenerateCandidatesBatch(const std::string& system_prompt_utf8, const std::string& user_prompt_utf8, size_t n_parallel, int max_new_tokens);
  // 预处理并缓存 system prompt 的 KV 状态
  bool PrepareSystemPrompt(const std::string& system_prompt_utf8);

  bool m_enabled;
  std::string m_model_path;      // 模型文件路径
  int m_n_ctx;                    // 上下文大小
  int m_n_gpu_layers;             // GPU层数
  int m_max_tokens;               // 最大生成token数
  double m_temperature;           // 温度参数
  int m_top_k;                    // Top-K 采样
  double m_top_p;                 // Top-P (nucleus) 采样
  double m_repeat_penalty;        // 重复惩罚
  double m_presence_penalty;      // 出现惩罚
  double m_frequency_penalty;     // 频率惩罚
  int m_mirostat;                 // 0=关闭, 1=mirostat v1, 2=mirostat v2
  double m_min_p;                 // 最小概率过滤
  double m_typical_p;             // typical sampling
  int m_n_threads;                // 线程数
  bool m_instruct_model;          // true=Instruct 使用指令 prompt，false=Base 仅用 context 补全

  // llama.cpp 对象（使用前向声明避免包含头文件）
  void* m_model;                  // llama_model*
  void* m_context;                // llama_context*
  void* m_sampler;                 // llama_sampler*
  void* m_memory;                  // llama_memory_t (缓存，避免每次获取)
  void* m_vocab;                   // const llama_vocab* (缓存，避免每次获取)
  int m_ctx_size;                  // 实际上下文大小（缓存，避免每次获取）
  std::string m_system_prompt_utf8;
  std::vector<uint8_t> m_system_state;
  size_t m_system_state_size;
  bool m_system_prompt_ready;
  bool m_model_loaded;             // 模型是否已加载
};

// HF Constraint 接口提供者（/v1/generate/completions）
// 请求体: {"prompt": "历史上下文", "pinyin_constraints": ["当前输入"]}
class HFConstraintProvider : public LLMProvider {
 public:
  HFConstraintProvider();
  ~HFConstraintProvider() override;  bool LoadConfig(const std::string& config_name) override;
  std::vector<std::wstring> PredictCandidates(
      const std::wstring& context,
      const std::wstring& current_input,
      size_t max_candidates) override;
  bool IsAvailable() const override;
  std::string GetProviderName() const override {
    return "HF Constraint";
  }

 private:
  bool ExecuteRequest(const std::string& url,
                      const std::string& request_body,
                      std::string& response_body);
  std::vector<std::wstring> ParseResponse(const std::string& json_response);
  void CloseConnection();  // 关闭并清空复用的 HTTP 连接

  bool m_enabled;
  std::string m_api_url;  // 默认 http://localhost:8000/v1/generate/completions
  void* m_hSession;       // HINTERNET，复用的 WinHTTP 会话
  void* m_hConnect;       // HINTERNET，复用的连接
  std::string m_cached_url;  // 当前连接对应的 URL，变化时重建连接
};
