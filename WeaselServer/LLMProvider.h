#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

enum class LLMRequestType : uint8_t {
  NoInputPrediction = 0,
  PinyinConstrainedPrediction = 1,
  RimeReorder = 2,
};

struct LLMRequest {
  LLMRequestType type = LLMRequestType::NoInputPrediction;
  std::wstring context;          // 最近上下文
  std::wstring current_input;    // 当前拼音/输入串
  std::vector<std::wstring> rime_candidates;  // Rime 原始候选，仅 Rime 重排使用
  size_t max_candidates = 5;
};

using LLMPartialCallback =
    std::function<bool(const std::vector<std::wstring>& candidates)>;

namespace llm_request {

struct InstructPrompt {
  std::wstring system_prompt;
  std::wstring user_prompt;
};

bool IsExecutable(const LLMRequest& request);
std::wstring GetRequestTypeName(LLMRequestType type);
size_t GetOutputLimit(const LLMRequest& request);
std::wstring JoinCandidatesForPrompt(
    const std::vector<std::wstring>& candidates);
InstructPrompt BuildInstructPrompt(const LLMRequest& request);
std::wstring BuildCompactPrompt(const LLMRequest& request);
std::wstring BuildBaseCompletionPrompt(const LLMRequest& request);
std::vector<std::string> BuildPinyinConstraintParts(
    const LLMRequest& request);

}  // namespace llm_request

// LLM提供者抽象基类
class LLMProvider {
 public:
  virtual ~LLMProvider() = default;

  // 从配置文件加载配置
  virtual bool LoadConfig(const std::string& config_name) = 0;

  // 执行统一请求，由主程序提前分流请求类型
  virtual std::vector<std::wstring> ExecuteRequest(
      const LLMRequest& request,
      const LLMPartialCallback& on_partial = nullptr) = 0;

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
  std::vector<std::wstring> ExecuteRequest(
      const LLMRequest& request,
      const LLMPartialCallback& on_partial = nullptr) override;
  bool IsAvailable() const override;
  std::string GetProviderName() const override { return "OpenAI Compatible"; }

 private:
  // 执行HTTP请求
  bool ExecuteHttpRequest(const std::string& url,
                          const std::string& request_body,
                          size_t max_candidates,
                          const LLMPartialCallback& on_partial,
                          std::string& response_body);
  bool ExecuteOllamaGenerateRequest(const std::string& url,
                                    const std::string& request_body,
                                    const std::string& prompt_prefix_utf8,
                                    size_t max_candidates,
                                    const LLMPartialCallback& on_partial,
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
  std::string m_extra_body_json;
  std::vector<std::pair<std::string, std::string>> m_extra_headers;
  void* m_hSession;          // HINTERNET，复用的 WinHTTP 会话
  void* m_hConnect;          // HINTERNET，复用的连接
  std::string m_cached_url;  // 当前连接对应的 URL，变化时重建连接
};

// llama.cpp 本地推理提供者
class LlamaCppProvider : public LLMProvider {
 public:
  LlamaCppProvider();
  ~LlamaCppProvider() override;

  bool LoadConfig(const std::string& config_name) override;
  std::vector<std::wstring> ExecuteRequest(
      const LLMRequest& request,
      const LLMPartialCallback& on_partial = nullptr) override;
  bool IsAvailable() const override;
  std::string GetProviderName() const override { return "llama.cpp Local"; }

 private:
  // 初始化模型
  bool InitializeModel();
  // 清理资源
  void Cleanup();
  // 生成文本
  std::string GenerateText(const std::string& prompt, size_t max_tokens);
  // 批量采样：n_parallel 条序列并行，每条只生成一个词（最多 max_new_tokens 个
  // token）；与单次生成一样复用 system 的 KV cache
  std::vector<std::string> GenerateCandidatesBatch(
      const std::string& system_prompt_utf8,
      const std::string& user_prompt_utf8,
      size_t n_parallel,
      int max_new_tokens);
  // 预处理并缓存 system prompt 的 KV 状态
  bool PrepareSystemPrompt(const std::string& system_prompt_utf8);

  bool m_enabled;
  std::string m_model_path;  // 模型文件路径
  int m_n_ctx;               // 上下文大小
  int m_n_gpu_layers;        // GPU层数
  int m_max_tokens;          // 最大生成token数
  double m_temperature;      // 温度参数
  int m_n_threads;           // 线程数
  bool m_instruct_model;  // true=Instruct 使用指令 prompt，false=Base 仅用
                          // context 补全

  // llama.cpp 对象（使用前向声明避免包含头文件）
  void* m_model;    // llama_model*
  void* m_context;  // llama_context*
  void* m_sampler;  // llama_sampler*
  void* m_memory;   // llama_memory_t (缓存，避免每次获取)
  void* m_vocab;    // const llama_vocab* (缓存，避免每次获取)
  int m_ctx_size;   // 实际上下文大小（缓存，避免每次获取）
  std::string m_system_prompt_utf8;
  std::vector<uint8_t> m_system_state;
  size_t m_system_state_size;
  bool m_system_prompt_ready;
  bool m_model_loaded;  // 模型是否已加载
};

// HF Constraint 接口提供者（/v1/generate/completions）
// 统一请求会转换为 {"prompt": "...", "pinyin_constraints": [...]} 形式
class HFConstraintProvider : public LLMProvider {
 public:
  HFConstraintProvider();
  ~HFConstraintProvider() override;
  bool LoadConfig(const std::string& config_name) override;
  std::vector<std::wstring> ExecuteRequest(
      const LLMRequest& request,
      const LLMPartialCallback& on_partial = nullptr) override;
  bool IsAvailable() const override;
  std::string GetProviderName() const override { return "HF Constraint"; }

 private:
  bool ExecuteHttpRequest(const std::string& url,
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
