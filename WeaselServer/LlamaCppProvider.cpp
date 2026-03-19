#include "stdafx.h"
#include "LLMProvider.h"
#include "DevConsole.h"
#include <WeaselUtility.h>
#include <rime_api.h>
#include <algorithm>
#include <cctype>
#include <sstream>

// 包含 llama.cpp 头文件
#include "llama.h"

namespace {

bool IsExecutableRequest(const LLMRequest& request) {
  switch (request.type) {
    case LLMRequestType::NoInputPrediction:
      return !request.context.empty() || !request.preference_hint.empty();
    case LLMRequestType::PinyinConstrainedPrediction:
      return !request.current_input.empty();
    case LLMRequestType::RimeReorder:
      return !request.rime_candidates.empty();
  }
  return false;
}

std::wstring GetRequestTypeName(LLMRequestType type) {
  switch (type) {
    case LLMRequestType::NoInputPrediction:
      return L"无输入预测";
    case LLMRequestType::PinyinConstrainedPrediction:
      return L"拼音约束预测";
    case LLMRequestType::RimeReorder:
      return L"Rime 重排";
  }
  return L"未知请求";
}

std::wstring JoinCandidatesForPrompt(
    const std::vector<std::wstring>& candidates) {
  std::wstring joined;
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (i > 0) {
      joined += L"\n";
    }
    joined += std::to_wstring(i + 1) + L". \"" + candidates[i] + L"\"";
  }
  return joined;
}

void BuildInstructPrompts(const LLMRequest& request,
                          std::wstring& system_prompt,
                          std::wstring& user_prompt) {
  const size_t output_limit = (std::min)(
      request.max_candidates,
      request.type == LLMRequestType::RimeReorder ? request.rime_candidates.size()
                                                  : request.max_candidates);

  switch (request.type) {
    case LLMRequestType::NoInputPrediction:
      system_prompt =
          L"你是一个智能中文输入法，请根据上下文预测接下来最可能出现的"
          + std::to_wstring(request.max_candidates) + L"个候选词。\n\n"
          L"要求：\n"
          L"1. 只返回候选词，不要任何解释或标点\n"
          L"2. 候选词之间用单个空格分隔\n"
          L"3. 按可能性从高到低排列\n"
          L"4. 确保候选词都是有效的中文词汇或常用短语\n"
          L"5. 返回词数严格不超过"
          + std::to_wstring(request.max_candidates) + L"个\n";
      if (!request.preference_hint.empty()) {
        system_prompt +=
            L"6. 用户偏好只作弱参考，必须优先保证与最近上下文的连贯性\n"
            L"7. 若用户偏好与当前上下文冲突，忽略用户偏好，以当前上下文为准\n";
      }
      user_prompt = L"上下文：\"" + request.context + L"\"\n";
      if (!request.preference_hint.empty()) {
        user_prompt +=
            L"用户偏好（弱参考，可忽略）：\"" + request.preference_hint + L"\"\n";
      }
      user_prompt += L"候选词：";
      return;
    case LLMRequestType::PinyinConstrainedPrediction:
      system_prompt =
          L"你是一个智能中文输入法，请根据上下文和当前拼音，预测接下来最可能出现的"
          + std::to_wstring(request.max_candidates) + L"个候选词。\n\n"
          L"要求：\n"
          L"1. 只返回候选词，不要任何解释或标点\n"
          L"2. 候选词之间用单个空格分隔\n"
          L"3. 按可能性从高到低排列\n"
          L"4. 每个候选词都必须严格匹配当前拼音约束\n"
          L"5. 如果上下文为空或无关，也必须优先满足拼音约束\n"
          L"6. 确保候选词都是有效的中文词汇或常用短语\n"
          L"7. 返回词数严格不超过"
          + std::to_wstring(request.max_candidates) + L"个\n";
      if (!request.preference_hint.empty()) {
        system_prompt +=
            L"8. 用户偏好只作弱参考，必须优先保证与最近上下文和当前拼音的连贯性\n"
            L"9. 若用户偏好与当前上下文冲突，忽略用户偏好，以当前上下文为准\n";
      }
      user_prompt = L"上下文：\"" + request.context + L"\"\n";
      if (!request.preference_hint.empty()) {
        user_prompt +=
            L"用户偏好（弱参考，可忽略）：\"" + request.preference_hint + L"\"\n";
      }
      user_prompt +=
          L"当前拼音：\"" + request.current_input + L"\"\n候选词：";
      return;
    case LLMRequestType::RimeReorder:
      system_prompt =
          L"你是一个智能中文输入法，请根据上下文、当前拼音和用户偏好，对给定的 Rime 候选词重新排序。\n\n"
          L"要求：\n"
          L"1. 只能从给定候选列表中选择，不得新增、改写或拆分候选词\n"
          L"2. 只返回重排后的候选词，不要解释、编号或标点\n"
          L"3. 候选词之间用单个空格分隔\n"
          L"4. 按更符合上下文的顺序输出\n"
          L"5. 若无法判断，尽量保持原顺序\n"
          L"6. 返回词数严格不超过"
          + std::to_wstring(output_limit) + L"个\n";
      user_prompt = L"上下文：\"" + request.context + L"\"\n";
      if (!request.preference_hint.empty()) {
        user_prompt +=
            L"用户偏好（弱参考，可忽略）：\"" + request.preference_hint + L"\"\n";
      }
      if (!request.current_input.empty()) {
        user_prompt += L"当前拼音：\"" + request.current_input + L"\"\n";
      }
      user_prompt +=
          L"Rime 候选词：\n" + JoinCandidatesForPrompt(request.rime_candidates)
          + L"\n重排结果：";
      return;
  }
}

std::wstring BuildBasePrompt(const LLMRequest& request) {
  switch (request.type) {
    case LLMRequestType::NoInputPrediction:
      if (!request.context.empty()) {
        return request.context;
      }
      return request.preference_hint;
    case LLMRequestType::PinyinConstrainedPrediction: {
      std::wstring prompt = request.context;
      if (prompt.empty() && !request.preference_hint.empty()) {
        prompt = request.preference_hint;
      }
      prompt += request.current_input;
      return prompt;
    }
    case LLMRequestType::RimeReorder: {
      const size_t output_limit = (std::min)(
          request.max_candidates, request.rime_candidates.size());
      std::wstring prompt =
          L"请根据以下上下文、当前拼音和用户偏好，对给定的 Rime 候选词重新排序。\n"
          L"只能从给定候选列表中选择，不得新增候选词；只返回重排后的前"
          + std::to_wstring(output_limit)
          + L"个候选词，使用空格分隔，不要解释。\n";
      prompt += L"上下文：\"" + request.context + L"\"\n";
      if (!request.preference_hint.empty()) {
        prompt +=
            L"用户偏好（弱参考，可忽略）：\"" + request.preference_hint + L"\"\n";
      }
      if (!request.current_input.empty()) {
        prompt += L"当前拼音：\"" + request.current_input + L"\"\n";
      }
      prompt +=
          L"Rime 候选词：\n" + JoinCandidatesForPrompt(request.rime_candidates)
          + L"\n重排结果：";
      return prompt;
    }
  }
  return std::wstring();
}

}  // namespace

LlamaCppProvider::LlamaCppProvider()
    : m_enabled(false),
      m_n_ctx(2048),
      m_n_gpu_layers(0),
      m_max_tokens(10),
      m_temperature(0.7),
      m_n_threads(4),
      m_instruct_model(true),
      m_model(nullptr),
      m_context(nullptr),
      m_sampler(nullptr),
      m_memory(nullptr),
      m_vocab(nullptr),
      m_ctx_size(0),
      m_system_prompt_utf8(),
      m_system_state(),
      m_system_state_size(0),
      m_system_prompt_ready(false),
      m_model_loaded(false) {
  // 初始化 llama backend（只需要调用一次）
  static bool backend_initialized = false;
  if (!backend_initialized) {
    llama_backend_init();
    ggml_backend_load_all();
    backend_initialized = true;
  }
}

LlamaCppProvider::~LlamaCppProvider() {
  Cleanup();
}

bool LlamaCppProvider::LoadConfig(const std::string& config_name) {
  extern DevConsole* g_dev_console;

  RimeApi* rime_api = rime_get_api();
  if (!rime_api) {
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] LoadConfig失败: rime_api未初始化");
    }
    return false;
  }

  if (g_dev_console && g_dev_console->IsEnabled()) {
    g_dev_console->WriteLine(L"[LLM] 开始从配置文件加载 llama.cpp 配置: " + u8tow(config_name));
  }

  RimeConfig config = {NULL};
  if (!rime_api->config_open(config_name.c_str(), &config)) {
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] LoadConfig失败: 无法打开配置文件 " + u8tow(config_name));
    }
    return false;
  }

  // 读取LLM配置
  Bool enabled = false;
  bool found_enabled = rime_api->config_get_bool(&config, "llm/enabled", &enabled);

  if (found_enabled) {
    m_enabled = !!enabled;
  } else {
    m_enabled = false;
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] LoadConfig失败: 未找到配置项 llm/enabled");
    }
    rime_api->config_close(&config);
    return false;
  }

  if (!m_enabled) {
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] LoadConfig失败: llm/enabled 为 false");
    }
    rime_api->config_close(&config);
    return false;
  }

  // 读取 llama.cpp 配置
  const int BUF_SIZE = 512;
  char buffer[BUF_SIZE + 1] = {0};

  // 模型路径（必需）
  bool found_model_path = rime_api->config_get_string(&config, "llm/llamacpp/model_path", buffer, BUF_SIZE);
  if (found_model_path) {
    m_model_path = buffer;
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] 找到配置项 llm/llamacpp/model_path = " + u8tow(m_model_path));
    }
  } else {
    m_model_path = "";
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] 未找到配置项 llm/llamacpp/model_path，这是必需的");
    }
    rime_api->config_close(&config);
    return false;
  }

  // 上下文大小（可选，默认2048）
  int n_ctx = 2048;
  if (rime_api->config_get_int(&config, "llm/llamacpp/n_ctx", &n_ctx)) {
    m_n_ctx = n_ctx;
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] 找到配置项 llm/llamacpp/n_ctx = " + std::to_wstring(m_n_ctx));
    }
  } else {
    m_n_ctx = 2048;
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] 未找到配置项 llm/llamacpp/n_ctx，使用默认值 = " + std::to_wstring(m_n_ctx));
    }
  }

  // GPU层数（可选，默认0，即CPU推理）
  int n_gpu_layers = 0;
  if (rime_api->config_get_int(&config, "llm/llamacpp/n_gpu_layers", &n_gpu_layers)) {
    m_n_gpu_layers = n_gpu_layers;
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] 找到配置项 llm/llamacpp/n_gpu_layers = " + std::to_wstring(m_n_gpu_layers));
    }
  } else {
    m_n_gpu_layers = 0;
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] 未找到配置项 llm/llamacpp/n_gpu_layers，使用默认值 = " + std::to_wstring(m_n_gpu_layers));
    }
  }

  // 最大token数（可选，默认10）
  int max_tokens = 10;
  if (rime_api->config_get_int(&config, "llm/llamacpp/max_tokens", &max_tokens)) {
    m_max_tokens = max_tokens;
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] 找到配置项 llm/llamacpp/max_tokens = " + std::to_wstring(m_max_tokens));
    }
  } else {
    m_max_tokens = 10;
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] 未找到配置项 llm/llamacpp/max_tokens，使用默认值 = " + std::to_wstring(m_max_tokens));
    }
  }

  // 温度参数（可选，默认0.7）
  char temp_str[64] = {0};
  if (rime_api->config_get_string(&config, "llm/llamacpp/temperature", temp_str, sizeof(temp_str) - 1)) {
    m_temperature = atof(temp_str);
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] 找到配置项 llm/llamacpp/temperature = " + std::to_wstring(m_temperature));
    }
  } else {
    m_temperature = 0.7;
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] 未找到配置项 llm/llamacpp/temperature，使用默认值 = " + std::to_wstring(m_temperature));
    }
  }

  // 线程数（可选，默认4）
  int n_threads = 4;
  if (rime_api->config_get_int(&config, "llm/llamacpp/n_threads", &n_threads)) {
    m_n_threads = n_threads;
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] 找到配置项 llm/llamacpp/n_threads = " + std::to_wstring(m_n_threads));
    }
  } else {
    m_n_threads = 4;
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] 未找到配置项 llm/llamacpp/n_threads，使用默认值 = " + std::to_wstring(m_n_threads));
    }
  }

  // 模型类型：Base 或 Instruct（可选，默认 Instruct）
  // Instruct 使用指令式 prompt；Base 直接使用 context 补全，无额外指令
  char model_type_buf[32] = {0};
  if (rime_api->config_get_string(&config, "llm/llamacpp/model_type", model_type_buf, sizeof(model_type_buf) - 1)) {
    std::string model_type_str(model_type_buf);
    std::transform(model_type_str.begin(), model_type_str.end(), model_type_str.begin(), ::tolower);
    m_instruct_model = (model_type_str != "base");
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] 找到配置项 llm/llamacpp/model_type = " + u8tow(model_type_buf) +
          L" -> " + (m_instruct_model ? L"Instruct" : L"Base"));
    }
  } else {
    m_instruct_model = true;
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] 未找到配置项 llm/llamacpp/model_type，使用默认值 Instruct");
    }
  }

  rime_api->config_close(&config);

  // 初始化模型
  if (!InitializeModel()) {
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] 模型初始化失败");
    }
    return false;
  }

  if (g_dev_console && g_dev_console->IsEnabled()) {
    g_dev_console->WriteLine(L"[LLM] LoadConfig: llama.cpp 配置加载成功");
  }

  return true;
}

bool LlamaCppProvider::InitializeModel() {
  extern DevConsole* g_dev_console;

  // 如果模型已加载，先清理
  if (m_model_loaded) {
    Cleanup();
  }

  if (m_model_path.empty()) {
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] 模型路径为空，无法初始化");
    }
    return false;
  }

  if (g_dev_console && g_dev_console->IsEnabled()) {
    g_dev_console->WriteLine(L"[LLM] 开始加载模型: " + u8tow(m_model_path));
  }

  // 设置日志回调（所有日志输出到开发终端）
  llama_log_set([](enum ggml_log_level level, const char* text, void* /* user_data */) {
    extern DevConsole* g_dev_console;
    if (g_dev_console && g_dev_console->IsEnabled()) {
      std::string text_str(text);
      std::wstring text_w = u8tow(text_str);
      // 根据日志级别添加前缀
      std::wstring prefix = L"[LLM llama.cpp] ";
      switch (level) {
        case GGML_LOG_LEVEL_ERROR:
          prefix = L"[LLM llama.cpp ERROR] ";
          break;
        case GGML_LOG_LEVEL_WARN:
          prefix = L"[LLM llama.cpp WARN] ";
          break;
        case GGML_LOG_LEVEL_INFO:
          prefix = L"[LLM llama.cpp INFO] ";
          break;
        default:
          prefix = L"[LLM llama.cpp] ";
          break;
      }
      g_dev_console->WriteLine(prefix + text_w);
    }
  }, nullptr);

  // 加载模型
  llama_model_params model_params = llama_model_default_params();
  model_params.n_gpu_layers = m_n_gpu_layers;

  llama_model* model = llama_model_load_from_file(m_model_path.c_str(), model_params);
  if (!model) {
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] 无法加载模型: " + u8tow(m_model_path));
    }
    return false;
  }

  m_model = model;

  // 初始化上下文
  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.n_ctx = m_n_ctx;
  ctx_params.n_batch = m_n_ctx;
  ctx_params.n_threads = m_n_threads;
  ctx_params.n_threads_batch = m_n_threads;
  // 批量解码需要多序列，seq_id 会用到 0..n_parallel-1，默认 1 会导致 "seq_id >= n_seq_max" 报错
  ctx_params.n_seq_max = 64;

  llama_context* ctx = llama_init_from_model(model, ctx_params);
  if (!ctx) {
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] 无法创建上下文");
    }
    llama_model_free(model);
    m_model = nullptr;
    return false;
  }

  m_context = ctx;

  // 初始化采样器
  llama_sampler_chain_params smpl_params = llama_sampler_chain_default_params();
  llama_sampler* smpl = llama_sampler_chain_init(smpl_params);
  
  // 添加 min_p 采样器
  llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
  
  // 添加温度采样器
  llama_sampler_chain_add(smpl, llama_sampler_init_temp((float)m_temperature));
  
  // 添加随机采样器
  llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

  m_sampler = smpl;

  // 缓存常用对象，避免在 GenerateText 中重复获取
  m_memory = (void*)llama_get_memory(ctx);
  m_vocab = (void*)llama_model_get_vocab(model);
  m_ctx_size = llama_n_ctx(ctx);

  m_model_loaded = true;

  if (g_dev_console && g_dev_console->IsEnabled()) {
    g_dev_console->WriteLine(L"[LLM] 模型加载成功");
  }

  return true;
}

void LlamaCppProvider::Cleanup() {
  if (m_sampler) {
    llama_sampler_free((llama_sampler*)m_sampler);
    m_sampler = nullptr;
  }

  if (m_context) {
    llama_free((llama_context*)m_context);
    m_context = nullptr;
  }

  if (m_model) {
    llama_model_free((llama_model*)m_model);
    m_model = nullptr;
  }

  m_memory = nullptr;
  m_vocab = nullptr;
  m_ctx_size = 0;
  m_system_prompt_utf8.clear();
  m_system_state.clear();
  m_system_state_size = 0;
  m_system_prompt_ready = false;
  m_model_loaded = false;
}

bool LlamaCppProvider::PrepareSystemPrompt(const std::string& system_prompt_utf8) {
  extern DevConsole* g_dev_console;

  if (!m_model_loaded || !m_model || !m_context || !m_sampler || !m_memory || !m_vocab) {
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] System prompt缓存失败: 模型未加载或资源未初始化");
    }
    return false;
  }

  if (m_system_prompt_ready && system_prompt_utf8 == m_system_prompt_utf8) {
    return true;
  }

  m_system_prompt_utf8 = system_prompt_utf8;
  m_system_prompt_ready = false;
  m_system_state.clear();
  m_system_state_size = 0;

  llama_context* ctx = (llama_context*)m_context;
  llama_memory_t mem = (llama_memory_t)m_memory;
  const llama_vocab* vocab = (const llama_vocab*)m_vocab;

  // 清空当前序列，重新预填system prompt
  llama_memory_seq_rm(mem, 0, -1, -1);

  const int n_prompt_tokens = -llama_tokenize(
      vocab,
      system_prompt_utf8.c_str(),
      (int32_t)system_prompt_utf8.size(),
      NULL,
      0,
      true,
      true);
  if (n_prompt_tokens <= 0) {
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] System prompt缓存失败: tokenize失败");
    }
    return false;
  }

  std::vector<llama_token> prompt_tokens(n_prompt_tokens);
  if (llama_tokenize(vocab,
                     system_prompt_utf8.c_str(),
                     (int32_t)system_prompt_utf8.size(),
                     prompt_tokens.data(),
                     (int32_t)prompt_tokens.size(),
                     true,
                     true) < 0) {
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] System prompt缓存失败: tokenize失败");
    }
    return false;
  }

  llama_batch batch = llama_batch_get_one(prompt_tokens.data(), (int32_t)prompt_tokens.size());
  if (llama_decode(ctx, batch) != 0) {
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] System prompt缓存失败: decode失败");
    }
    return false;
  }

  size_t state_size = llama_state_seq_get_size(ctx, 0);
  if (state_size == 0) {
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] System prompt缓存失败: state size 为 0");
    }
    return false;
  }

  m_system_state.resize(state_size);
  size_t written = llama_state_seq_get_data(ctx, m_system_state.data(), state_size, 0);
  if (written == 0) {
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] System prompt缓存失败: state复制失败");
    }
    return false;
  }

  m_system_state_size = written;
  m_system_prompt_ready = true;

  if (g_dev_console && g_dev_console->IsEnabled()) {
    g_dev_console->WriteLine(L"[LLM] System prompt 已缓存，避免重复prefill");
  }

  return true;
}

std::string LlamaCppProvider::GenerateText(const std::string& prompt, size_t max_tokens) {
  extern DevConsole* g_dev_console;

  if (!m_model_loaded || !m_model || !m_context || !m_sampler || !m_memory || !m_vocab) {
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] GenerateText失败: 模型未加载或资源未初始化");
    }
    return "";
  }

  llama_context* ctx = (llama_context*)m_context;
  llama_sampler* smpl = (llama_sampler*)m_sampler;
  llama_memory_t mem = (llama_memory_t)m_memory;
  const llama_vocab* vocab = (const llama_vocab*)m_vocab;

  // 还原 system prompt 的 KV 状态，避免重复 prefill
  bool add_special = true;
  if (m_system_prompt_ready) {
    size_t restored = llama_state_seq_set_data(ctx, m_system_state.data(), m_system_state_size, 0);
    if (restored == 0) {
      if (!PrepareSystemPrompt(m_system_prompt_utf8) ||
          llama_state_seq_set_data(ctx, m_system_state.data(), m_system_state_size, 0) == 0) {
        if (g_dev_console && g_dev_console->IsEnabled()) {
          g_dev_console->WriteLine(L"[LLM] GenerateText失败: system prompt 状态恢复失败");
        }
        return "";
      }
    }
    add_special = false;
  } else if (!m_system_prompt_utf8.empty()) {
    if (!PrepareSystemPrompt(m_system_prompt_utf8) ||
        llama_state_seq_set_data(ctx, m_system_state.data(), m_system_state_size, 0) == 0) {
      if (g_dev_console && g_dev_console->IsEnabled()) {
        g_dev_console->WriteLine(L"[LLM] GenerateText失败: system prompt 缓存失败");
      }
      return "";
    }
    add_special = false;
  } else {
    llama_memory_seq_rm(mem, 0, -1, -1);
  }

  // Tokenize prompt
  const int n_prompt_tokens = -llama_tokenize(
      vocab,
      prompt.c_str(),
      (int32_t)prompt.size(),
      NULL,
      0,
      add_special,
      true);
  if (n_prompt_tokens <= 0) {
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] GenerateText失败: tokenize失败");
    }
    return "";
  }

  std::vector<llama_token> prompt_tokens(n_prompt_tokens);
  if (llama_tokenize(vocab,
                     prompt.c_str(),
                     (int32_t)prompt.size(),
                     prompt_tokens.data(),
                     (int32_t)prompt_tokens.size(),
                     add_special,
                     true) < 0) {
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] GenerateText失败: tokenize失败");
    }
    return "";
  }

  // 准备批次并生成
  llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
  std::string response;

  // 耗时统计
  ULONGLONG total_start = GetTickCount64();
  ULONGLONG t_ctx_check_total = 0;
  ULONGLONG t_decode_total = 0;
  ULONGLONG t_sample_total = 0;
  ULONGLONG t_convert_total = 0;
  ULONGLONG t_batch_prep_total = 0;
  size_t tokens_generated = 0;

  for (size_t i = 0; i < max_tokens; ++i) {
    // 检查上下文大小
    ULONGLONG t0 = GetTickCount64();
    int n_ctx_used = llama_memory_seq_pos_max(mem, 0) + 1;
    ULONGLONG t_ctx_check = GetTickCount64() - t0;
    t_ctx_check_total += t_ctx_check;
    
    if (n_ctx_used + batch.n_tokens > m_ctx_size) {
      if (g_dev_console && g_dev_console->IsEnabled()) {
        g_dev_console->WriteLine(L"[LLM] Token " + std::to_wstring(i) + L": 上下文已满，停止生成");
      }
      break;
    }

    // 解码
    ULONGLONG t1 = GetTickCount64();
    if (llama_decode(ctx, batch) != 0) {
      if (g_dev_console && g_dev_console->IsEnabled()) {
        g_dev_console->WriteLine(L"[LLM] Token " + std::to_wstring(i) + L": 解码失败");
      }
      break;
    }
    ULONGLONG t_decode = GetTickCount64() - t1;
    t_decode_total += t_decode;
    
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] Token " + std::to_wstring(i) + L": decode耗时 " + std::to_wstring(t_decode) + L" ms");
    }

    // 采样下一个token
    ULONGLONG t2 = GetTickCount64();
    llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);
    ULONGLONG t_sample = GetTickCount64() - t2;
    t_sample_total += t_sample;
    
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] Token " + std::to_wstring(i) + L": sample耗时 " + std::to_wstring(t_sample) + L" ms (token_id=" + std::to_wstring(new_token_id) + L")");
    }
    
    if (llama_vocab_is_eog(vocab, new_token_id)) {
      if (g_dev_console && g_dev_console->IsEnabled()) {
        g_dev_console->WriteLine(L"[LLM] Token " + std::to_wstring(i) + L": 遇到结束token");
      }
      break;
    }

    // 转换为文本
    ULONGLONG t3 = GetTickCount64();
    char buf[256];
    int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
    ULONGLONG t_convert = GetTickCount64() - t3;
    t_convert_total += t_convert;
    
    if (n < 0) {
      if (g_dev_console && g_dev_console->IsEnabled()) {
        g_dev_console->WriteLine(L"[LLM] Token " + std::to_wstring(i) + L": token_to_piece失败");
      }
      break;
    }
    response.append(buf, n);
    
    if (g_dev_console && g_dev_console->IsEnabled()) {
      std::string piece(buf, n);
      g_dev_console->WriteLine(L"[LLM] Token " + std::to_wstring(i) + L": convert耗时 " + std::to_wstring(t_convert) + L" ms (text=\"" + u8tow(piece) + L"\")");
    }

    // 准备下一个批次
    ULONGLONG t4 = GetTickCount64();
    batch = llama_batch_get_one(&new_token_id, 1);
    ULONGLONG t_batch_prep = GetTickCount64() - t4;
    t_batch_prep_total += t_batch_prep;
    
    tokens_generated++;
    
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] Token " + std::to_wstring(i) + L": batch_prep耗时 " + std::to_wstring(t_batch_prep) + L" ms");
      ULONGLONG token_total = t_ctx_check + t_decode + t_sample + t_convert + t_batch_prep;
      g_dev_console->WriteLine(L"[LLM] Token " + std::to_wstring(i) + L": 总耗时 " + std::to_wstring(token_total) + L" ms");
    }
  }

  ULONGLONG total_time = GetTickCount64() - total_start;

  // 输出总体统计
  if (g_dev_console && g_dev_console->IsEnabled()) {
    g_dev_console->WriteLine(L"[LLM] GenerateText 总体统计:");
    g_dev_console->WriteLine(L"  生成token数: " + std::to_wstring(tokens_generated));
    g_dev_console->WriteLine(L"  上下文检查总耗时: " + std::to_wstring(t_ctx_check_total) + L" ms" + 
                             (tokens_generated > 0 ? L" (平均: " + std::to_wstring(t_ctx_check_total / tokens_generated) + L" ms/token)" : L""));
    g_dev_console->WriteLine(L"  解码总耗时: " + std::to_wstring(t_decode_total) + L" ms" + 
                             (tokens_generated > 0 ? L" (平均: " + std::to_wstring(t_decode_total / tokens_generated) + L" ms/token)" : L""));
    g_dev_console->WriteLine(L"  采样总耗时: " + std::to_wstring(t_sample_total) + L" ms" + 
                             (tokens_generated > 0 ? L" (平均: " + std::to_wstring(t_sample_total / tokens_generated) + L" ms/token)" : L""));
    g_dev_console->WriteLine(L"  转换总耗时: " + std::to_wstring(t_convert_total) + L" ms" + 
                             (tokens_generated > 0 ? L" (平均: " + std::to_wstring(t_convert_total / tokens_generated) + L" ms/token)" : L""));
    g_dev_console->WriteLine(L"  批次准备总耗时: " + std::to_wstring(t_batch_prep_total) + L" ms" + 
                             (tokens_generated > 0 ? L" (平均: " + std::to_wstring(t_batch_prep_total / tokens_generated) + L" ms/token)" : L""));
    g_dev_console->WriteLine(L"  总耗时: " + std::to_wstring(total_time) + L" ms");
  }

  return response;
}

std::vector<std::string> LlamaCppProvider::GenerateCandidatesBatch(
    const std::string& system_prompt_utf8, const std::string& user_prompt_utf8, size_t n_parallel, int max_new_tokens) {
  std::vector<std::string> candidates;
  extern DevConsole* g_dev_console;

  if (!m_model_loaded || !m_model || !m_context || !m_sampler || !m_memory || !m_vocab) {
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] GenerateCandidatesBatch失败: 模型未加载或资源未初始化");
    }
    return candidates;
  }

  if (n_parallel == 0 || max_new_tokens <= 0) {
    return candidates;
  }

  llama_context* ctx = (llama_context*)m_context;
  llama_sampler* smpl = (llama_sampler*)m_sampler;
  llama_memory_t mem = (llama_memory_t)m_memory;
  const llama_vocab* vocab = (const llama_vocab*)m_vocab;

  int n_system_tokens = 0;
  int n_user_tokens = 0;
  std::vector<llama_token> prompt_tokens;
  bool add_special = false;

  if (!system_prompt_utf8.empty()) {
    // 与单次生成一致：复用 system 的 KV cache，只 prefill user 部分
    if (m_system_prompt_ready) {
      size_t restored = llama_state_seq_set_data(ctx, m_system_state.data(), m_system_state_size, 0);
      if (restored == 0) {
        if (!PrepareSystemPrompt(system_prompt_utf8) ||
            llama_state_seq_set_data(ctx, m_system_state.data(), m_system_state_size, 0) == 0) {
          if (g_dev_console && g_dev_console->IsEnabled()) {
            g_dev_console->WriteLine(L"[LLM] GenerateCandidatesBatch: system 状态恢复失败");
          }
          return candidates;
        }
      }
    } else {
      if (!PrepareSystemPrompt(system_prompt_utf8) ||
          llama_state_seq_set_data(ctx, m_system_state.data(), m_system_state_size, 0) == 0) {
        if (g_dev_console && g_dev_console->IsEnabled()) {
          g_dev_console->WriteLine(L"[LLM] GenerateCandidatesBatch: system prompt 缓存失败");
        }
        return candidates;
      }
    }
    n_system_tokens = (int)(llama_memory_seq_pos_max(mem, 0) + 1);
    add_special = false;
  } else {
    llama_memory_seq_rm(mem, -1, -1, -1);
    add_special = false;
  }

  // Tokenize 本次要 prefill 的部分（有 system 时仅 user，无 system 时为完整 user）
  const char* to_tokenize = user_prompt_utf8.c_str();
  int to_tokenize_len = (int)user_prompt_utf8.size();
  const int n_prefill_tokens = -llama_tokenize(vocab, to_tokenize, to_tokenize_len, NULL, 0, add_special, true);
  if (n_prefill_tokens <= 0) {
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] GenerateCandidatesBatch: tokenize 失败");
    }
    return candidates;
  }
  prompt_tokens.resize((size_t)n_prefill_tokens);
  if (llama_tokenize(vocab, to_tokenize, to_tokenize_len, prompt_tokens.data(), (int32_t)prompt_tokens.size(), add_special, true) < 0) {
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] GenerateCandidatesBatch: tokenize 失败");
    }
    return candidates;
  }
  n_user_tokens = n_prefill_tokens;
  const int n_prompt_tokens = n_system_tokens + n_user_tokens;

  const int n_parallel_i = (int)(n_parallel <= (size_t)INT32_MAX ? n_parallel : INT32_MAX);
  const int n_len = n_prompt_tokens + max_new_tokens;
  const int64_t n_kv_req = (int64_t)n_prompt_tokens + (int64_t)max_new_tokens * (int64_t)n_parallel_i;
  if (n_kv_req > m_ctx_size) {
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] GenerateCandidatesBatch: KV 需求 " + std::to_wstring(n_kv_req) +
                               L" 超过 n_ctx " + std::to_wstring(m_ctx_size) + L"，请减少 n_parallel 或增大 n_ctx");
    }
    return candidates;
  }

  const int32_t batch_cap = (int32_t)((size_t)n_prefill_tokens >= (size_t)n_parallel_i ? (size_t)n_prefill_tokens : (size_t)n_parallel_i);
  llama_batch batch = llama_batch_init(batch_cap, 0, n_parallel_i);
  if (batch.token == nullptr) {
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] GenerateCandidatesBatch: batch 分配失败");
    }
    return candidates;
  }

  // Prefill：仅对序列 0 喂入 user 部分；有 system 时 pos 从 n_system_tokens 起
  batch.n_tokens = n_prefill_tokens;
  for (int32_t i = 0; i < n_prefill_tokens; i++) {
    batch.token[i] = prompt_tokens[i];
    if (batch.pos) batch.pos[i] = (llama_pos)(n_system_tokens + i);
    if (batch.seq_id && batch.seq_id[i]) *batch.seq_id[i] = 0;
    if (batch.n_seq_id) batch.n_seq_id[i] = 1;
    if (batch.logits) batch.logits[i] = (i == n_prefill_tokens - 1) ? 1 : 0;
  }

  ULONGLONG total_start = GetTickCount64();
  ULONGLONG t_prefill = 0;
  ULONGLONG t_sample_convert_total = 0;
  ULONGLONG t_decode_total = 0;
  size_t tokens_generated = 0;
  int decode_step = 0;

  ULONGLONG t_prefill_start = GetTickCount64();
  if (llama_decode(ctx, batch) != 0) {
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] GenerateCandidatesBatch: 初始 decode 失败");
    }
    llama_batch_free(batch);
    return candidates;
  }
  t_prefill = GetTickCount64() - t_prefill_start;

  for (int32_t i = 1; i < n_parallel_i; ++i) {
    llama_memory_seq_cp(mem, 0, (llama_seq_id)i, 0, -1);
  }

  candidates.resize((size_t)n_parallel_i);
  const int32_t first_logits_batch_idx = n_prefill_tokens - 1;
  std::vector<int32_t> i_batch((size_t)n_parallel_i, first_logits_batch_idx);

  int n_cur = n_prompt_tokens;
  const int n_vocab = llama_vocab_n_tokens(vocab);
  std::vector<llama_token_data> candidates_vec;
  candidates_vec.reserve((size_t)n_vocab);

  while (n_cur < n_len) {
    batch.n_tokens = 0;

    ULONGLONG t_sample_convert_start = GetTickCount64();
    for (int32_t i = 0; i < n_parallel_i; ++i) {
      if (i_batch[i] < 0) continue;

      float* logits = llama_get_logits_ith(ctx, i_batch[i]);
      if (!logits) continue;

      candidates_vec.clear();
      for (llama_token id = 0; id < (llama_token)n_vocab; id++) {
        candidates_vec.push_back(llama_token_data{ id, logits[id], 0.0f });
      }
      llama_token_data_array cur_p = { candidates_vec.data(), candidates_vec.size(), -1, false };
      llama_sampler_apply(smpl, &cur_p);
      llama_token new_token_id = llama_sampler_sample(smpl, ctx, i_batch[i]);

      if (llama_vocab_is_eog(vocab, new_token_id) || n_cur >= n_len) {
        i_batch[i] = -1;
        continue;
      }

      char buf[256];
      int n = llama_token_to_piece(vocab, new_token_id, buf, (int32_t)sizeof(buf), 0, true);
      if (n > 0) {
        candidates[i].append(buf, (size_t)n);
      }

      const int32_t batch_idx = batch.n_tokens;
      batch.token[batch_idx] = new_token_id;
      if (batch.pos) batch.pos[batch_idx] = (llama_pos)n_cur;
      if (batch.seq_id && batch.seq_id[batch_idx]) *batch.seq_id[batch_idx] = (llama_seq_id)i;
      if (batch.n_seq_id) batch.n_seq_id[batch_idx] = 1;
      if (batch.logits) batch.logits[batch_idx] = 1;
      batch.n_tokens += 1;

      i_batch[i] = batch_idx;
    }
    t_sample_convert_total += GetTickCount64() - t_sample_convert_start;

    if (batch.n_tokens == 0) break;

    tokens_generated += (size_t)batch.n_tokens;
    n_cur += 1;

    ULONGLONG t_decode_start = GetTickCount64();
    if (llama_decode(ctx, batch) != 0) {
      if (g_dev_console && g_dev_console->IsEnabled()) {
        g_dev_console->WriteLine(L"[LLM] GenerateCandidatesBatch: 循环 decode 失败");
      }
      break;
    }
    ULONGLONG t_step_decode = GetTickCount64() - t_decode_start;
    t_decode_total += t_step_decode;
    decode_step++;

    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] GenerateCandidatesBatch 步骤 " + std::to_wstring(decode_step) +
          L": decode " + std::to_wstring(batch.n_tokens) + L" tokens, 本步 decode 耗时 " +
          std::to_wstring(t_step_decode) + L" ms");
    }
  }

  ULONGLONG total_time = GetTickCount64() - total_start;

  if (g_dev_console && g_dev_console->IsEnabled()) {
    g_dev_console->WriteLine(L"[LLM] GenerateCandidatesBatch 总体统计:");
    g_dev_console->WriteLine(L"  生成 token 数: " + std::to_wstring(tokens_generated));
    g_dev_console->WriteLine(L"  prefill decode 耗时: " + std::to_wstring(t_prefill) + L" ms");
    g_dev_console->WriteLine(L"  采样+转换总耗时: " + std::to_wstring(t_sample_convert_total) + L" ms" +
        (tokens_generated > 0 ? L" (平均: " + std::to_wstring(t_sample_convert_total / tokens_generated) + L" ms/token)" : L""));
    g_dev_console->WriteLine(L"  解码总耗时: " + std::to_wstring(t_decode_total) + L" ms" +
        (tokens_generated > 0 ? L" (平均: " + std::to_wstring(t_decode_total / tokens_generated) + L" ms/token)" : L""));
    g_dev_console->WriteLine(L"  总耗时: " + std::to_wstring(total_time) + L" ms" +
        (tokens_generated > 0 ? L", 速度: " + std::to_wstring((total_time > 0) ? (tokens_generated * 1000 / total_time) : 0) + L" token/s" : L""));
  }

  llama_batch_free(batch);
  llama_memory_seq_rm(mem, -1, -1, -1);

  return candidates;
}

std::vector<std::wstring> LlamaCppProvider::ExecuteRequest(
    const LLMRequest& request) {
  std::vector<std::wstring> candidates;

  if (!IsAvailable() || !IsExecutableRequest(request)) {
    return candidates;
  }

  // Base 模型补全可能产生末尾 U+FFFD（不完整 UTF-8 等），需去除
  auto trim_trailing_fffd = [](std::wstring& s) {
    while (!s.empty() && s.back() == L'\uFFFD') s.pop_back();
  };
  // 移除候选词内部所有空格
  auto remove_all_spaces = [](std::wstring& s) {
    s.erase(std::remove(s.begin(), s.end(), L' '), s.end());
  };

  std::string system_prompt_utf8;
  std::string prompt_utf8;

  if (m_instruct_model) {
    std::wstring system_prompt;
    std::wstring user_prompt;
    BuildInstructPrompts(request, system_prompt, user_prompt);
    if (system_prompt.empty() || user_prompt.empty()) {
      return candidates;
    }
    system_prompt += L"\n";
    system_prompt_utf8 = wtou8(system_prompt);
    prompt_utf8 = wtou8(user_prompt);
  } else {
    std::wstring context_prefix = BuildBasePrompt(request);
    if (context_prefix.empty()) {
      return candidates;
    }
    system_prompt_utf8.clear();
    prompt_utf8 = wtou8(context_prefix);
  }

  extern DevConsole* g_dev_console;
  if (g_dev_console && g_dev_console->IsEnabled()) {
    g_dev_console->WriteLine(L"[LLM] 发送请求 (llama.cpp)");
    g_dev_console->WriteLine(L"  请求类型: " + GetRequestTypeName(request.type));
    g_dev_console->WriteLine(L"  上下文: " + request.context);
    if (!request.preference_hint.empty()) {
      g_dev_console->WriteLine(L"  用户偏好: " + request.preference_hint);
    }
    if (!request.current_input.empty()) {
      g_dev_console->WriteLine(L"  当前输入: " + request.current_input);
    }
    if (!request.rime_candidates.empty()) {
      g_dev_console->WriteLine(
          L"  Rime候选数: " + std::to_wstring(request.rime_candidates.size()));
    }
  }

  // 当需要多个候选时，使用批量采样（与单次生成一样复用 system KV cache）
  if (request.max_candidates > 1) {
    ULONGLONG start_time = GetTickCount64();
    std::vector<std::string> raw = GenerateCandidatesBatch(
        system_prompt_utf8, prompt_utf8, request.max_candidates, 4);
    ULONGLONG elapsed_ms = GetTickCount64() - start_time;
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] 批量采样完成，耗时: " + std::to_wstring(elapsed_ms) + L" ms");
    }
    for (const auto& s : raw) {
      std::wstring w = u8tow(s);
      if (!w.empty()) candidates.push_back(w);
    }
    if (!m_instruct_model) {
      for (auto& c : candidates) trim_trailing_fffd(c);
    }
    for (auto& c : candidates) remove_all_spaces(c);
    return candidates;
  }

  // 单候选：Instruct 需先缓存 system prompt，Base 无需
  if (!system_prompt_utf8.empty()) {
    if (!PrepareSystemPrompt(system_prompt_utf8)) {
      if (g_dev_console && g_dev_console->IsEnabled()) {
        g_dev_console->WriteLine(L"[LLM] System prompt 缓存失败，放弃本次预测");
      }
      return candidates;
    }
  }

  // 生成文本（单候选或回退）
  ULONGLONG start_time = GetTickCount64();
  std::string response = GenerateText(prompt_utf8, m_max_tokens);
  ULONGLONG end_time = GetTickCount64();
  ULONGLONG elapsed_ms = end_time - start_time;

  if (g_dev_console && g_dev_console->IsEnabled()) {
    g_dev_console->WriteLine(L"[LLM] 收到响应 (llama.cpp)");
    g_dev_console->WriteLine(L"  响应内容: " + u8tow(response));
    g_dev_console->WriteLine(L"[LLM] @WeaselServer/LlamaCppProvider.cpp:434 耗时: " + std::to_wstring(elapsed_ms) + L" ms");
  }

  // 解析响应（按空格分割）
  std::wstring response_w = u8tow(response);
  std::wstringstream ss(response_w);
  std::wstring word;
  while (ss >> word && candidates.size() < request.max_candidates) {
    if (!word.empty()) {
      candidates.push_back(word);
    }
  }

  if (!m_instruct_model) {
    for (auto& c : candidates) trim_trailing_fffd(c);
  }
  for (auto& c : candidates) remove_all_spaces(c);
  return candidates;
}

bool LlamaCppProvider::IsAvailable() const {
  return m_enabled && m_model_loaded && !m_model_path.empty();
}

