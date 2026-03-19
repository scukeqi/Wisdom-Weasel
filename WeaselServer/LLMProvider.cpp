#include "stdafx.h"
#include "LLMProvider.h"
#include "ConfigJsonUtils.h"
#include "DevConsole.h"
#include <WeaselUtility.h>
#include <rime_api.h>
#include <winhttp.h>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cstring>

#pragma comment(lib, "winhttp.lib")

namespace {

bool HeaderNameEquals(const std::string& lhs, const char* rhs) {
  if (!rhs || lhs.size() != strlen(rhs)) {
    return false;
  }

  for (size_t i = 0; i < lhs.size(); ++i) {
    if (std::tolower(static_cast<unsigned char>(lhs[i])) !=
        std::tolower(static_cast<unsigned char>(rhs[i]))) {
      return false;
    }
  }
  return true;
}

bool HasHeaderNamed(
    const std::vector<std::pair<std::string, std::string>>& headers,
    const char* header_name) {
  for (const auto& header : headers) {
    if (HeaderNameEquals(header.first, header_name)) {
      return true;
    }
  }
  return false;
}

std::string StripJsonObjectBraces(const std::string& json_object) {
  if (json_object.size() < 2 || json_object.front() != '{' ||
      json_object.back() != '}') {
    return std::string();
  }
  return json_object.substr(1, json_object.size() - 2);
}

bool FindFirstStringFieldByName(const boost::property_tree::ptree& node,
                                const char* field_name,
                                std::string& value) {
  for (const auto& child : node) {
    if (child.first == field_name) {
      const std::string field_value = child.second.get_value<std::string>();
      if (!field_value.empty()) {
        value = field_value;
        return true;
      }
    }

    if (FindFirstStringFieldByName(child.second, field_name, value)) {
      return true;
    }
  }

  return false;
}

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

std::wstring BuildOpenAIPrompt(const LLMRequest& request) {
  const size_t output_limit = (std::min)(
      request.max_candidates,
      request.type == LLMRequestType::RimeReorder ? request.rime_candidates.size()
                                                  : request.max_candidates);

  switch (request.type) {
    case LLMRequestType::NoInputPrediction: {
      std::wstring prompt =
          L"你是一个智能中文输入法，请根据以下上下文预测接下来最可能出现的"
          + std::to_wstring(request.max_candidates) + L"个候选词。\n\n"
          L"要求：\n"
          L"1. 只返回候选词，不要任何解释或标点\n"
          L"2. 候选词之间用单个空格分隔\n"
          L"3. 按可能性从高到低排列\n"
          L"4. 确保候选词都是有效的中文词汇或常用短语\n"
          L"5. 返回词数严格不超过"
          + std::to_wstring(request.max_candidates) + L"个\n";
      if (!request.preference_hint.empty()) {
        prompt +=
            L"6. 用户偏好只作弱参考，必须优先保证与最近上下文的连贯性\n"
            L"7. 若用户偏好与当前上下文冲突，忽略用户偏好，以当前上下文为准\n";
      }
      prompt += L"\n上下文：\"" + request.context + L"\"\n";
      if (!request.preference_hint.empty()) {
        prompt +=
            L"用户偏好（弱参考，可忽略）：\"" + request.preference_hint + L"\"\n";
      }
      prompt += L"候选词：";
      return prompt;
    }
    case LLMRequestType::PinyinConstrainedPrediction: {
      std::wstring prompt =
          L"你是一个智能中文输入法，请根据以下上下文和当前拼音，预测接下来最可能出现的"
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
        prompt +=
            L"8. 用户偏好只作弱参考，必须优先保证与最近上下文和当前拼音的连贯性\n"
            L"9. 若用户偏好与当前上下文冲突，忽略用户偏好，以当前上下文为准\n";
      }
      prompt += L"\n上下文：\"" + request.context + L"\"\n";
      if (!request.preference_hint.empty()) {
        prompt +=
            L"用户偏好（弱参考，可忽略）：\"" + request.preference_hint + L"\"\n";
      }
      prompt += L"当前拼音：\"" + request.current_input + L"\"\n"
                L"候选词：";
      return prompt;
    }
    case LLMRequestType::RimeReorder: {
      std::wstring prompt =
          L"你是一个智能中文输入法，请根据以下上下文、当前拼音和用户偏好，对给定的 Rime 候选词重新排序。\n\n"
          L"要求：\n"
          L"1. 只能从给定候选列表中选择，不得新增、改写或拆分候选词\n"
          L"2. 只返回重排后的候选词，不要解释、编号或标点\n"
          L"3. 候选词之间用单个空格分隔\n"
          L"4. 按更符合上下文的顺序输出\n"
          L"5. 若无法判断，尽量保持原顺序\n"
          L"6. 返回词数严格不超过"
          + std::to_wstring(output_limit) + L"个\n";
      prompt += L"\n上下文：\"" + request.context + L"\"\n";
      if (!request.preference_hint.empty()) {
        prompt +=
            L"用户偏好（弱参考，可忽略）：\"" + request.preference_hint + L"\"\n";
      }
      if (!request.current_input.empty()) {
        prompt += L"当前拼音：\"" + request.current_input + L"\"\n";
      }
      prompt += L"Rime 候选词：\n" + JoinCandidatesForPrompt(request.rime_candidates)
                + L"\n重排结果：";
      return prompt;
    }
  }
  return std::wstring();
}

}  // namespace

OpenAICompatibleProvider::OpenAICompatibleProvider()
    : m_enabled(false),
      m_max_tokens(10),
      m_temperature(0.7),
      m_hSession(nullptr),
      m_hConnect(nullptr) {}

OpenAICompatibleProvider::~OpenAICompatibleProvider() {
  CloseConnection();
}

void OpenAICompatibleProvider::CloseConnection() {
  if (m_hConnect) {
    WinHttpCloseHandle((HINTERNET)m_hConnect);
    m_hConnect = nullptr;
  }
  if (m_hSession) {
    WinHttpCloseHandle((HINTERNET)m_hSession);
    m_hSession = nullptr;
  }
  m_cached_url.clear();
}

bool OpenAICompatibleProvider::LoadConfig(const std::string& config_name) {
  extern DevConsole* g_dev_console;

  // 硬编码测试配置（用于测试）
  // 注意：如果设置为 true，将不会读取yaml配置文件
  bool use_hardcoded_config =
      false;  // 设置为 true 使用硬编码配置，false 使用配置文件

  if (use_hardcoded_config) {
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] 使用硬编码测试配置");
    }
    m_enabled = true;
    m_api_url = "http://localhost:11434/v1/chat/completions";
    m_api_key = "";
    m_model = "qwen3:8b";
    m_max_tokens = 10;
    m_temperature = 0.7;
    m_extra_body_json.clear();
    m_extra_headers.clear();

    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] LoadConfig: llm/enabled = true");
      g_dev_console->WriteLine(L"[LLM] LoadConfig: api_url = " +
                               u8tow(m_api_url));
      g_dev_console->WriteLine(L"[LLM] LoadConfig: model = " + u8tow(m_model));
      g_dev_console->WriteLine(L"[LLM] LoadConfig: max_tokens = " +
                               std::to_wstring(m_max_tokens));
      g_dev_console->WriteLine(L"[LLM] LoadConfig: temperature = " +
                               std::to_wstring(m_temperature));
      g_dev_console->WriteLine(L"[LLM] LoadConfig: 配置加载成功（硬编码）");
    }
    CloseConnection();  // URL 可能变化，下次请求时重建连接
    return true;
  }

  RimeApi* rime_api = rime_get_api();
  if (!rime_api) {
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] LoadConfig失败: rime_api未初始化");
      g_dev_console->WriteLine(
          L"[LLM] 可能原因: Rime API在LoadConfig之前未正确初始化");
    }
    return false;
  }

  if (g_dev_console && g_dev_console->IsEnabled()) {
    g_dev_console->WriteLine(L"[LLM] 开始从配置文件加载: " +
                             u8tow(config_name));
  }

  RimeConfig config = {NULL};
  if (!rime_api->config_open(config_name.c_str(), &config)) {
    if (g_dev_console && g_dev_console->IsEnabled()) {
      std::wstring config_name_w = u8tow(config_name);
      g_dev_console->WriteLine(L"[LLM] LoadConfig失败: 无法打开配置文件 " +
                               config_name_w);
      g_dev_console->WriteLine(L"[LLM] 可能原因:");
      g_dev_console->WriteLine(
          L"[LLM]   1. 配置文件不存在: weasel.yaml 或 weasel.custom.yaml");
      g_dev_console->WriteLine(L"[LLM]   2. 配置文件路径错误");
      g_dev_console->WriteLine(L"[LLM]   3. 配置文件格式错误（YAML语法错误）");
      g_dev_console->WriteLine(L"[LLM]   4. Rime未正确初始化");
    }
    return false;
  }

  if (g_dev_console && g_dev_console->IsEnabled()) {
    g_dev_console->WriteLine(L"[LLM] 配置文件打开成功，开始读取配置项");
  }

  // 读取LLM配置
  Bool enabled = false;
  bool found_enabled =
      rime_api->config_get_bool(&config, "llm/enabled", &enabled);

  if (g_dev_console && g_dev_console->IsEnabled()) {
    if (found_enabled) {
      g_dev_console->WriteLine(L"[LLM] 找到配置项 llm/enabled = " +
                               std::wstring(enabled ? L"true" : L"false"));
    } else {
      g_dev_console->WriteLine(L"[LLM] 未找到配置项 llm/enabled");
      g_dev_console->WriteLine(L"[LLM] 尝试读取的路径: llm/enabled");
    }
  }

  if (found_enabled) {
    m_enabled = !!enabled;
  } else {
    m_enabled = false;
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(
          L"[LLM] LoadConfig失败: 未找到配置项 llm/enabled");
      g_dev_console->WriteLine(L"[LLM] 请在配置文件中添加：");
      g_dev_console->WriteLine(L"[LLM]   llm:");
      g_dev_console->WriteLine(L"[LLM]     enabled: true");
      g_dev_console->WriteLine(
          L"[LLM] 配置文件位置通常在: %APPDATA%\\Rime\\weasel.yaml");
    }
    rime_api->config_close(&config);
    return false;
  }

  if (!m_enabled) {
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] LoadConfig失败: llm/enabled 为 false");
      g_dev_console->WriteLine(
          L"[LLM] 请将配置文件中的 llm/enabled 设置为 true");
    }
    rime_api->config_close(&config);
    return false;
  }

  // 读取OpenAI配置
  const int BUF_SIZE = 512;
  char buffer[BUF_SIZE + 1] = {0};

  bool found_api_url = rime_api->config_get_string(
      &config, "llm/openai/api_url", buffer, BUF_SIZE);
  if (found_api_url) {
    m_api_url = buffer;
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] 找到配置项 llm/openai/api_url = " +
                               u8tow(m_api_url));
    }
  } else {
    m_api_url = "https://api.openai.com/v1/chat/completions";
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(
          L"[LLM] 未找到配置项 llm/openai/api_url，使用默认值 = " +
          u8tow(m_api_url));
      g_dev_console->WriteLine(
          L"[LLM] 建议在配置文件中添加: llm/openai/api_url");
    }
  }

  // api_key是可选的，对于本地服务（如Ollama）可以为空
  bool found_api_key = rime_api->config_get_string(
      &config, "llm/openai/api_key", buffer, BUF_SIZE);
  if (found_api_key) {
    m_api_key = buffer;
    if (g_dev_console && g_dev_console->IsEnabled()) {
      if (m_api_key.empty()) {
        g_dev_console->WriteLine(
            L"[LLM] 找到配置项 llm/openai/api_key = "
            L"(空，适用于本地服务如Ollama)");
      } else {
        // 只显示前8个字符，保护隐私
        std::wstring key_preview =
            m_api_key.length() > 8 ? (u8tow(m_api_key.substr(0, 8)) + L"...")
                                   : u8tow(m_api_key);
        g_dev_console->WriteLine(L"[LLM] 找到配置项 llm/openai/api_key = " +
                                 key_preview);
      }
    }
  } else {
    // api_key未配置，使用空字符串（适用于本地服务）
    m_api_key = "";
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(
          L"[LLM] 未找到配置项 "
          L"llm/openai/api_key，使用空字符串（适用于本地服务如Ollama）");
    }
  }

  bool found_model = rime_api->config_get_string(&config, "llm/openai/model",
                                                 buffer, BUF_SIZE);
  if (found_model) {
    m_model = buffer;
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] 找到配置项 llm/openai/model = " +
                               u8tow(m_model));
    }
  } else {
    m_model = "gpt-3.5-turbo";
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(
          L"[LLM] 未找到配置项 llm/openai/model，使用默认值 = " +
          u8tow(m_model));
    }
  }

  int max_tokens = 10;
  if (rime_api->config_get_int(&config, "llm/openai/max_tokens", &max_tokens)) {
    m_max_tokens = max_tokens;
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] LoadConfig: max_tokens = " +
                               std::to_wstring(m_max_tokens));
    }
  } else {
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] LoadConfig: 使用默认 max_tokens = " +
                               std::to_wstring(m_max_tokens));
    }
  }

  // Rime API可能不支持config_get_double，使用字符串读取然后转换
  char temp_str[64] = {0};
  if (rime_api->config_get_string(&config, "llm/openai/temperature", temp_str,
                                  sizeof(temp_str) - 1)) {
    m_temperature = atof(temp_str);
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] LoadConfig: temperature = " +
                               std::to_wstring(m_temperature));
    }
  } else {
    m_temperature = 0.7;
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] LoadConfig: 使用默认 temperature = " +
                               std::to_wstring(m_temperature));
    }
  }

  m_extra_body_json.clear();
  if (weasel::config_json::SerializeConfigMapToJsonObject(
          rime_api, &config, "llm/openai/extra_body", m_extra_body_json)) {
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] LoadConfig: extra_body = " +
                               u8tow(m_extra_body_json));
    }
  }

  m_extra_headers.clear();
  if (weasel::config_json::LoadConfigStringMap(
          rime_api, &config, "llm/openai/extra_headers", m_extra_headers)) {
    if (g_dev_console && g_dev_console->IsEnabled() &&
        !m_extra_headers.empty()) {
      std::wstring headers_summary;
      for (size_t i = 0; i < m_extra_headers.size(); ++i) {
        if (i > 0) {
          headers_summary += L", ";
        }
        headers_summary += u8tow(m_extra_headers[i].first) + L"=" +
                           u8tow(m_extra_headers[i].second);
      }
      g_dev_console->WriteLine(L"[LLM] LoadConfig: extra_headers = " +
                               headers_summary);
    }
  }

  CloseConnection();  // URL 可能变化，下次请求时重建连接
  rime_api->config_close(&config);

  if (g_dev_console && g_dev_console->IsEnabled()) {
    g_dev_console->WriteLine(L"[LLM] LoadConfig: 配置加载成功");
  }

  return true;
}

std::vector<std::wstring> OpenAICompatibleProvider::ExecuteRequest(
    const LLMRequest& request) {
  std::vector<std::wstring> candidates;

  if (!IsAvailable() || !IsExecutableRequest(request)) {
    return candidates;
  }

  const std::wstring prompt = BuildOpenAIPrompt(request);
  if (prompt.empty()) {
    return candidates;
  }

  // 构建JSON请求体
  std::string prompt_utf8 = wtou8(prompt);
  std::string escaped_prompt =
      weasel::config_json::EscapeJsonString(prompt_utf8);
  std::string extra_body_members = StripJsonObjectBraces(m_extra_body_json);

  std::ostringstream json;
  json << "{"
       << "\"model\":\"" << weasel::config_json::EscapeJsonString(m_model)
       << "\","
       << "\"messages\":["
       << "{\"role\":\"user\",\"content\":\"" << escaped_prompt << "\"}"
       << "],"
       << "\"max_tokens\":" << m_max_tokens << ","
       << "\"temperature\":" << m_temperature;
  if (!extra_body_members.empty()) {
    json << "," << extra_body_members;
  }
  json << "}";

  std::string request_body = json.str();

  // 输出请求内容到开发终端
  extern DevConsole* g_dev_console;
  if (g_dev_console && g_dev_console->IsEnabled()) {
    g_dev_console->WriteLine(L"[LLM] 发送请求（OpenAI Compatible）");
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
    g_dev_console->WriteLine(L"  请求URL: " + u8tow(m_api_url));
    g_dev_console->WriteLine(L"  请求体: " + u8tow(request_body));
  }

  // 执行HTTP请求
  std::string response_body;
  if (!ExecuteHttpRequest(m_api_url, request_body, response_body)) {
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] 请求失败");
    }
    return candidates;
  }

  // 输出响应内容到开发终端
  if (g_dev_console && g_dev_console->IsEnabled()) {
    g_dev_console->WriteLine(L"[LLM] 收到响应");
    g_dev_console->WriteLine(L"  响应内容: " + u8tow(response_body));
  }

  // 解析响应
  candidates = ParseResponse(response_body);

  // if (g_dev_console && g_dev_console->IsEnabled()) {
  //   std::wstringstream ss;
  //   ss << L"[LLM] 解析得到 " << candidates.size() << L" 个候选词";
  //   g_dev_console->WriteLine(ss.str());
  //   for (size_t i = 0; i < candidates.size(); ++i) {
  //     std::wstringstream ss2;
  //     ss2 << L"  " << (i + 1) << L". " << candidates[i];
  //     g_dev_console->WriteLine(ss2.str());
  //   }
  // }

  return candidates;
}

bool OpenAICompatibleProvider::IsAvailable() const {
  // api_key可以为空（适用于本地服务如Ollama），只需要enabled和api_url不为空
  return m_enabled && !m_api_url.empty();
}

bool OpenAICompatibleProvider::ExecuteHttpRequest(
    const std::string& url,
    const std::string& request_body,
    std::string& response_body) {
  URL_COMPONENTS url_comp = {0};
  url_comp.dwStructSize = sizeof(URL_COMPONENTS);
  url_comp.dwSchemeLength = (DWORD)-1;
  url_comp.dwHostNameLength = (DWORD)-1;
  url_comp.dwUrlPathLength = (DWORD)-1;
  url_comp.dwExtraInfoLength = (DWORD)-1;

  std::wstring url_w = u8tow(url);
  wchar_t hostname[256] = {0};
  wchar_t path[1024] = {0};
  url_comp.lpszHostName = hostname;
  url_comp.lpszUrlPath = path;

  if (!WinHttpCrackUrl(url_w.c_str(), (DWORD)url_w.length(), 0, &url_comp)) {
    return false;
  }

  INTERNET_PORT port = url_comp.nPort;
  bool use_https = (url_comp.nScheme == INTERNET_SCHEME_HTTPS);
  if (port == 0) {
    port = use_https ? INTERNET_DEFAULT_HTTPS_PORT : INTERNET_DEFAULT_HTTP_PORT;
  }

  std::wstring hostname_str(hostname, url_comp.dwHostNameLength);
  std::wstring path_str(path, url_comp.dwUrlPathLength);

  HINTERNET hSession = (HINTERNET)m_hSession;
  HINTERNET hConnect = (HINTERNET)m_hConnect;

  if (m_cached_url != url || !hSession || !hConnect) {
    CloseConnection();
    bool is_localhost =
        (hostname_str == L"localhost" || hostname_str == L"127.0.0.1");
    DWORD access_type = is_localhost ? WINHTTP_ACCESS_TYPE_NO_PROXY
                                     : WINHTTP_ACCESS_TYPE_DEFAULT_PROXY;
    hSession =
        WinHttpOpen(L"Weasel IME/1.0", access_type,
                    is_localhost ? (LPCWSTR)WINHTTP_NO_PROXY_NAME : NULL,
                    is_localhost ? (LPCWSTR)WINHTTP_NO_PROXY_BYPASS : NULL, 0);
    if (!hSession) {
      return false;
    }
    DWORD timeout = 10000;
    WinHttpSetTimeouts(hSession, timeout, timeout, timeout, timeout);
    hConnect = WinHttpConnect(hSession, hostname_str.c_str(), port, 0);
    if (!hConnect) {
      WinHttpCloseHandle(hSession);
      return false;
    }
    m_hSession = hSession;
    m_hConnect = hConnect;
    m_cached_url = url;
  }

  HINTERNET hRequest = WinHttpOpenRequest(
      hConnect, L"POST", path_str.c_str(), NULL, WINHTTP_NO_REFERER,
      WINHTTP_DEFAULT_ACCEPT_TYPES, use_https ? WINHTTP_FLAG_SECURE : 0);
  if (!hRequest) {
    CloseConnection();
    return false;
  }

  std::wstring headers;
  if (!HasHeaderNamed(m_extra_headers, "Content-Type")) {
    headers += L"Content-Type: application/json\r\n";
  }
  if (!m_api_key.empty() && !HasHeaderNamed(m_extra_headers, "Authorization")) {
    std::wstring api_key_w = u8tow(m_api_key);
    headers += L"Authorization: Bearer " + api_key_w + L"\r\n";
  }
  for (const auto& header : m_extra_headers) {
    headers += u8tow(header.first) + L": " + u8tow(header.second) + L"\r\n";
  }

  if (!WinHttpSendRequest(
          hRequest, headers.c_str(), (DWORD)-1, (LPVOID)request_body.c_str(),
          (DWORD)request_body.length(), (DWORD)request_body.length(), 0)) {
    WinHttpCloseHandle(hRequest);
    CloseConnection();
    return false;
  }

  if (!WinHttpReceiveResponse(hRequest, NULL)) {
    WinHttpCloseHandle(hRequest);
    CloseConnection();
    return false;
  }

  DWORD bytes_available = 0;
  response_body.clear();
  while (WinHttpQueryDataAvailable(hRequest, &bytes_available) &&
         bytes_available > 0) {
    std::vector<char> buffer(bytes_available);
    DWORD bytes_read = 0;
    if (WinHttpReadData(hRequest, buffer.data(), bytes_available,
                        &bytes_read)) {
      response_body.append(buffer.data(), bytes_read);
    } else {
      break;
    }
  }

  WinHttpCloseHandle(hRequest);
  return !response_body.empty();
}

std::vector<std::wstring> OpenAICompatibleProvider::ParseResponse(
    const std::string& json_response) {
  std::vector<std::wstring> candidates;

  std::string content;
  try {
    boost::property_tree::ptree root;
    std::istringstream json_stream(json_response);
    boost::property_tree::read_json(json_stream, root);

    const char* field_names[] = {"content", "text", "generated_text",
                                 "responses"};
    for (const char* field_name : field_names) {
      if (FindFirstStringFieldByName(root, field_name, content)) {
        break;
      }
    }
  } catch (const boost::property_tree::json_parser_error&) {
    return candidates;
  }

  if (content.empty()) {
    return candidates;
  }

  std::wstring content_w = u8tow(content);

  // 按空格分割
  std::wstringstream ss(content_w);
  std::wstring word;
  while (ss >> word) {
    if (!word.empty()) {
      candidates.push_back(word);
    }
  }

  return candidates;
}

// 全局开发终端实例（供LLMProvider使用）
DevConsole* g_dev_console = nullptr;
