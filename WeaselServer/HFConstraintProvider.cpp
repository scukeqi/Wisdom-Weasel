#include "stdafx.h"
#include "LLMProvider.h"
#include "DevConsole.h"
#include <WeaselUtility.h>
#include <rime_api.h>
#include <winhttp.h>
#include <algorithm>
#include <sstream>

#pragma comment(lib, "winhttp.lib")

namespace {

std::string EscapeJsonString(const std::string& s) {
  std::string out;
  for (char c : s) {
    if (c == '"')
      out += "\\\"";
    else if (c == '\\')
      out += "\\\\";
    else if (c == '\n')
      out += "\\n";
    else if (c == '\r')
      out += "\\r";
    else if (c == '\t')
      out += "\\t";
    else
      out += c;
  }
  return out;
}

}  // namespace

HFConstraintProvider::HFConstraintProvider()
    : m_enabled(false),
      m_api_url("http://localhost:8000/v1/generate/completions"),
      m_hSession(nullptr),
      m_hConnect(nullptr) {}

HFConstraintProvider::~HFConstraintProvider() {
  CloseConnection();
}

void HFConstraintProvider::CloseConnection() {
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

bool HFConstraintProvider::LoadConfig(const std::string& config_name) {
  extern DevConsole* g_dev_console;

  RimeApi* rime_api = rime_get_api();
  if (!rime_api) {
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] LoadConfig失败: rime_api未初始化");
    }
    return false;
  }

  if (g_dev_console && g_dev_console->IsEnabled()) {
    g_dev_console->WriteLine(L"[LLM] 开始从配置文件加载 HF Constraint 配置: " +
                             u8tow(config_name));
  }

  RimeConfig config = {NULL};
  if (!rime_api->config_open(config_name.c_str(), &config)) {
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] LoadConfig失败: 无法打开配置文件 " +
                              u8tow(config_name));
    }
    return false;
  }

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

  const int BUF_SIZE = 512;
  char buffer[BUF_SIZE + 1] = {0};

  bool found_api_url =
      rime_api->config_get_string(&config, "llm/hf_constraint/api_url", buffer, BUF_SIZE);
  if (found_api_url) {
    m_api_url = buffer;
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] 找到配置项 llm/hf_constraint/api_url = " +
                              u8tow(m_api_url));
    }
  } else {
    m_api_url = "http://localhost:8000/v1/generate/completions";
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] 未找到配置项 llm/hf_constraint/api_url，使用默认值 = " +
                              u8tow(m_api_url));
    }
  }

  CloseConnection();  // URL 可能变化，下次请求时重建连接
  rime_api->config_close(&config);

  if (g_dev_console && g_dev_console->IsEnabled()) {
    g_dev_console->WriteLine(L"[LLM] HF Constraint 配置加载成功");
  }

  return true;
}

std::vector<std::wstring> HFConstraintProvider::ExecuteRequest(
    const LLMRequest& request,
    const LLMPartialCallback& /*on_partial*/) {
  std::vector<std::wstring> candidates;

  if (!IsAvailable() || !llm_request::IsExecutable(request)) {
    return candidates;
  }

  std::wstring prompt_text = llm_request::BuildCompactPrompt(request);
  if (prompt_text.empty()) {
    return candidates;
  }
  std::string prompt_utf8 = wtou8(prompt_text);
  std::string escaped_prompt = EscapeJsonString(prompt_utf8);

  std::vector<std::string> constraint_parts =
      llm_request::BuildPinyinConstraintParts(request);

  std::ostringstream json;
  json << "{\"prompt\":\"" << escaped_prompt << "\",\"pinyin_constraints\":[";
  for (size_t i = 0; i < constraint_parts.size(); ++i) {
    if (i > 0)
      json << ",";
    json << "\"" << EscapeJsonString(constraint_parts[i]) << "\"";
  }
  json << "]}";

  std::string request_body = json.str();

  extern DevConsole* g_dev_console;
  if (g_dev_console && g_dev_console->IsEnabled()) {
    g_dev_console->WriteLine(L"[LLM] [HF Constraint] 发送请求");
    g_dev_console->WriteLine(L"  请求类型: " +
                             llm_request::GetRequestTypeName(request.type));
    g_dev_console->WriteLine(L"  上下文: " + request.context);
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

  std::string response_body;
  if (!ExecuteHttpRequest(m_api_url, request_body, response_body)) {
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] [HF Constraint] 请求失败");
    }
    return candidates;
  }

  if (g_dev_console && g_dev_console->IsEnabled()) {
    g_dev_console->WriteLine(L"[LLM] [HF Constraint] 收到响应");
    g_dev_console->WriteLine(L"  响应内容: " + u8tow(response_body));
  }

  candidates = ParseResponse(response_body);

  if (candidates.size() > request.max_candidates) {
    candidates.resize(request.max_candidates);
  }

  return candidates;
}

bool HFConstraintProvider::IsAvailable() const {
  return m_enabled && !m_api_url.empty();
}

bool HFConstraintProvider::ExecuteHttpRequest(const std::string& url,
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
    bool is_localhost = (hostname_str == L"localhost" || hostname_str == L"127.0.0.1");
    DWORD access_type = is_localhost ? WINHTTP_ACCESS_TYPE_NO_PROXY : WINHTTP_ACCESS_TYPE_DEFAULT_PROXY;
    hSession = WinHttpOpen(
        L"Weasel IME/1.0", access_type,
        is_localhost ? (LPCWSTR)WINHTTP_NO_PROXY_NAME : NULL,
        is_localhost ? (LPCWSTR)WINHTTP_NO_PROXY_BYPASS : NULL, 0);
    if (!hSession) {
      return false;
    }
    // HF 后端冷启动首个请求可能超过 10 秒；放宽接收超时避免首请求被打断。
    WinHttpSetTimeouts(hSession, 10000, 10000, 15000, 60000);
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
      WINHTTP_DEFAULT_ACCEPT_TYPES,
      use_https ? WINHTTP_FLAG_SECURE : 0);
  if (!hRequest) {
    CloseConnection();
    return false;
  }

  std::wstring headers = L"Content-Type: application/json\r\n";
  if (!WinHttpSendRequest(hRequest, headers.c_str(), (DWORD)-1,
                          (LPVOID)request_body.c_str(),
                          (DWORD)request_body.length(),
                          (DWORD)request_body.length(),
                          0)) {
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

std::vector<std::wstring> HFConstraintProvider::ParseResponse(
    const std::string& json_response) {
  std::vector<std::wstring> candidates;

  // 解析 "responses" 字段，格式如 {"responses":"螃蟹 披 苹果 泡 葡萄"}
  const char* field_names[] = {"\"responses\"", "\"text\"", "\"generated_text\"", "\"content\""};
  size_t content_pos = std::string::npos;

  for (const char* field : field_names) {
    content_pos = json_response.find(field);
    if (content_pos != std::string::npos)
      break;
  }

  if (content_pos == std::string::npos) {
    return candidates;
  }

  size_t colon_pos = json_response.find(':', content_pos);
  if (colon_pos == std::string::npos) {
    return candidates;
  }

  size_t quote_start = json_response.find('"', colon_pos);
  if (quote_start == std::string::npos) {
    return candidates;
  }

  size_t quote_end = quote_start + 1;
  while (quote_end < json_response.size()) {
    size_t next = json_response.find('"', quote_end);
    if (next == std::string::npos)
      break;
    if (json_response[next - 1] != '\\') {
      quote_end = next;
      break;
    }
    quote_end = next + 1;
  }

  if (quote_end >= json_response.size()) {
    return candidates;
  }

  std::string content = json_response.substr(quote_start + 1,
                                             quote_end - quote_start - 1);
  std::wstring content_w = u8tow(content);

  std::wstringstream ss(content_w);
  std::wstring word;
  while (ss >> word) {
    if (!word.empty()) {
      candidates.push_back(word);
    }
  }

  return candidates;
}
