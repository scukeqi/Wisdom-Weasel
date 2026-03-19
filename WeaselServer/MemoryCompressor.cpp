#include "stdafx.h"
#include "ConfigJsonUtils.h"
#include "MemoryCompressor.h"
#include "DevConsole.h"
#include <WeaselUtility.h>
#include <rime_api.h>
#include <winhttp.h>
#include <sstream>
#include <future>
#include <cctype>
#include <cstring>

#pragma comment(lib, "winhttp.lib")

namespace {

std::wstring WordsToSpaceSeparated(const std::vector<std::wstring>& words) {
  std::wstringstream ss;
  for (size_t i = 0; i < words.size(); ++i) {
    if (i > 0)
      ss << L" ";
    ss << words[i];
  }
  return ss.str();
}

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

}  // namespace

MemoryCompressor::MemoryCompressor()
    : m_enabled(false),
      m_max_tokens(100),
      m_hSession(nullptr),
      m_hConnect(nullptr) {}

MemoryCompressor::~MemoryCompressor() {
  CloseConnection();
}

void MemoryCompressor::CloseConnection() {
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

bool MemoryCompressor::LoadConfig(const std::string& config_name) {
  extern DevConsole* g_dev_console;

  RimeApi* rime_api = rime_get_api();
  if (!rime_api) {
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[记忆压缩] LoadConfig失败: rime_api未初始化");
    }
    return false;
  }

  RimeConfig config = {NULL};
  if (!rime_api->config_open(config_name.c_str(), &config)) {
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[记忆压缩] 无法打开配置文件，记忆压缩未启用");
    }
    return false;
  }

  const int BUF_SIZE = 512;
  char buffer[BUF_SIZE + 1] = {0};

  Bool enabled = false;
  bool found =
      rime_api->config_get_bool(&config, "llm/memory/enabled", &enabled);
  m_enabled = found && !!enabled;

  if (!m_enabled) {
    rime_api->config_close(&config);
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(
          L"[记忆压缩] 未启用（在 weasel.yaml 中设置 llm/memory/enabled: true "
          L"并配置 api_url 以启用）");
    }
    return true;  // 未启用也算加载成功
  }

  found = rime_api->config_get_string(&config, "llm/memory/api_url", buffer,
                                      BUF_SIZE);
  if (found) {
    m_api_url = buffer;
  } else {
    m_api_url.clear();
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(
          L"[记忆压缩] 未配置 llm/memory/api_url，记忆压缩不可用");
    }
  }

  found = rime_api->config_get_string(&config, "llm/memory/api_key", buffer,
                                      BUF_SIZE);
  m_api_key = found ? buffer : "";

  found = rime_api->config_get_string(&config, "llm/memory/model", buffer,
                                      BUF_SIZE);
  m_model = found ? buffer : "gpt-3.5-turbo";

  int max_tokens = 100;
  if (rime_api->config_get_int(&config, "llm/memory/max_tokens", &max_tokens)) {
    m_max_tokens = max_tokens;
  }

  m_extra_body_json.clear();
  weasel::config_json::SerializeConfigMapToJsonObject(
      rime_api, &config, "llm/memory/extra_body", m_extra_body_json);

  m_extra_headers.clear();
  weasel::config_json::LoadConfigStringMap(
      rime_api, &config, "llm/memory/extra_headers", m_extra_headers);

  CloseConnection();
  rime_api->config_close(&config);

  if (g_dev_console && g_dev_console->IsEnabled() && IsAvailable()) {
    g_dev_console->WriteLine(L"[记忆压缩] 已启用，api_url = " +
                             u8tow(m_api_url));
    if (!m_extra_body_json.empty()) {
      g_dev_console->WriteLine(L"[记忆压缩] extra_body = " +
                               u8tow(m_extra_body_json));
    }
    if (!m_extra_headers.empty()) {
      std::wstring headers_summary;
      for (size_t i = 0; i < m_extra_headers.size(); ++i) {
        if (i > 0) {
          headers_summary += L", ";
        }
        headers_summary += u8tow(m_extra_headers[i].first) + L"=" +
                           u8tow(m_extra_headers[i].second);
      }
      g_dev_console->WriteLine(L"[记忆压缩] extra_headers = " +
                               headers_summary);
    }
  }
  return true;
}

void MemoryCompressor::CompressAsync(
    const std::vector<std::wstring>& words,
    std::function<void(std::vector<std::wstring>)> callback) {
  if (!IsAvailable() || words.empty()) {
    if (callback)
      callback(std::vector<std::wstring>());
    return;
  }

  std::wstring words_str = WordsToSpaceSeparated(words);
  std::string prompt_utf8 = wtou8(
      L"请将以下用户输入历史的词序列压缩为更短的摘要，保留关键信息。"
      L"只输出压缩后的词，词语间用单个空格分隔，不要任何解释或标点，不超过10个"
      L"词。\n\n词序列：\"" +
      words_str + L"\"");

  std::string escaped_prompt =
      weasel::config_json::EscapeJsonString(prompt_utf8);
  std::string extra_body_members = StripJsonObjectBraces(m_extra_body_json);

  std::ostringstream json;
  json << "{\"model\":\"" << weasel::config_json::EscapeJsonString(m_model)
       << "\","
       << "\"messages\":[{\"role\":\"user\",\"content\":\"" << escaped_prompt
       << "\"}],"
       << "\"max_tokens\":" << m_max_tokens;
  if (!extra_body_members.empty()) {
    json << "," << extra_body_members;
  }
  json << "}";
  std::string request_body = json.str();
  std::string api_url = m_api_url;
  std::string api_key = m_api_key;
  std::vector<std::pair<std::string, std::string>> extra_headers =
      m_extra_headers;

  std::async(std::launch::async,
             [this, request_body, api_url, api_key, extra_headers, callback]() {
               std::string response_body;
               if (!ExecuteRequestOneShot(api_url, api_key, extra_headers,
                                          request_body, response_body)) {
                 if (callback)
                   callback(std::vector<std::wstring>());
                 return;
               }
               std::vector<std::wstring> result = ParseResponse(response_body);
               if (callback)
                 callback(result);
             });
}

bool MemoryCompressor::ExecuteRequest(const std::string& url,
                                      const std::string& request_body,
                                      std::string& response_body) {
  return ExecuteRequestOneShot(url, m_api_key, m_extra_headers, request_body,
                               response_body);
}

bool MemoryCompressor::ExecuteRequestOneShot(
    const std::string& url,
    const std::string& api_key,
    const std::vector<std::pair<std::string, std::string>>& extra_headers,
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

  bool is_localhost =
      (hostname_str == L"localhost" || hostname_str == L"127.0.0.1");
  DWORD access_type = is_localhost ? WINHTTP_ACCESS_TYPE_NO_PROXY
                                   : WINHTTP_ACCESS_TYPE_DEFAULT_PROXY;
  HINTERNET hSession =
      WinHttpOpen(L"Weasel IME Memory/1.0", access_type,
                  is_localhost ? (LPCWSTR)WINHTTP_NO_PROXY_NAME : NULL,
                  is_localhost ? (LPCWSTR)WINHTTP_NO_PROXY_BYPASS : NULL, 0);
  if (!hSession)
    return false;
  DWORD timeout = 15000;
  WinHttpSetTimeouts(hSession, timeout, timeout, timeout, timeout);
  HINTERNET hConnect = WinHttpConnect(hSession, hostname_str.c_str(), port, 0);
  if (!hConnect) {
    WinHttpCloseHandle(hSession);
    return false;
  }

  HINTERNET hRequest = WinHttpOpenRequest(
      hConnect, L"POST", path_str.c_str(), NULL, WINHTTP_NO_REFERER,
      WINHTTP_DEFAULT_ACCEPT_TYPES, use_https ? WINHTTP_FLAG_SECURE : 0);
  if (!hRequest) {
    WinHttpCloseHandle(hConnect);
    WinHttpCloseHandle(hSession);
    return false;
  }

  std::wstring headers;
  if (!HasHeaderNamed(extra_headers, "Content-Type")) {
    headers += L"Content-Type: application/json\r\n";
  }
  if (!api_key.empty() && !HasHeaderNamed(extra_headers, "Authorization")) {
    headers += L"Authorization: Bearer " + u8tow(api_key) + L"\r\n";
  }
  for (const auto& header : extra_headers) {
    headers += u8tow(header.first) + L": " + u8tow(header.second) + L"\r\n";
  }

  if (!WinHttpSendRequest(
          hRequest, headers.c_str(), (DWORD)-1, (LPVOID)request_body.c_str(),
          (DWORD)request_body.length(), (DWORD)request_body.length(), 0)) {
    WinHttpCloseHandle(hRequest);
    WinHttpCloseHandle(hConnect);
    WinHttpCloseHandle(hSession);
    return false;
  }

  if (!WinHttpReceiveResponse(hRequest, NULL)) {
    WinHttpCloseHandle(hRequest);
    WinHttpCloseHandle(hConnect);
    WinHttpCloseHandle(hSession);
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
  WinHttpCloseHandle(hConnect);
  WinHttpCloseHandle(hSession);
  return !response_body.empty();
}

std::vector<std::wstring> MemoryCompressor::ParseResponse(
    const std::string& json_response) {
  std::vector<std::wstring> words;
  size_t content_pos = json_response.find("\"content\"");
  if (content_pos == std::string::npos)
    return words;
  size_t colon_pos = json_response.find(':', content_pos);
  if (colon_pos == std::string::npos)
    return words;
  size_t quote_start = json_response.find('"', colon_pos);
  if (quote_start == std::string::npos)
    return words;
  size_t quote_end = json_response.find('"', quote_start + 1);
  if (quote_end == std::string::npos)
    return words;
  std::string content =
      json_response.substr(quote_start + 1, quote_end - quote_start - 1);
  std::wstring content_w = u8tow(content);
  std::wstringstream ss(content_w);
  std::wstring word;
  while (ss >> word) {
    if (!word.empty())
      words.push_back(word);
  }
  return words;
}
