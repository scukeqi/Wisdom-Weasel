#include "stdafx.h"
#include "LLMProvider.h"
#include "ConfigJsonUtils.h"
#include "DevConsole.h"
#include <WeaselUtility.h>
#include <rime_api.h>
#include <winhttp.h>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <cstring>

#pragma comment(lib, "winhttp.lib")

namespace {

void SkipJsonWhitespace(const std::string& text, size_t& pos) {
  while (pos < text.size() &&
         std::isspace(static_cast<unsigned char>(text[pos]))) {
    ++pos;
  }
}

void AppendUtf8CodePoint(std::string& output, uint32_t codepoint) {
  if (codepoint <= 0x7F) {
    output.push_back(static_cast<char>(codepoint));
  } else if (codepoint <= 0x7FF) {
    output.push_back(static_cast<char>(0xC0 | (codepoint >> 6)));
    output.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
  } else if (codepoint <= 0xFFFF) {
    output.push_back(static_cast<char>(0xE0 | (codepoint >> 12)));
    output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
    output.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
  } else {
    output.push_back(static_cast<char>(0xF0 | (codepoint >> 18)));
    output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3F)));
    output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3F)));
    output.push_back(static_cast<char>(0x80 | (codepoint & 0x3F)));
  }
}

bool ParseJsonStringLiteral(const std::string& text,
                            size_t quote_pos,
                            std::string& value,
                            size_t* next_pos = nullptr) {
  if (quote_pos >= text.size() || text[quote_pos] != '"') {
    return false;
  }

  value.clear();
  for (size_t i = quote_pos + 1; i < text.size(); ++i) {
    const char ch = text[i];
    if (ch == '"') {
      if (next_pos) {
        *next_pos = i + 1;
      }
      return true;
    }
    if (ch != '\\') {
      value.push_back(ch);
      continue;
    }

    if (++i >= text.size()) {
      return false;
    }

    switch (text[i]) {
      case '"':
      case '\\':
      case '/':
        value.push_back(text[i]);
        break;
      case 'b':
        value.push_back('\b');
        break;
      case 'f':
        value.push_back('\f');
        break;
      case 'n':
        value.push_back('\n');
        break;
      case 'r':
        value.push_back('\r');
        break;
      case 't':
        value.push_back('\t');
        break;
      case 'u': {
        if (i + 4 >= text.size()) {
          return false;
        }
        uint32_t codepoint = 0;
        for (size_t hex_pos = i + 1; hex_pos <= i + 4; ++hex_pos) {
          codepoint <<= 4;
          const unsigned char hex_char = static_cast<unsigned char>(text[hex_pos]);
          if (hex_char >= '0' && hex_char <= '9') {
            codepoint |= (hex_char - '0');
          } else if (hex_char >= 'a' && hex_char <= 'f') {
            codepoint |= (hex_char - 'a' + 10);
          } else if (hex_char >= 'A' && hex_char <= 'F') {
            codepoint |= (hex_char - 'A' + 10);
          } else {
            return false;
          }
        }
        AppendUtf8CodePoint(value, codepoint);
        i += 4;
        break;
      }
      default:
        value.push_back(text[i]);
        break;
    }
  }

  return false;
}

bool ExtractJsonStringField(const std::string& json,
                            const std::string& field_name,
                            std::string& value) {
  const std::string key = "\"" + field_name + "\"";
  size_t key_pos = json.find(key);
  while (key_pos != std::string::npos) {
    size_t pos = key_pos + key.size();
    SkipJsonWhitespace(json, pos);
    if (pos >= json.size() || json[pos] != ':') {
      key_pos = json.find(key, key_pos + key.size());
      continue;
    }
    ++pos;
    SkipJsonWhitespace(json, pos);
    if (pos >= json.size() || json[pos] != '"') {
      key_pos = json.find(key, key_pos + key.size());
      continue;
    }
    return ParseJsonStringLiteral(json, pos, value);
  }
  return false;
}

bool ParseJsonStringArray(const std::string& json,
                          size_t array_pos,
                          std::vector<std::string>& values) {
  values.clear();
  if (array_pos >= json.size() || json[array_pos] != '[') {
    return false;
  }

  size_t pos = array_pos + 1;
  while (pos < json.size()) {
    SkipJsonWhitespace(json, pos);
    if (pos >= json.size()) {
      return false;
    }
    if (json[pos] == ']') {
      return true;
    }
    if (json[pos] != '"') {
      return false;
    }

    std::string item;
    if (!ParseJsonStringLiteral(json, pos, item, &pos)) {
      return false;
    }
    values.push_back(item);

    SkipJsonWhitespace(json, pos);
    if (pos >= json.size()) {
      return false;
    }
    if (json[pos] == ',') {
      ++pos;
      continue;
    }
    if (json[pos] == ']') {
      return true;
    }
    return false;
  }

  return false;
}

std::string TrimAsciiWhitespace(const std::string& text) {
  size_t begin = 0;
  while (begin < text.size() &&
         std::isspace(static_cast<unsigned char>(text[begin]))) {
    ++begin;
  }

  size_t end = text.size();
  while (end > begin &&
         std::isspace(static_cast<unsigned char>(text[end - 1]))) {
    --end;
  }

  return text.substr(begin, end - begin);
}

std::string StripMarkdownCodeFence(const std::string& text) {
  const std::string trimmed = TrimAsciiWhitespace(text);
  if (trimmed.rfind("```", 0) != 0) {
    return trimmed;
  }

  const size_t first_newline = trimmed.find('\n');
  if (first_newline == std::string::npos) {
    return trimmed;
  }
  const size_t fence_end = trimmed.rfind("```");
  if (fence_end == std::string::npos || fence_end <= first_newline) {
    return trimmed;
  }
  return TrimAsciiWhitespace(
      trimmed.substr(first_newline + 1, fence_end - first_newline - 1));
}

std::vector<std::wstring> ParsePlainTextCandidates(const std::string& content) {
  std::vector<std::wstring> candidates;
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

std::vector<std::wstring> ParseStructuredCandidates(const std::string& content) {
  std::vector<std::wstring> candidates;
  const std::string trimmed = StripMarkdownCodeFence(content);
  if (trimmed.empty()) {
    return candidates;
  }

  auto append_candidates = [&candidates](const std::vector<std::string>& items) {
    for (const auto& item : items) {
      if (!item.empty()) {
        candidates.push_back(u8tow(item));
      }
    }
  };

  std::string normalized = trimmed;
  const size_t object_begin = trimmed.find('{');
  const size_t object_end = trimmed.rfind('}');
  if (object_begin != std::string::npos && object_end != std::string::npos &&
      object_begin < object_end) {
    normalized = trimmed.substr(object_begin, object_end - object_begin + 1);
  }

  std::vector<std::string> items;
  if (normalized.front() == '[' && ParseJsonStringArray(normalized, 0, items)) {
    append_candidates(items);
    return candidates;
  }

  const std::string key = "\"candidates\"";
  size_t key_pos = normalized.find(key);
  if (key_pos != std::string::npos) {
    size_t array_pos = normalized.find('[', key_pos + key.size());
    if (array_pos != std::string::npos &&
        ParseJsonStringArray(normalized, array_pos, items)) {
      append_candidates(items);
      return candidates;
    }
  }

  return candidates;
}

std::wstring BuildLegacyPredictionPrompt(const std::wstring& context,
                                         const std::wstring& current_input,
                                         size_t max_candidates) {
  return L"你是一个智能中文输入法，请根据以下上下文和当前输入，预测接下来最可能出"
         L"现的" +
         std::to_wstring(max_candidates) +
         L"个候选词。\n\n"
         L"要求：\n"
         L"1. 只返回候选词，不要任何解释或标点\n"
         L"2. 候选词之间用单个空格分隔\n"
         L"3. 按可能性从高到低排列\n"
         L"4. 如果上下文为空或无关，仅基于当前输入预测\n"
         L"5. 确保候选词都是有效的中文词汇或常用短语\n"
         L"6. 返回词数严格不超过" +
         std::to_wstring(max_candidates) +
         L"个\n\n"
         L"上下文：\"" + context + L"\"\n"
         L"当前输入：\"" + current_input + L"\"\n"
         L"候选词：";
}

std::wstring BuildJsonPredictionPrompt(const std::wstring& context,
                                       const std::wstring& current_input,
                                       size_t max_candidates) {
  return L"你是一个智能中文输入法，请根据以下上下文和当前输入预测候选词。\n\n"
         L"请仅返回一个 JSON 对象，不要输出 Markdown 代码块，不要输出额外解释。\n"
         L"JSON 格式必须是：{\"candidates\":[\"候选1\",\"候选2\"]}\n"
         L"要求：\n"
         L"1. key 必须为 candidates\n"
         L"2. candidates 必须是字符串数组\n"
         L"3. 候选词按可能性从高到低排列\n"
         L"4. candidates 数量严格不超过" +
         std::to_wstring(max_candidates) +
         L"\n"
         L"5. 每个候选词都必须是有效的中文词汇或常用短语\n"
         L"6. 如果上下文为空或无关，仅基于当前输入预测\n\n"
         L"上下文：\"" + context + L"\"\n"
         L"当前输入：\"" + current_input + L"\"";
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

OpenAICompatibleProvider::OpenAICompatibleProvider()
    : m_enabled(false),
      m_max_tokens(10),
      m_temperature(0.7),
      m_has_custom_response_format(false),
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
    m_has_custom_response_format = false;
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

  std::string response_format_json;
  m_has_custom_response_format =
      weasel::config_json::SerializeConfigValueToJson(
          rime_api, &config, "llm/openai/extra_body/response_format",
          response_format_json);

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

std::vector<std::wstring> OpenAICompatibleProvider::PredictCandidates(
    const std::wstring& context,
    const std::wstring& current_input,
    size_t max_candidates) {
  std::vector<std::wstring> candidates;

  if (!IsAvailable() || context.empty()) {
    return candidates;
  }
  extern DevConsole* g_dev_console;
  auto send_request = [&](bool use_json_output) -> std::vector<std::wstring> {
    const std::wstring prompt =
        use_json_output
            ? BuildJsonPredictionPrompt(context, current_input, max_candidates)
            : BuildLegacyPredictionPrompt(context, current_input, max_candidates);

    const std::string prompt_utf8 = wtou8(prompt);
    const std::string escaped_prompt =
        weasel::config_json::EscapeJsonString(prompt_utf8);
    const std::string extra_body_members =
        StripJsonObjectBraces(m_extra_body_json);

    std::ostringstream json;
    json << "{"
         << "\"model\":\"" << weasel::config_json::EscapeJsonString(m_model)
         << "\","
         << "\"messages\":["
         << "{\"role\":\"user\",\"content\":\"" << escaped_prompt << "\"}"
         << "],"
         << "\"max_tokens\":" << m_max_tokens << ","
         << "\"temperature\":" << m_temperature;
    if (use_json_output && !m_has_custom_response_format) {
      json << ",\"response_format\":{\"type\":\"json_object\"}";
    }
    if (!extra_body_members.empty()) {
      json << "," << extra_body_members;
    }
    json << "}";

    const std::string request_body = json.str();

    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(use_json_output
                                   ? L"[LLM] 发送预测请求（JSON 约束模式）"
                                   : L"[LLM] 发送预测请求（纯文本回退模式）");
      g_dev_console->WriteLine(L"  上下文: " + context);
      g_dev_console->WriteLine(L"  请求URL: " + u8tow(m_api_url));
      g_dev_console->WriteLine(L"  请求体: " + u8tow(request_body));
    }

    std::string response_body;
    if (!ExecuteRequest(m_api_url, request_body, response_body)) {
      if (g_dev_console && g_dev_console->IsEnabled()) {
        g_dev_console->WriteLine(L"[LLM] 请求失败");
      }
      return {};
    }

    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(L"[LLM] 收到响应");
      g_dev_console->WriteLine(L"  响应内容: " + u8tow(response_body));
    }

    return ParseResponse(response_body);
  };

  candidates = send_request(true);
  if (candidates.empty() && !m_has_custom_response_format) {
    if (g_dev_console && g_dev_console->IsEnabled()) {
      g_dev_console->WriteLine(
          L"[LLM] JSON 约束输出解析失败，回退到纯文本请求重试");
    }
    candidates = send_request(false);
  }

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

bool OpenAICompatibleProvider::ExecuteRequest(const std::string& url,
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
  std::string content;
  if (!ExtractJsonStringField(json_response, "content", content)) {
    return {};
  }

  std::vector<std::wstring> candidates = ParseStructuredCandidates(content);
  if (!candidates.empty()) {
    return candidates;
  }

  return ParsePlainTextCandidates(content);
}

// 全局开发终端实例（供LLMProvider使用）
DevConsole* g_dev_console = nullptr;
