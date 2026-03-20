#include "stdafx.h"
#include "ContextHistory.h"
#include "MemoryCompressor.h"
#include "DevConsole.h"
#include <algorithm>
#include <sstream>
#include <locale>
#include <codecvt>
#include <set>
#include <cwctype>

ContextHistory::ContextHistory(size_t max_size)
    : m_max_size(max_size > 0 ? max_size : 50),
      m_memory_compressor(nullptr),
      m_compressing(false) {
  m_history.reserve(m_max_size);
  m_text_history.reserve(m_max_size);
}

ContextHistory::~ContextHistory() {
  Clear(nullptr);
}

void ContextHistory::AddText(const std::wstring& text, DevConsole* dev_console) {
  if (text.empty()) {
    return;
  }

  std::vector<std::wstring> words = DeduplicateWords(SplitIntoWords(text));
  std::wstring trimmed_text = text;
  while (!trimmed_text.empty() && std::iswspace(trimmed_text.front())) {
    trimmed_text.erase(trimmed_text.begin());
  }
  while (!trimmed_text.empty() && std::iswspace(trimmed_text.back())) {
    trimmed_text.pop_back();
  }
  size_t history_words_added = 0;
  size_t deduplicated_input_words = words.size();
  size_t deduplicated_history_words = 0;
  size_t current_size = 0;
  bool should_trigger_compression = false;

  {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (const auto& word : words) {
      if (word.empty()) {
        continue;
      }

      if (!m_history.empty() && m_history.back() == word) {
        deduplicated_history_words++;
        continue;
      }

      m_history.push_back(word);
      history_words_added++;
    }
    if (!trimmed_text.empty()) {
      if (m_text_history.empty() || m_text_history.back() != trimmed_text) {
        m_text_history.push_back(trimmed_text);
      }
      while (m_text_history.size() > m_max_size) {
        m_text_history.erase(m_text_history.begin());
      }
    }
    if (!m_memory_compressor || !m_memory_compressor->IsAvailable()) {
      size_t batch = GetCompressWordCount();
      if (batch == 0) batch = 1;
      while (m_history.size() > m_max_size) {
        size_t erase_count = (std::min)(batch, m_history.size());
        m_history.erase(m_history.begin(), m_history.begin() + erase_count);
      }
    } else {
      while (m_history.size() > m_max_size) {
        m_history.erase(m_history.begin());
      }
    }
    current_size = m_history.size();
    if (current_size >= m_max_size && m_memory_compressor &&
        m_memory_compressor->IsAvailable() && !m_compressing &&
        m_history.size() >= GetCompressWordCount()) {
      should_trigger_compression = true;
    }
  }

  if (dev_console && dev_console->IsEnabled()) {
    std::wstringstream ss;
    ss << L"[上下文更新] 添加文本: " << text;
    if (deduplicated_input_words > 0) {
      ss << L" -> 去重后 " << deduplicated_input_words << L" 个词";
    }
    if (deduplicated_history_words > 0) {
      ss << L"（连续重复跳过 " << deduplicated_history_words << L" 个）";
    }
    ss << L" | 实际写入历史 " << history_words_added << L" 个词";
    ss << L" | 当前历史记录数: " << current_size;
    dev_console->WriteLine(ss.str());
    if (current_size > 0) {
      std::wstring recent = GetRecentContext(10);
      if (!recent.empty()) dev_console->WriteLine(L"  最近10个词: " + recent);
    }
  }

  if (should_trigger_compression) {
    TryTriggerCompression(dev_console);
  }
}

std::wstring ContextHistory::GetRecentContext(size_t count) const {
  std::lock_guard<std::mutex> lock(m_mutex);
  if (m_history.empty()) return L"";
  size_t actual_count = (std::min)(count, m_history.size());
  size_t start = m_history.size() - actual_count;
  std::wstringstream ss;
  for (size_t i = start; i < m_history.size(); ++i) {
    if (i > start) ss << L" ";
    ss << m_history[i];
  }
  return ss.str();
}

std::wstring ContextHistory::GetRecentTextContext(size_t max_chars,
                                                  bool prefer_sentence_boundary) const {
  std::lock_guard<std::mutex> lock(m_mutex);
  if (m_text_history.empty()) {
    return L"";
  }

  const size_t hard_char_limit = max_chars > 0 ? max_chars : 96;
  std::vector<std::wstring> segments;
  size_t accumulated_chars = 0;
  for (auto it = m_text_history.rbegin(); it != m_text_history.rend(); ++it) {
    if (it->empty()) {
      continue;
    }
    segments.push_back(*it);
    accumulated_chars += it->size();
    if (accumulated_chars >= hard_char_limit) {
      break;
    }
  }

  std::wstring text;
  for (auto it = segments.rbegin(); it != segments.rend(); ++it) {
    text += *it;
  }

  if (hard_char_limit > 0 && text.size() > hard_char_limit) {
    text = text.substr(text.size() - hard_char_limit);
  }

  if (!prefer_sentence_boundary || text.empty()) {
    return text;
  }

  size_t last_non_space = text.find_last_not_of(L" \t\r\n");
  if (last_non_space == std::wstring::npos) {
    return text;
  }

  const bool ends_with_boundary = IsStrongSentenceBoundary(text[last_non_space]);
  size_t boundary_pos = std::wstring::npos;

  if (ends_with_boundary) {
    if (last_non_space > 0) {
      boundary_pos = text.find_last_of(L"。！？!?；;\n", last_non_space - 1);
    }
  } else {
    boundary_pos = text.find_last_of(L"。！？!?；;\n", last_non_space);
  }

  if (boundary_pos != std::wstring::npos && boundary_pos + 1 < text.size()) {
    std::wstring sentence = text.substr(boundary_pos + 1);
    size_t first_non_space = sentence.find_first_not_of(L" \t\r\n");
    if (first_non_space != std::wstring::npos) {
      sentence.erase(0, first_non_space);
    }
    if (!sentence.empty()) {
      return sentence;
    }
  }

  return text;
}

std::vector<std::wstring> ContextHistory::GetAllHistory() const {
  std::lock_guard<std::mutex> lock(m_mutex);
  return m_history;
}

void ContextHistory::Clear(DevConsole* dev_console) {
  std::lock_guard<std::mutex> lock(m_mutex);
  m_history.clear();
  m_text_history.clear();
  m_compressing = false;
  if (dev_console && dev_console->IsEnabled()) {
    dev_console->WriteLine(L"[上下文更新] 历史记录已清空");
  }
}

size_t ContextHistory::GetSize() const {
  std::lock_guard<std::mutex> lock(m_mutex);
  return m_history.size();
}

size_t ContextHistory::GetMaxSize() const {
  return m_max_size;
}

void ContextHistory::TryTriggerCompression(DevConsole* dev_console) {
  size_t compress_count = GetCompressWordCount();
  std::vector<std::wstring> to_compress;
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_compressing || m_history.size() < compress_count ||
        !m_memory_compressor || !m_memory_compressor->IsAvailable()) {
      return;
    }
    to_compress = GetOldestWords(compress_count);
    m_compressing = true;
  }
  if (dev_console && dev_console->IsEnabled()) {
    dev_console->WriteLine(L"[记忆压缩] 异步压缩最旧 " +
                           std::to_wstring(compress_count) + L" 个词");
  }
  m_memory_compressor->CompressAsync(to_compress, [this, dev_console](
      std::vector<std::wstring> compressed) {
    if (compressed.empty()) {
      std::lock_guard<std::mutex> lock(m_mutex);
      m_compressing = false;
      if (dev_console && dev_console->IsEnabled()) {
        dev_console->WriteLine(L"[记忆压缩] 压缩失败或返回为空，保留原词");
      }
      return;
    }
    ReplaceOldestWithCompressed(compressed, dev_console);
  });
}

std::vector<std::wstring> ContextHistory::GetOldestWords(size_t count) const {
  std::vector<std::wstring> result;
  size_t n = (std::min)(count, m_history.size());
  result.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    result.push_back(m_history[i]);
  }
  return result;
}

void ContextHistory::ReplaceOldestWithCompressed(
    const std::vector<std::wstring>& compressed, DevConsole* dev_console) {
  std::lock_guard<std::mutex> lock(m_mutex);
  size_t compress_count = GetCompressWordCount();
  if (m_history.size() < compress_count) return;
  m_history.erase(m_history.begin(), m_history.begin() + compress_count);
  m_history.insert(m_history.begin(), compressed.begin(), compressed.end());
  while (m_history.size() > m_max_size) {
    m_history.erase(m_history.begin());
  }
  while (m_text_history.size() > m_max_size) {
    m_text_history.erase(m_text_history.begin());
  }
  m_compressing = false;
  if (dev_console && dev_console->IsEnabled()) {
    std::wstringstream ss;
    ss << L"[记忆压缩] 完成，压缩为 " << compressed.size() << L" 个词，当前历史: "
       << m_history.size();
    dev_console->WriteLine(ss.str());
  }
}

std::vector<std::wstring> ContextHistory::SplitIntoWords(
    const std::wstring& text) const {
  std::vector<std::wstring> words;
  if (text.empty()) {
    return words;
  }

  std::wstring current_word;
  for (wchar_t ch : text) {
    if (IsSeparator(ch)) {
      if (!current_word.empty()) {
        words.push_back(current_word);
        current_word.clear();
      }
    } else {
      current_word += ch;
    }
  }

  // 添加最后一个词（如果有）
  if (!current_word.empty()) {
    words.push_back(current_word);
  }

  return words;
}

std::vector<std::wstring> ContextHistory::DeduplicateWords(
    const std::vector<std::wstring>& words) const {
  std::vector<std::wstring> deduplicated_words;
  std::set<std::wstring> seen_words;
  deduplicated_words.reserve(words.size());

  for (const auto& word : words) {
    if (word.empty()) {
      continue;
    }
    if (seen_words.insert(word).second) {
      deduplicated_words.push_back(word);
    }
  }

  return deduplicated_words;
}

bool ContextHistory::IsSeparator(wchar_t ch) const {
  // 空格、制表符、换行符
  if (ch == L' ' || ch == L'\t' || ch == L'\n' || ch == L'\r') {
    return true;
  }

  // 常见中文标点符号
  if (ch == L'，' || ch == L'。' || ch == L'、' || ch == L'；' ||
      ch == L'：' || ch == L'？' || ch == L'！' || ch == L'…' ||
      ch == L'—' || ch == L'–' || ch == L'（' || ch == L'）' ||
      ch == L'【' || ch == L'】' || ch == L'《' || ch == L'》') {
    return true;
  }

  // 常见英文标点符号
  if (ch == L',' || ch == L'.' || ch == L';' || ch == L':' ||
      ch == L'?' || ch == L'!' || ch == L'-' || ch == L'_' ||
      ch == L'(' || ch == L')' || ch == L'[' || ch == L']' ||
      ch == L'{' || ch == L'}' || ch == L'"' || ch == L'\'' ||
      ch == L'/' || ch == L'\\') {
    return true;
  }

  return false;
}

bool ContextHistory::IsStrongSentenceBoundary(wchar_t ch) const {
  return ch == L'。' || ch == L'！' || ch == L'？' || ch == L'!' ||
         ch == L'?' || ch == L'；' || ch == L';' || ch == L'\n' ||
         ch == L'\r';
}

