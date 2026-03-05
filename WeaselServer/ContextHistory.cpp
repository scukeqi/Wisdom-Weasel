#include "stdafx.h"
#include "ContextHistory.h"
#include "MemoryCompressor.h"
#include "DevConsole.h"
#include <algorithm>
#include <sstream>
#include <locale>
#include <codecvt>

ContextHistory::ContextHistory(size_t max_size)
    : m_max_size(max_size > 0 ? max_size : 50),
      m_memory_compressor(nullptr),
      m_compressing(false) {
  m_history.reserve(m_max_size);
}

ContextHistory::~ContextHistory() {
  Clear(nullptr);
}

void ContextHistory::AddText(const std::wstring& text, DevConsole* dev_console) {
  if (text.empty()) {
    return;
  }

  std::vector<std::wstring> words = SplitIntoWords(text);
  size_t words_added = 0;
  size_t current_size = 0;
  bool should_trigger_compression = false;

  {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (const auto& word : words) {
      if (word.empty()) continue;
      m_history.push_back(word);
      words_added++;
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
    if (words_added > 0) ss << L" -> 分割为 " << words_added << L" 个词";
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

std::vector<std::wstring> ContextHistory::GetAllHistory() const {
  std::lock_guard<std::mutex> lock(m_mutex);
  return m_history;
}

void ContextHistory::Clear(DevConsole* dev_console) {
  std::lock_guard<std::mutex> lock(m_mutex);
  m_history.clear();
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

