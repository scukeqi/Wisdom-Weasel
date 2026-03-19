#include "stdafx.h"
#include "ContextHistory.h"
#include "MemoryCompressor.h"
#include "DevConsole.h"
#include <algorithm>
#include <sstream>
#include <locale>
#include <codecvt>
#include <set>

ContextHistory::ContextHistory(size_t max_size)
    : m_max_size(max_size > 0 ? max_size : 50),
      m_preference_sequence(0),
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

  std::vector<std::wstring> words = DeduplicateWords(SplitIntoWords(text));
  size_t history_words_added = 0;
  size_t preference_words_added = 0;
  size_t deduplicated_history_words = 0;
  size_t current_size = 0;
  bool should_trigger_compression = false;

  {
    std::lock_guard<std::mutex> lock(m_mutex);
    for (const auto& word : words) {
      if (word.empty()) {
        continue;
      }

      preference_words_added++;

      if (!m_history.empty() && m_history.back() == word) {
        deduplicated_history_words++;
        continue;
      }

      m_history.push_back(word);
      history_words_added++;
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
    RebuildPreferenceStatsUnlocked();
  }

  if (dev_console && dev_console->IsEnabled()) {
    std::wstringstream ss;
    ss << L"[上下文更新] 添加文本: " << text;
    if (preference_words_added > 0) {
      ss << L" -> 去重后 " << preference_words_added << L" 个词";
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
      std::wstring preference_hint = GetPreferenceHint(6);
      if (!preference_hint.empty()) {
        dev_console->WriteLine(L"  偏好词: " + preference_hint);
      }
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

std::wstring ContextHistory::GetPreferenceHint(size_t count) const {
  std::lock_guard<std::mutex> lock(m_mutex);
  if (m_preference_stats.empty() || count == 0) {
    return L"";
  }

  struct RankedPreference {
    std::wstring word;
    size_t count;
    uint64_t last_seen_order;
  };

  std::vector<RankedPreference> ranked_preferences;
  ranked_preferences.reserve(m_preference_stats.size());
  for (const auto& entry : m_preference_stats) {
    ranked_preferences.push_back(
        {entry.first, entry.second.count, entry.second.last_seen_order});
  }

  std::sort(ranked_preferences.begin(), ranked_preferences.end(),
            [](const RankedPreference& lhs, const RankedPreference& rhs) {
              if (lhs.count != rhs.count) {
                return lhs.count > rhs.count;
              }
              if (lhs.last_seen_order != rhs.last_seen_order) {
                return lhs.last_seen_order > rhs.last_seen_order;
              }
              return lhs.word < rhs.word;
            });

  const size_t actual_count = (std::min)(count, ranked_preferences.size());
  std::wstringstream ss;
  for (size_t i = 0; i < actual_count; ++i) {
    if (i > 0) {
      ss << L" ";
    }
    ss << ranked_preferences[i].word;
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
  m_preference_stats.clear();
  m_preference_sequence = 0;
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
  RebuildPreferenceStatsUnlocked();
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

void ContextHistory::RebuildPreferenceStatsUnlocked() {
  m_preference_stats.clear();
  m_preference_sequence = 0;
  for (const auto& word : m_history) {
    if (word.empty()) {
      continue;
    }
    PreferenceStat& stat = m_preference_stats[word];
    stat.count++;
    stat.last_seen_order = ++m_preference_sequence;
  }
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

