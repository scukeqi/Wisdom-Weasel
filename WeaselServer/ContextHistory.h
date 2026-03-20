#pragma once

#include <string>
#include <vector>
#include <mutex>
#include <memory>
#include <functional>
#include <map>
#include <cstdint>

// 前向声明
class DevConsole;
class MemoryCompressor;

// 用户输入上下文历史记录类
// 维护最近 N 个用户输入的词，用于 LLM 预测候选词
// 当超过 max_size 时，异步将最旧的 (max_size/2) 个词通过记忆 LLM 压缩
class ContextHistory {
 public:
  // 构造函数，指定最大记录数量（默认 50）
  explicit ContextHistory(size_t max_size = 50);

  ~ContextHistory();

  // 添加用户提交的文本到历史记录
  // 文本会被分割成词并添加到历史中
  // dev_console: 开发终端实例，用于输出日志（可为nullptr）
  void AddText(const std::wstring& text, DevConsole* dev_console = nullptr);

  // 获取最近N个词作为上下文
  // 返回的字符串用空格分隔
  std::wstring GetRecentContext(size_t count) const;

  // 获取最近的原始连续文本上下文（保留中文自然连接与标点）
  // max_chars 为 0 表示不限制；prefer_sentence_boundary 为 true 时尽量只返回最近一句/当前句片段
  std::wstring GetRecentTextContext(size_t max_chars,
                                    bool prefer_sentence_boundary = true) const;

  // 获取所有历史记录
  std::vector<std::wstring> GetAllHistory() const;

  // 清空历史记录
  // dev_console: 开发终端实例，用于输出日志（可为nullptr）
  void Clear(DevConsole* dev_console = nullptr);

  // 获取当前记录数量
  size_t GetSize() const;

  // 获取最大记录数量
  size_t GetMaxSize() const;

  // 设置记忆压缩器（从 weasel 配置 llm/memory/ 加载），用于超过容量时压缩旧词
  void SetMemoryCompressor(MemoryCompressor* compressor) { m_memory_compressor = compressor; }

 private:
  // 将文本分割成词（简单实现：按空格和标点符号分割）
  std::vector<std::wstring> SplitIntoWords(const std::wstring& text) const;

  // 对输入词进行去重，保留顺序
  std::vector<std::wstring> DeduplicateWords(
      const std::vector<std::wstring>& words) const;

  // 判断字符是否为分隔符
  bool IsSeparator(wchar_t ch) const;

  // 判断字符是否为强句边界
  bool IsStrongSentenceBoundary(wchar_t ch) const;

  // 当 size >= max_size 时尝试异步压缩最旧的 (max_size/2) 个词（不阻塞）
  void TryTriggerCompression(DevConsole* dev_console);

  // 每次压缩取最旧的词数（max_size 的一半）
  size_t GetCompressWordCount() const { return m_max_size / 2; }

  // 获取最旧的 count 个词（在锁内复制）
  std::vector<std::wstring> GetOldestWords(size_t count) const;

  // 用压缩后的词序列替换最旧的 100 个词
  void ReplaceOldestWithCompressed(const std::vector<std::wstring>& compressed,
                                   DevConsole* dev_console);

  mutable std::mutex m_mutex;  // 线程安全锁
  std::vector<std::wstring> m_history;  // 历史记录（线性：最旧在 0，最新在 back）
  std::vector<std::wstring> m_text_history;  // 原始提交片段历史（保留自然文本形态）
  size_t m_max_size;  // 最大记录数量
  MemoryCompressor* m_memory_compressor;  // 记忆压缩 LLM（可为 nullptr）
  bool m_compressing;  // 是否正在压缩，避免重复触发
};

