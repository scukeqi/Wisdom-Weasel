#include "stdafx.h"
#include <logging.h>
#include <RimeWithWeasel.h>
#include <StringAlgorithm.hpp>
#include <WeaselConstants.h>
#include <WeaselUtility.h>
#include <FixedWMemStreamBuf.h>

#include <algorithm>
#include <filesystem>
#include <cwctype>
#include <map>
#include <regex>
#include <rime_api.h>

// 判断字符是否为分隔符（仅空格/标点），用于决定 commit 是否触发进入 LLM 预测模式
static bool IsSeparatorOrPunctuation(wchar_t ch) {
  if (ch == L' ' || ch == L'\t' || ch == L'\n' || ch == L'\r')
    return true;
  // 常见中文标点
  if (ch == L'，' || ch == L'。' || ch == L'、' || ch == L'；' ||
      ch == L'：' || ch == L'？' || ch == L'！' || ch == L'…' ||
      ch == L'—' || ch == L'–' || ch == L'（' || ch == L'）' ||
      ch == L'【' || ch == L'】' || ch == L'《' || ch == L'》' ||
      ch == L'“' || ch == L'”' || ch == L'‘' || ch == L'’')
    return true;
  // 常见英文/通用标点
  if (ch == L',' || ch == L'.' || ch == L';' || ch == L':' ||
      ch == L'?' || ch == L'!' || ch == L'-' || ch == L'_' ||
      ch == L'(' || ch == L')' || ch == L'[' || ch == L']' ||
      ch == L'{' || ch == L'}' || ch == L'"' || ch == L'\'' ||
      ch == L'/' || ch == L'\\' || ch == L'*' || ch == L'#' ||
      ch == L'@' || ch == L'$' || ch == L'%' || ch == L'^' ||
      ch == L'&' || ch == L'+' || ch == L'=' || ch == L'~' ||
      ch == L'`' || ch == L'|' || ch == L'<' || ch == L'>')
    return true;
  return false;
}

// 文本是否包含至少一个“有意义”字符（非纯标点/空格），用于避免仅输入符号时误入 LLM 预测模式
static bool CommitHasMeaningfulContent(const std::wstring& text) {
  for (wchar_t ch : text) {
    if (!IsSeparatorOrPunctuation(ch))
      return true;
  }
  return false;
}

// 包含 ContextHistory 的完整定义（需要调用其方法）
#include "../WeaselServer/ContextHistory.h"
// 包含 LLMProvider 的完整定义
#include "../WeaselServer/LLMProvider.h"
// 包含 DevConsole 的完整定义（需要调用其方法）
#include "../WeaselServer/DevConsole.h"

static const wchar_t* GetLLMRequestTypeName(LLMRequestType request_type) {
  switch (request_type) {
    case LLMRequestType::NoInputPrediction:
      return L"无输入预测";
    case LLMRequestType::PinyinConstrainedPrediction:
      return L"拼音约束预测";
    case LLMRequestType::RimeReorder:
      return L"Rime 重排";
  }
  return L"未知请求";
}

#define TRANSPARENT_COLOR 0x00000000
#define ARGB2ABGR(value)                                 \
  ((value & 0xff000000) | ((value & 0x000000ff) << 16) | \
   (value & 0x0000ff00) | ((value & 0x00ff0000) >> 16))
#define RGBA2ABGR(value)                                   \
  (((value & 0xff) << 24) | ((value & 0xff000000) >> 24) | \
   ((value & 0x00ff0000) >> 8) | ((value & 0x0000ff00) << 8))
typedef enum { COLOR_ABGR = 0, COLOR_ARGB, COLOR_RGBA } ColorFormat;

#ifdef USE_SHARP_COLOR_CODE
#define HEX_REGEX std::regex("^(0x|#)[0-9a-f]+$", std::regex::icase)
#define TRIMHEAD_REGEX std::regex("0x|#", std::regex::icase)
#else
#define HEX_REGEX std::regex("^0x[0-9a-f]+$", std::regex::icase)
#define TRIMHEAD_REGEX std::regex("0x", std::regex::icase)
#endif
using namespace weasel;
static bool hide_ime_mode_icon = false;

static RimeApi* rime_api;
WeaselSessionId _GenerateNewWeaselSessionId(SessionStatusMap sm, DWORD pid) {
  if (sm.empty())
    return (WeaselSessionId)(pid + 1);
  return (WeaselSessionId)(sm.rbegin()->first + 1);
}

int expand_ibus_modifier(int m) {
  return (m & 0xff) | ((m & 0xff00) << 16);
}

RimeWithWeaselHandler::RimeWithWeaselHandler(UI* ui)
    : m_ui(ui),
      m_active_session(0),
      m_disabled(true),
      m_current_dark_mode(false),
      m_global_ascii_mode(false),
      m_show_notifications_time(1200),
      _UpdateUICallback(NULL),
      m_context_history(nullptr),
      m_dev_console(nullptr),
      m_llm_provider(nullptr),
      m_llm_prediction_mode(false),
      m_pending_llm_commit(L""),
      m_current_llm_candidates_require_rime(false),
      m_current_llm_candidates_enable_rime_reorder(false),
      m_llm_developer_mode(false),
      m_llm_context_recent_words(50),
      m_llm_context_max_chars(0),
      m_llm_input_prediction_debounce_ms(120),
      m_llm_rerank_suppressed_until(0),
      m_last_edit_key_time(0),
      m_consecutive_edit_key_count(0),
      m_has_display_highlight_override(false),
      m_display_highlight_override(0),
      m_last_grave_key_time(0) {
  rime_api = rime_get_api();
  assert(rime_api);
  m_pid = GetCurrentProcessId();
  uint16_t msbit = 0;
  for (auto i = 31; i >= 0; i--) {
    if (m_pid & (1 << i)) {
      msbit = i;
      break;
    }
  }
  m_pid = (m_pid << (31 - msbit));
  _Setup();
}

RimeWithWeaselHandler::~RimeWithWeaselHandler() {
  m_show_notifications.clear();
  m_session_status_map.clear();
  m_app_options.clear();
}

bool add_session = false;
void _UpdateUIStyle(RimeConfig* config, UI* ui, bool initialize);
bool _UpdateUIStyleColor(RimeConfig* config,
                         UIStyle& style,
                         std::string color = "");
void _LoadAppOptions(RimeConfig* config, AppOptionsByAppName& app_options);

void _RefreshTrayIcon(const RimeSessionId session_id,
                      const std::function<void()> _UpdateUICallback) {
  // Dangerous, don't touch
  static char app_name[256] = {0};
  auto ret = rime_api->get_property(session_id, "client_app", app_name,
                                    sizeof(app_name) - 1);
  if (!ret || u8tow(app_name) == std::wstring(L"explorer.exe"))
    auto th = std::make_unique<ScopedThread>([=]() {
      ::Sleep(100);
      if (_UpdateUICallback)
        _UpdateUICallback();
    });
  else if (_UpdateUICallback)
    _UpdateUICallback();
}

void RimeWithWeaselHandler::_Setup() {
  RIME_STRUCT(RimeTraits, weasel_traits);
  std::string shared_dir = wtou8(WeaselSharedDataPath().wstring());
  std::string user_dir = wtou8(WeaselUserDataPath().wstring());
  weasel_traits.shared_data_dir = shared_dir.c_str();
  weasel_traits.user_data_dir = user_dir.c_str();
  weasel_traits.prebuilt_data_dir = weasel_traits.shared_data_dir;
  std::string distribution_name = wtou8(get_weasel_ime_name());
  weasel_traits.distribution_name = distribution_name.c_str();
  weasel_traits.distribution_code_name = WEASEL_CODE_NAME;
  weasel_traits.distribution_version = WEASEL_VERSION;
  weasel_traits.app_name = "rime.weasel";
  std::string log_dir = WeaselLogPath().u8string();
  weasel_traits.log_dir = log_dir.c_str();
  rime_api->setup(&weasel_traits);
  rime_api->set_notification_handler(&RimeWithWeaselHandler::OnNotify, this);
}

void RimeWithWeaselHandler::Initialize() {
  m_disabled = _IsDeployerRunning();
  if (m_disabled) {
    return;
  }

  LOG(INFO) << "Initializing la rime.";
  rime_api->initialize(NULL);
  HANDLE hMutex =
      CreateMutexW(NULL, FALSE, L"Global\\WeaselStartMaintenanceMutex");
  if (hMutex) {
    if (WaitForSingleObject(hMutex, 0) == WAIT_OBJECT_0) {
      if (rime_api->start_maintenance(/*full_check = */ False)) {
        rime_api->join_maintenance_thread();
        m_disabled = true;
      }
      ReleaseMutex(hMutex);
    }
    CloseHandle(hMutex);
  }
  RimeConfig config = {NULL};
  if (rime_api->config_open("weasel", &config)) {
    if (m_ui) {
      _UpdateUIStyle(&config, m_ui, true);
      _UpdateShowNotifications(&config, true);
      m_current_dark_mode = IsUserDarkMode();
      if (m_current_dark_mode) {
        const int BUF_SIZE = 255;
        char buffer[BUF_SIZE + 1] = {0};
        if (rime_api->config_get_string(&config, "style/color_scheme_dark",
                                        buffer, BUF_SIZE)) {
          std::string color_name(buffer);
          _UpdateUIStyleColor(&config, m_ui->style(), color_name);
        }
      }
      m_base_style = m_ui->style();
    }
    Bool global_ascii = false;
    if (rime_api->config_get_bool(&config, "global_ascii", &global_ascii))
      m_global_ascii_mode = !!global_ascii;
    Bool llm_developer_mode = false;
    if (rime_api->config_get_bool(&config, "llm/developer_mode",
                                  &llm_developer_mode)) {
      m_llm_developer_mode = !!llm_developer_mode;
    } else {
      m_llm_developer_mode = false;
    }
    int llm_context_recent_words = 50;
    if (rime_api->config_get_int(&config, "llm/context_recent_words",
                                 &llm_context_recent_words) &&
        llm_context_recent_words > 0) {
      m_llm_context_recent_words =
          static_cast<size_t>(llm_context_recent_words);
    } else {
      m_llm_context_recent_words = 50;
    }
    int llm_context_max_chars = 0;
    if (rime_api->config_get_int(&config, "llm/context_max_chars",
                                  &llm_context_max_chars) &&
        llm_context_max_chars >= 0) {
      m_llm_context_max_chars = static_cast<size_t>(llm_context_max_chars);
    } else {
      m_llm_context_max_chars = 0;
    }
    int llm_input_prediction_debounce_ms = 120;
    if (rime_api->config_get_int(&config, "llm/input_prediction_debounce_ms",
                                 &llm_input_prediction_debounce_ms) &&
        llm_input_prediction_debounce_ms >= 0) {
      m_llm_input_prediction_debounce_ms =
          static_cast<DWORD>(llm_input_prediction_debounce_ms);
    } else {
      m_llm_input_prediction_debounce_ms = 120;
    }
    if (m_ui && m_llm_developer_mode && m_base_style.comment_font_point <= 0) {
      m_base_style.comment_font_point =
          m_base_style.font_point > 0 ? m_base_style.font_point : 12;
      if (m_base_style.comment_font_face.empty()) {
        m_base_style.comment_font_face = m_base_style.font_face;
      }
      m_ui->style() = m_base_style;
    }
    if (!rime_api->config_get_int(&config, "show_notifications_time",
                                  &m_show_notifications_time))
      m_show_notifications_time = 1200;
    _LoadAppOptions(&config, m_app_options);
    
    // 初始化LLM Provider (注意：此时m_dev_console可能还未初始化)
    Bool llm_enabled = false;
    if (rime_api->config_get_bool(&config, "llm/enabled", &llm_enabled)) {
      if (llm_enabled) {
        // 读取 provider_type 配置项，默认为 "openai"
        const int BUF_SIZE = 64;
        char provider_type_buf[BUF_SIZE + 1] = {0};
        std::string provider_type = "openai";  // 默认值
        if (rime_api->config_get_string(&config, "llm/provider_type", provider_type_buf, BUF_SIZE)) {
          provider_type = provider_type_buf;
        }
        
        // 根据 provider_type 创建相应的 provider
        if (provider_type == "llamacpp") {
          m_llm_provider = std::make_unique<LlamaCppProvider>();
          LOG(INFO) << "LLM Provider type: llamacpp";
        } else if (provider_type == "hf_constraint") {
          m_llm_provider = std::make_unique<HFConstraintProvider>();
          LOG(INFO) << "LLM Provider type: hf_constraint";
        } else {
          // 默认使用 OpenAICompatibleProvider
          m_llm_provider = std::make_unique<OpenAICompatibleProvider>();
          LOG(INFO) << "LLM Provider type: " << provider_type << " (defaulting to openai)";
        }
        
        if (m_llm_provider->LoadConfig("weasel")) {
          LOG(INFO) << "LLM Provider initialized successfully: "
                    << m_llm_provider->GetProviderName();
        } else {
          LOG(ERROR) << "LLM Provider initialization failed: LoadConfig returned false";
          LOG(ERROR) << "Please check your weasel.yaml configuration:";
          if (provider_type == "llamacpp") {
            LOG(ERROR) << "  llm:";
            LOG(ERROR) << "    enabled: true";
            LOG(ERROR) << "    provider_type: llamacpp";
            LOG(ERROR) << "    llamacpp:";
            LOG(ERROR) << "      model_path: \"path/to/model.gguf\"";
          } else if (provider_type == "hf_constraint") {
            LOG(ERROR) << "  llm:";
            LOG(ERROR) << "    enabled: true";
            LOG(ERROR) << "    provider_type: hf_constraint";
            LOG(ERROR) << "    hf_constraint:";
            LOG(ERROR) << "      api_url: \"http://localhost:8000/v1/generate/completions\"";
          } else {
            LOG(ERROR) << "  llm:";
            LOG(ERROR) << "    enabled: true";
            LOG(ERROR) << "    provider_type: openai";
            LOG(ERROR) << "    openai:";
            LOG(ERROR) << "      api_key: \"your-api-key\"";
          }
          m_llm_provider.reset();
        }
      } else {
        LOG(INFO) << "LLM is disabled in configuration (llm/enabled = false)";
      }
    } else {
      LOG(INFO) << "LLM configuration not found (llm/enabled not set)";
    }
    
    rime_api->config_close(&config);
  }
  m_last_schema_id.clear();
}

void RimeWithWeaselHandler::Finalize() {
  m_active_session = 0;
  m_disabled = true;
  m_session_status_map.clear();
  LOG(INFO) << "Finalizing la rime.";
  rime_api->finalize();
}

DWORD RimeWithWeaselHandler::FindSession(WeaselSessionId ipc_id) {
  if (m_disabled)
    return 0;
  Bool found = rime_api->find_session(to_session_id(ipc_id));
  DLOG(INFO) << "Find session: session_id = " << to_session_id(ipc_id)
             << ", found = " << found;
  return found ? (ipc_id) : 0;
}

DWORD RimeWithWeaselHandler::AddSession(LPWSTR buffer, EatLine eat) {
  if (m_disabled) {
    DLOG(INFO) << "Trying to resume service.";
    EndMaintenance();
    if (m_disabled)
      return 0;
  }
  RimeSessionId session_id = (RimeSessionId)rime_api->create_session();
  if (m_global_ascii_mode) {
    for (const auto& pair : m_session_status_map) {
      if (pair.first) {
        rime_api->set_option(session_id, "ascii_mode",
                             !!pair.second.status.is_ascii_mode);
        break;
      }
    }
  }

  WeaselSessionId ipc_id =
      _GenerateNewWeaselSessionId(m_session_status_map, m_pid);
  DLOG(INFO) << "Add session: created session_id = " << session_id
             << ", ipc_id = " << ipc_id;
  SessionStatus& session_status = new_session_status(ipc_id);
  session_status.style = m_base_style;
  session_status.session_id = session_id;
  _ReadClientInfo(ipc_id, buffer);

  RIME_STRUCT(RimeStatus, status);
  if (rime_api->get_status(session_id, &status)) {
    std::string schema_id = status.schema_id;
    m_last_schema_id = schema_id;
    _LoadSchemaSpecificSettings(ipc_id, schema_id);
    _LoadAppInlinePreeditSet(ipc_id, true);
    _UpdateInlinePreeditStatus(ipc_id);
    _RefreshTrayIcon(session_id, _UpdateUICallback);
    session_status.status = status;
    session_status.__synced = false;
    rime_api->free_status(&status);
  }
  m_ui->style() = session_status.style;
  // show session's welcome message :-) if any
  if (eat) {
    _Respond(ipc_id, eat);
  }
  add_session = true;
  _UpdateUI(ipc_id);
  add_session = false;
  m_active_session = ipc_id;
  return ipc_id;
}

DWORD RimeWithWeaselHandler::RemoveSession(WeaselSessionId ipc_id) {
  if (m_ui)
    m_ui->Hide();
  if (m_disabled)
    return 0;
  DLOG(INFO) << "Remove session: session_id = " << to_session_id(ipc_id);
  // TODO: force committing? otherwise current composition would be lost
  rime_api->destroy_session(to_session_id(ipc_id));
  m_session_status_map.erase(ipc_id);
  m_active_session = 0;
  return 0;
}

void RimeWithWeaselHandler::UpdateColorTheme(BOOL darkMode) {
  RimeConfig config = {NULL};
  if (rime_api->config_open("weasel", &config)) {
    if (m_ui) {
      _UpdateUIStyle(&config, m_ui, true);
      m_current_dark_mode = darkMode;
      if (darkMode) {
        const int BUF_SIZE = 255;
        char buffer[BUF_SIZE + 1] = {0};
        if (rime_api->config_get_string(&config, "style/color_scheme_dark",
                                        buffer, BUF_SIZE)) {
          std::string color_name(buffer);
          _UpdateUIStyleColor(&config, m_ui->style(), color_name);
        }
      }
      m_base_style = m_ui->style();
    }
    rime_api->config_close(&config);
  }

  for (auto& pair : m_session_status_map) {
    RIME_STRUCT(RimeStatus, status);
    if (rime_api->get_status(to_session_id(pair.first), &status)) {
      _LoadSchemaSpecificSettings(pair.first, std::string(status.schema_id));
      _LoadAppInlinePreeditSet(pair.first, true);
      _UpdateInlinePreeditStatus(pair.first);
      pair.second.status = status;
      pair.second.__synced = false;
      rime_api->free_status(&status);
    }
  }
  m_ui->style() = get_session_status(m_active_session).style;
}

BOOL RimeWithWeaselHandler::ProcessKeyEvent(KeyEvent keyEvent,
                                            WeaselSessionId ipc_id,
                                            EatLine eat) {
  DLOG(INFO) << "Process key event: keycode = " << keyEvent.keycode
             << ", mask = " << keyEvent.mask << ", ipc_id = " << ipc_id;
  if (m_disabled)
    return FALSE;
  
  RimeSessionId session_id = to_session_id(ipc_id);
  const DWORD event_time = GetTickCount();
  const bool is_destructive_edit_key =
      !(keyEvent.mask & ibus::Modifier::RELEASE_MASK) &&
      (keyEvent.keycode == ibus::Keycode::BackSpace ||
       keyEvent.keycode == ibus::Keycode::Delete);
  const bool is_composition_edit_key =
      !(keyEvent.mask & ibus::Modifier::RELEASE_MASK) &&
      (((keyEvent.keycode >= 'a' && keyEvent.keycode <= 'z') ||
        (keyEvent.keycode >= 'A' && keyEvent.keycode <= 'Z') ||
        keyEvent.keycode == '\'') ||
       is_destructive_edit_key);

  if (is_composition_edit_key) {
    if (m_last_edit_key_time > 0 &&
        (event_time - m_last_edit_key_time) <= LLM_EDIT_BURST_WINDOW_MS) {
      ++m_consecutive_edit_key_count;
    } else {
      m_consecutive_edit_key_count = 1;
    }
    m_last_edit_key_time = event_time;

    if (is_destructive_edit_key ||
        m_consecutive_edit_key_count >= LLM_EDIT_BURST_THRESHOLD) {
      const DWORD suppress_until = event_time + LLM_RERANK_SUPPRESS_MS;
      if (suppress_until > m_llm_rerank_suppressed_until) {
        m_llm_rerank_suppressed_until = suppress_until;
        if (m_dev_console && m_dev_console->IsEnabled()) {
          std::wstringstream ss;
          ss << L"[LLM] 检测到"
             << (is_destructive_edit_key ? L"退格/删除" : L"连续编辑")
             << L"，在接下来 " << LLM_RERANK_SUPPRESS_MS
             << L" ms 内禁止 rerank";
          m_dev_console->WriteLine(ss.str());
        }
      }
    }
  }
  
  // 处理·键（反引号键）：触发LLM预测（仅在composing状态下）或清空上下文（双击）
  if (!(keyEvent.mask & ibus::Modifier::RELEASE_MASK) &&
      (keyEvent.keycode == ibus::Keycode::grave || keyEvent.keycode == 0x060)) {
    bool is_double_click = false;
    
    // 检测双击（500ms内连续按下两次）
    if (m_last_grave_key_time > 0 && 
        (event_time - m_last_grave_key_time) < GRAVE_DOUBLE_CLICK_TIMEOUT) {
      is_double_click = true;
    }
    m_last_grave_key_time = event_time;
    
    if (m_dev_console && m_dev_console->IsEnabled()) {
      if (is_double_click) {
        m_dev_console->WriteLine(L"[LLM] 检测到双击·键");
      } else {
        m_dev_console->WriteLine(L"[LLM] 用户按下·键");
      }
    }
    
    // 双击·键：清空上下文历史记录
    if (is_double_click) {
      if (m_context_history) {
        if (m_dev_console && m_dev_console->IsEnabled()) {
          size_t size_before = m_context_history->GetSize();
          m_dev_console->WriteLine(L"[LLM] 双击·键，清空上下文历史记录（清空前记录数: " + std::to_wstring(size_before) + L"）");
        }
        m_context_history->Clear(m_dev_console);
        if (m_dev_console && m_dev_console->IsEnabled()) {
          m_dev_console->WriteLine(L"[LLM] 上下文历史记录已清空");
        }
      } else {
        if (m_dev_console && m_dev_console->IsEnabled()) {
          m_dev_console->WriteLine(L"[LLM] 上下文历史记录未初始化，无法清空");
        }
      }
      // 清空上下文后，阻止按键继续传递
      return TRUE;
    }
    
    // 检查是否处于composing状态
    RIME_STRUCT(RimeStatus, status);
    bool is_composing = false;
    if (rime_api->get_status(session_id, &status)) {
      is_composing = !!status.is_composing;
      rime_api->free_status(&status);
    }
    
    // 只有在composing状态下才触发LLM预测，否则让·键正常输入
    if (!is_composing) {
      if (m_dev_console && m_dev_console->IsEnabled()) {
        m_dev_console->WriteLine(L"[LLM] 不在composing状态，允许·键正常输入");
      }
      // 不阻止按键，让Rime正常处理（允许输入·符号）
      // 继续执行后续代码，让Rime正常处理该按键
    } else {
      // 在composing状态下，检查LLM是否可用
      if (!m_llm_provider) {
        if (m_dev_console && m_dev_console->IsEnabled()) {
          m_dev_console->WriteLine(L"[LLM] LLM提供者未初始化");
          m_dev_console->WriteLine(L"[LLM] 请检查weasel.yaml配置文件中是否启用了LLM功能：");
          m_dev_console->WriteLine(L"[LLM]   llm:");
          m_dev_console->WriteLine(L"[LLM]     enabled: true");
          m_dev_console->WriteLine(L"[LLM]     openai:");
          m_dev_console->WriteLine(L"[LLM]       api_key: \"your-api-key\"");
        }
        // 不阻止按键，让Rime正常处理
        // 继续执行后续代码
      } else if (!m_llm_provider->IsAvailable()) {
        if (m_dev_console && m_dev_console->IsEnabled()) {
          m_dev_console->WriteLine(L"[LLM] LLM提供者已初始化，但不可用");
          m_dev_console->WriteLine(L"[LLM] 可能的原因：");
          m_dev_console->WriteLine(L"[LLM]   1. llm/enabled 未设置为 true");
          m_dev_console->WriteLine(L"[LLM]   2. llm/openai/api_key 未配置或为空");
          m_dev_console->WriteLine(L"[LLM]   3. llm/openai/api_url 未配置或为空");
        }
        // 不阻止按键，让Rime正常处理
        // 继续执行后续代码
      } else {
        // LLM可用，在composing状态下触发预测
        if (m_dev_console && m_dev_console->IsEnabled()) {
          m_dev_console->WriteLine(L"[LLM] composing状态=true，触发LLM预测");
        }
        
        // 获取当前键入的拼音（preedit）
        std::wstring current_preedit;
        size_t rime_candidate_count = 0;
        RIME_STRUCT(RimeContext, ctx);
        if (rime_api->get_context(session_id, &ctx)) {
          rime_candidate_count = static_cast<size_t>(ctx.menu.num_candidates);
          if (ctx.composition.length > 0 && ctx.composition.preedit) {
            current_preedit = u8tow(ctx.composition.preedit);
            if (m_dev_console && m_dev_console->IsEnabled()) {
              m_dev_console->WriteLine(L"[LLM] 获取到当前拼音: " + current_preedit);
            }
          }
          rime_api->free_context(&ctx);
        }

        // 如果不在LLM预测模式，进入LLM预测模式（上下文统一从 m_context_history 获取）
        if (!m_llm_prediction_mode) {
          m_llm_prediction_mode = true;
          if (m_dev_console && m_dev_console->IsEnabled()) {
            if (m_context_history && m_context_history->GetSize() > 0) {
              m_dev_console->WriteLine(L"[LLM] 进入LLM预测模式，将使用上下文历史");
            } else {
              m_dev_console->WriteLine(L"[LLM] 进入LLM预测模式，上下文历史为空");
            }
          }
        }
        
        // 触发LLM预测，传入当前拼音
        const bool require_rime_candidates =
            !current_preedit.empty() && rime_candidate_count > 0;
        const bool enable_rime_reorder =
            require_rime_candidates &&
            event_time >= m_llm_rerank_suppressed_until;
        const LLMRequestType request_type =
            enable_rime_reorder ? LLMRequestType::RimeReorder
                                : LLMRequestType::PinyinConstrainedPrediction;
        if (m_dev_console && m_dev_console->IsEnabled()) {
          m_dev_console->WriteLine(
              require_rime_candidates
                  ? (enable_rime_reorder
                         ? L"[LLM] 当前有 Rime 候选词，将执行异步重排"
                         : L"[LLM] 当前有 Rime 候选词，但处于保护窗口内，将执行纯 LLM 候选生成")
                  : L"[LLM] 当前没有 Rime 候选词，将执行纯 LLM 候选生成");
        }
        if (require_rime_candidates && !enable_rime_reorder &&
            m_dev_console && m_dev_console->IsEnabled()) {
          std::wstringstream ss;
          ss << L"[LLM] 当前处于连续编辑保护窗口内，关闭 rerank，仅执行异步 LLM 生成（剩余 "
             << (m_llm_rerank_suppressed_until - event_time) << L" ms）";
          m_dev_console->WriteLine(ss.str());
        }
        _TriggerLLMPrediction(
            ipc_id, request_type, current_preedit, require_rime_candidates, 0);
        
        // 更新UI
        // _UpdateUI(ipc_id);
        
        // 阻止按键继续传递
        return TRUE;
      }
    }
  }
  
  if (is_destructive_edit_key && m_llm_prediction_mode) {
    if (m_dev_console && m_dev_console->IsEnabled()) {
      m_dev_console->WriteLine(
          L"[LLM] 检测到退格/删除操作，退出LLM预测模式以降低编辑时开销");
    }
    _ExitLLMPredictionMode(ipc_id);
  }

  // 如果处于LLM预测模式，处理特殊按键
  if (m_llm_prediction_mode && !(keyEvent.mask & ibus::Modifier::RELEASE_MASK)) {
    // ESC键：退出LLM预测模式
    if (keyEvent.keycode == ibus::Keycode::Escape) {
      _ExitLLMPredictionMode(ipc_id);
      return TRUE;
    }

    auto llm_snapshot = _SnapshotLLMCandidates();
    RIME_STRUCT(RimeContext, ctx);
    const bool has_context = rime_api->get_context(session_id, &ctx);
    const auto display_candidates =
        _BuildDisplayCandidates(has_context ? &ctx : nullptr, llm_snapshot);

    size_t display_index = 0;
    bool should_select_display_candidate = false;

    if (keyEvent.keycode == ibus::Keycode::space) {
      should_select_display_candidate = !display_candidates.empty();
      display_index = 0;
    } else if (has_context) {
      should_select_display_candidate = _TryResolveDisplaySelectionIndex(
          keyEvent, ctx, display_candidates.size(), display_index);
    } else if (keyEvent.keycode >= '1' && keyEvent.keycode <= '9') {
      display_index = static_cast<size_t>(keyEvent.keycode - '1');
      should_select_display_candidate =
          display_index < display_candidates.size();
    } else if (keyEvent.keycode == '0') {
      display_index = 9;
      should_select_display_candidate =
          display_index < display_candidates.size();
    }

    if (should_select_display_candidate &&
        display_index < display_candidates.size()) {
      if (has_context) {
        rime_api->free_context(&ctx);
      }
      if (_SelectDisplayCandidate(
              display_candidates[display_index],
              llm_snapshot.candidates,
              ipc_id,
              eat)) {
        return TRUE;
      }
    }

    if (has_context) {
      rime_api->free_context(&ctx);
    }
    
    // 如果输入的是拼音（字母），退出LLM预测模式，回到正常输入
    if ((keyEvent.keycode >= 'a' && 
         keyEvent.keycode <= 'z') ||
        (keyEvent.keycode >= 'A' && 
         keyEvent.keycode <= 'Z')) {
      _ExitLLMPredictionMode(ipc_id);
      // 继续处理按键，进入正常输入流程
    }
  }
  
  m_has_display_highlight_override = false;
  Bool handled = rime_api->process_key(session_id, keyEvent.keycode,
                                       expand_ibus_modifier(keyEvent.mask));
  // vim_mode when keydown only
  if (!handled && !(keyEvent.mask & ibus::Modifier::RELEASE_MASK)) {
    bool isVimBackInCommandMode =
        (keyEvent.keycode == ibus::Keycode::Escape) ||
        ((keyEvent.mask & (1 << 2)) &&
         (keyEvent.keycode == ibus::Keycode::XK_c ||
          keyEvent.keycode == ibus::Keycode::XK_C ||
          keyEvent.keycode == ibus::Keycode::XK_bracketleft));
    if (isVimBackInCommandMode &&
        rime_api->get_option(session_id, "vim_mode") &&
        !rime_api->get_option(session_id, "ascii_mode")) {
      rime_api->set_option(session_id, "ascii_mode", True);
    }
  }

  const bool should_auto_predict_input =
      !(keyEvent.mask & ibus::Modifier::RELEASE_MASK) &&
      handled &&
      m_llm_provider && m_llm_provider->IsAvailable() &&
      keyEvent.keycode != ibus::Keycode::grave &&
      keyEvent.keycode != 0x060 &&
      (((keyEvent.keycode >= 'a' && keyEvent.keycode <= 'z') ||
        (keyEvent.keycode >= 'A' && keyEvent.keycode <= 'Z') ||
        keyEvent.keycode == '\''));
  if (should_auto_predict_input) {
    std::wstring current_preedit;
    size_t rime_candidate_count = 0;
    RIME_STRUCT(RimeContext, auto_ctx);
    if (rime_api->get_context(session_id, &auto_ctx)) {
      rime_candidate_count = static_cast<size_t>(auto_ctx.menu.num_candidates);
      if (auto_ctx.composition.length > 0 && auto_ctx.composition.preedit) {
        current_preedit = u8tow(auto_ctx.composition.preedit);
      }
      rime_api->free_context(&auto_ctx);
    }

    if (!current_preedit.empty()) {
      if (!m_llm_prediction_mode) {
        m_llm_prediction_mode = true;
      }
      const bool require_rime_candidates = rime_candidate_count > 0;
      const bool enable_rime_reorder =
          require_rime_candidates &&
          event_time >= m_llm_rerank_suppressed_until;
      const LLMRequestType request_type =
          enable_rime_reorder ? LLMRequestType::RimeReorder
                              : LLMRequestType::PinyinConstrainedPrediction;
      if (m_dev_console && m_dev_console->IsEnabled()) {
        std::wstringstream ss;
        ss << L"[LLM] 普通输入自动触发 LLM，preedit=" << current_preedit
           << L"，Rime候选=" << rime_candidate_count
           << L"，请求类型=" << GetLLMRequestTypeName(request_type);
        m_dev_console->WriteLine(ss.str());
      }
      _TriggerLLMPrediction(ipc_id,
                            request_type,
                            current_preedit,
                            require_rime_candidates,
                            m_llm_input_prediction_debounce_ms);
    }
  }

  _Respond(ipc_id, eat);
  _UpdateUI(ipc_id);
  m_active_session = ipc_id;
  return (BOOL)handled;
}

void RimeWithWeaselHandler::CommitComposition(WeaselSessionId ipc_id) {
  DLOG(INFO) << "Commit composition: ipc_id = " << ipc_id;
  if (m_disabled)
    return;
  rime_api->commit_composition(to_session_id(ipc_id));
  _UpdateUI(ipc_id);
  m_active_session = ipc_id;
}

void RimeWithWeaselHandler::ClearComposition(WeaselSessionId ipc_id) {
  DLOG(INFO) << "Clear composition: ipc_id = " << ipc_id;
  if (m_disabled)
    return;
  rime_api->clear_composition(to_session_id(ipc_id));
  _UpdateUI(ipc_id);
  m_active_session = ipc_id;
}

void RimeWithWeaselHandler::SelectCandidateOnCurrentPage(
    size_t index,
    WeaselSessionId ipc_id) {
  DLOG(INFO) << "select candidate on current page, ipc_id = " << ipc_id
             << ", index = " << index;
  LOG(INFO) << "[DEBUG] SelectCandidateOnCurrentPage called: index=" << index
            << ", ipc_id=" << ipc_id << ", llm_mode=" << m_llm_prediction_mode;
  
  if (m_disabled)
    return;
  
  RimeSessionId session_id = to_session_id(ipc_id);

  if (m_llm_prediction_mode) {
    auto llm_snapshot = _SnapshotLLMCandidates();
    RIME_STRUCT(RimeContext, ctx);
    if (rime_api->get_context(session_id, &ctx)) {
      const auto display_candidates =
          _BuildDisplayCandidates(&ctx, llm_snapshot);
      rime_api->free_context(&ctx);

      if (index < display_candidates.size() &&
          _SelectDisplayCandidate(
              display_candidates[index],
              llm_snapshot.candidates,
              ipc_id,
              EatLine())) {
        return;
      }
    } else if (!llm_snapshot.candidates.empty()) {
      const auto display_candidates =
          _BuildDisplayCandidates(nullptr, llm_snapshot);
      if (index < display_candidates.size() &&
          _SelectDisplayCandidate(
              display_candidates[index],
              llm_snapshot.candidates,
              ipc_id,
              EatLine())) {
        return;
      }
    }
  }
  
  // 如果不是LLM候选词或不在LLM模式，按照正常流程处理Rime候选词
  LOG(INFO) << "[DEBUG] Processing as Rime candidate";
  rime_api->select_candidate_on_current_page(session_id, index);
}

bool RimeWithWeaselHandler::HighlightCandidateOnCurrentPage(
    size_t index,
    WeaselSessionId ipc_id,
    EatLine eat) {
  DLOG(INFO) << "highlight candidate on current page, ipc_id = " << ipc_id
             << ", index = " << index;
  bool res = false;

  if (m_llm_prediction_mode) {
    auto llm_snapshot = _SnapshotLLMCandidates();
    RIME_STRUCT(RimeContext, ctx);
    if (rime_api->get_context(to_session_id(ipc_id), &ctx)) {
      const auto display_candidates =
          _BuildDisplayCandidates(&ctx, llm_snapshot);
      rime_api->free_context(&ctx);

      if (index < display_candidates.size()) {
        const auto& candidate = display_candidates[index];
        if (candidate.source == DisplayCandidate::Source::Rime) {
          m_has_display_highlight_override = false;
          res = rime_api->highlight_candidate_on_current_page(
              to_session_id(ipc_id), candidate.index);
        } else {
          m_has_display_highlight_override = true;
          m_display_highlight_override = index;
          res = true;
        }
      }
    } else if (!llm_snapshot.candidates.empty()) {
      const auto display_candidates =
          _BuildDisplayCandidates(nullptr, llm_snapshot);
      if (index < display_candidates.size()) {
        m_has_display_highlight_override = true;
        m_display_highlight_override = index;
        res = true;
      }
    }
  }

  if (!m_llm_prediction_mode || !res) {
    m_has_display_highlight_override = false;
    res = rime_api->highlight_candidate_on_current_page(
        to_session_id(ipc_id), index);
  }

  _Respond(ipc_id, eat);
  _UpdateUI(ipc_id);
  return res;
}

bool RimeWithWeaselHandler::ChangePage(bool backward,
                                       WeaselSessionId ipc_id,
                                       EatLine eat) {
  DLOG(INFO) << "change page, ipc_id = " << ipc_id
             << (backward ? "backward" : "foreward");
  m_has_display_highlight_override = false;
  bool res = rime_api->change_page(to_session_id(ipc_id), backward);
  _Respond(ipc_id, eat);
  _UpdateUI(ipc_id);
  return res;
}

void RimeWithWeaselHandler::FocusIn(DWORD client_caps, WeaselSessionId ipc_id) {
  DLOG(INFO) << "Focus in: ipc_id = " << ipc_id
             << ", client_caps = " << client_caps;
  if (m_disabled)
    return;
  _UpdateUI(ipc_id);
  m_active_session = ipc_id;
}

void RimeWithWeaselHandler::FocusOut(DWORD param, WeaselSessionId ipc_id) {
  DLOG(INFO) << "Focus out: ipc_id = " << ipc_id;
  
  // 退出LLM预测模式（如果处于该模式）
  if (m_llm_prediction_mode) {
    _ExitLLMPredictionMode(ipc_id);
  }
  
  if (m_ui)
    m_ui->Hide();
  m_active_session = 0;
}

void RimeWithWeaselHandler::UpdateInputPosition(RECT const& rc,
                                                WeaselSessionId ipc_id) {
  DLOG(INFO) << "Update input position: (" << rc.left << ", " << rc.top
             << "), ipc_id = " << ipc_id
             << ", m_active_session = " << m_active_session;
  if (m_ui)
    m_ui->UpdateInputPosition(rc);
  if (m_disabled)
    return;
  if (m_active_session != ipc_id) {
    _UpdateUI(ipc_id);
    m_active_session = ipc_id;
  }
}

std::string RimeWithWeaselHandler::m_message_type;
std::string RimeWithWeaselHandler::m_message_value;
std::string RimeWithWeaselHandler::m_message_label;
std::string RimeWithWeaselHandler::m_option_name;

void RimeWithWeaselHandler::OnNotify(void* context_object,
                                     uintptr_t session_id,
                                     const char* message_type,
                                     const char* message_value) {
  // may be running in a thread when deploying rime
  RimeWithWeaselHandler* self =
      reinterpret_cast<RimeWithWeaselHandler*>(context_object);
  if (!self || !message_type || !message_value)
    return;
  m_message_type = message_type;
  m_message_value = message_value;
  if (RIME_API_AVAILABLE(rime_api, get_state_label) &&
      !strcmp(message_type, "option")) {
    Bool state = message_value[0] != '!';
    const char* option_name = message_value + !state;
    m_option_name = option_name;
    const char* state_label =
        rime_api->get_state_label(session_id, option_name, state);
    if (state_label) {
      m_message_label = std::string(state_label);
    }
  }
}

void RimeWithWeaselHandler::_ReadClientInfo(WeaselSessionId ipc_id,
                                            LPWSTR buffer) {
  std::string app_name;
  std::string client_type;
  // parse request text
  WMemStream bs((wchar_t*)buffer, WEASEL_IPC_BUFFER_LENGTH);
  std::wstring line;
  while (bs.good()) {
    std::getline(bs, line);
    if (!bs.good())
      break;
    // file ends
    if (line == L".")
      break;
    const std::wstring kClientAppKey = L"session.client_app=";
    if (starts_with(line, kClientAppKey)) {
      std::wstring lwr = line;
      to_lower(lwr);
      app_name = wtou8(lwr.substr(kClientAppKey.length()));
    }
    const std::wstring kClientTypeKey = L"session.client_type=";
    if (starts_with(line, kClientTypeKey)) {
      client_type = wtou8(line.substr(kClientTypeKey.length()));
    }
  }
  SessionStatus& session_status = get_session_status(ipc_id);
  RimeSessionId session_id = session_status.session_id;
  // set app specific options
  if (!app_name.empty()) {
    rime_api->set_property(session_id, "client_app", app_name.c_str());

    auto it = m_app_options.find(app_name);
    if (it != m_app_options.end()) {
      AppOptions& options(m_app_options[it->first]);
      for (const auto& pair : options) {
        DLOG(INFO) << "set app option: " << pair.first << " = " << pair.second;
        rime_api->set_option(session_id, pair.first.c_str(), Bool(pair.second));
      }
    }
  }
  // ime | tsf
  rime_api->set_property(session_id, "client_type", client_type.c_str());
  // inline preedit
  bool inline_preedit =
      session_status.style.inline_preedit && (client_type == "tsf");
  rime_api->set_option(session_id, "inline_preedit", Bool(inline_preedit));
  // show soft cursor on weasel panel but not inline
  rime_api->set_option(session_id, "soft_cursor", Bool(!inline_preedit));
}

RimeWithWeaselHandler::LLMCandidateSnapshot
RimeWithWeaselHandler::_SnapshotLLMCandidates() {
  LLMCandidateSnapshot snapshot;
  std::lock_guard<std::mutex> lock(m_llm_mutex);
  snapshot.candidates = m_current_llm_candidates;
  snapshot.require_rime_candidates = m_current_llm_candidates_require_rime;
  snapshot.enable_rime_reorder = m_current_llm_candidates_enable_rime_reorder;
  return snapshot;
}

std::vector<RimeWithWeaselHandler::DisplayCandidate>
RimeWithWeaselHandler::_BuildDisplayCandidates(
    const RimeContext* ctx,
    const LLMCandidateSnapshot& llm_snapshot) {
  std::vector<DisplayCandidate> display_candidates;
  const size_t rime_candidate_count =
      (ctx != nullptr) ? static_cast<size_t>(ctx->menu.num_candidates) : 0;
  const auto& llm_candidates = llm_snapshot.candidates;

  if (llm_snapshot.require_rime_candidates && rime_candidate_count == 0) {
    return display_candidates;
  }

  if (rime_candidate_count == 0 && llm_candidates.empty()) {
    return display_candidates;
  }

  std::vector<bool> used_rime_candidates(rime_candidate_count, false);
  std::vector<std::wstring> emitted_extra_llm_candidates;

  if (m_llm_prediction_mode && llm_snapshot.enable_rime_reorder &&
      !llm_candidates.empty()) {
    for (size_t llm_index = 0; llm_index < llm_candidates.size(); ++llm_index) {
      const std::wstring& llm_candidate = llm_candidates[llm_index];
      if (llm_candidate.empty()) {
        continue;
      }

      bool matched_rime_candidate = false;
      for (size_t rime_index = 0; rime_index < rime_candidate_count;
           ++rime_index) {
        if (used_rime_candidates[rime_index] || ctx == nullptr) {
          continue;
        }

        if (llm_candidate == u8tow(ctx->menu.candidates[rime_index].text)) {
          display_candidates.push_back(
              {DisplayCandidate::Source::Rime, rime_index, true});
          used_rime_candidates[rime_index] = true;
          matched_rime_candidate = true;
          break;
        }
      }

      if (matched_rime_candidate) {
        continue;
      }

      if (std::find(emitted_extra_llm_candidates.begin(),
                    emitted_extra_llm_candidates.end(),
                    llm_candidate) != emitted_extra_llm_candidates.end()) {
        continue;
      }

      display_candidates.push_back(
          {DisplayCandidate::Source::LLM, llm_index, true});
      emitted_extra_llm_candidates.push_back(llm_candidate);
    }
  }

  for (size_t rime_index = 0; rime_index < rime_candidate_count; ++rime_index) {
    if (!used_rime_candidates[rime_index]) {
      display_candidates.push_back(
          {DisplayCandidate::Source::Rime, rime_index, false});
    }
  }

  if (m_llm_prediction_mode && !llm_candidates.empty()) {
    for (size_t llm_index = 0; llm_index < llm_candidates.size(); ++llm_index) {
      const std::wstring& llm_candidate = llm_candidates[llm_index];
      if (llm_candidate.empty()) {
        continue;
      }
      if (std::find(emitted_extra_llm_candidates.begin(),
                    emitted_extra_llm_candidates.end(),
                    llm_candidate) != emitted_extra_llm_candidates.end()) {
        continue;
      }
      display_candidates.push_back(
          {DisplayCandidate::Source::LLM, llm_index, false});
      emitted_extra_llm_candidates.push_back(llm_candidate);
    }
  }

  return display_candidates;
}

std::wstring RimeWithWeaselHandler::_GetDisplayLabel(const RimeContext& ctx,
                                                     size_t display_index) {
  if (display_index < static_cast<size_t>(ctx.menu.num_candidates)) {
    if (RIME_STRUCT_HAS_MEMBER(ctx, ctx.select_labels) && ctx.select_labels &&
        ctx.select_labels[display_index]) {
      return escape_string(u8tow(ctx.select_labels[display_index]));
    }
    if (ctx.menu.select_keys && ctx.menu.select_keys[display_index]) {
      return escape_string(std::wstring(1, ctx.menu.select_keys[display_index]));
    }
  }
  return std::to_wstring((display_index + 1) % 10);
}

bool RimeWithWeaselHandler::_TryResolveDisplaySelectionIndex(
    const KeyEvent& key_event,
    const RimeContext& ctx,
    size_t display_candidate_count,
    size_t& display_index) {
  if (display_candidate_count == 0) {
    return false;
  }

  if (ctx.menu.select_keys) {
    for (size_t i = 0;
         i < display_candidate_count &&
         i < static_cast<size_t>(ctx.menu.num_candidates) &&
         ctx.menu.select_keys[i];
         ++i) {
      const wchar_t select_key =
          static_cast<unsigned char>(ctx.menu.select_keys[i]);
      const wchar_t keycode = static_cast<wchar_t>(key_event.keycode);
      if (keycode == select_key ||
          std::towlower(keycode) == std::towlower(select_key)) {
        display_index = i;
        return true;
      }
    }
  }

  if (key_event.keycode >= '1' && key_event.keycode <= '9') {
    display_index = static_cast<size_t>(key_event.keycode - '1');
    return display_index < display_candidate_count;
  }

  if (key_event.keycode == '0') {
    display_index = 9;
    return display_index < display_candidate_count;
  }

  return false;
}

bool RimeWithWeaselHandler::_SelectDisplayCandidate(
    const DisplayCandidate& candidate,
    const std::vector<std::wstring>& llm_candidates,
    WeaselSessionId ipc_id,
    EatLine eat) {
  if (m_disabled) {
    return false;
  }

  RimeSessionId session_id = to_session_id(ipc_id);
  m_has_display_highlight_override = false;

  if (candidate.source == DisplayCandidate::Source::Rime) {
    if (m_dev_console && m_dev_console->IsEnabled()) {
      std::wstringstream ss;
      ss << L"[LLM] 按排序后的显示顺序选择 Rime 候选词，原始索引="
         << candidate.index;
      m_dev_console->WriteLine(ss.str());
    }

    rime_api->select_candidate_on_current_page(session_id, candidate.index);
    if (eat) {
      _Respond(ipc_id, eat);
      _UpdateUI(ipc_id);
    }
    return true;
  }

  if (candidate.index >= llm_candidates.size()) {
    return false;
  }

  const std::wstring& selected = llm_candidates[candidate.index];
  if (m_dev_console && m_dev_console->IsEnabled()) {
    std::wstringstream ss;
    ss << L"[LLM] 按排序后的显示顺序选择 LLM 候选词: " << selected;
    m_dev_console->WriteLine(ss.str());
  }

  rime_api->clear_composition(session_id);
  m_pending_llm_commit = selected;

  if (eat) {
    _Respond(ipc_id, eat);
    _UpdateUI(ipc_id);
  }
  return true;
}

void RimeWithWeaselHandler::_GetCandidateInfo(CandidateInfo& cinfo,
                                              RimeContext& ctx) {
  const bool llm_mode = m_llm_prediction_mode;
  const auto llm_snapshot = _SnapshotLLMCandidates();
  const auto display_candidates =
      _BuildDisplayCandidates(&ctx, llm_snapshot);
  const auto& llm_candidates = llm_snapshot.candidates;

  cinfo.candies.clear();
  cinfo.comments.clear();
  cinfo.labels.clear();
  cinfo.currentPage = ctx.menu.page_no;
  cinfo.is_last_page = ctx.menu.is_last_page;
  cinfo.highlighted = 0;

  for (size_t display_index = 0; display_index < display_candidates.size();
       ++display_index) {
    const auto& candidate = display_candidates[display_index];
    Text candidate_text;
    Text comment_text;
    Text label_text;
    std::wstring comment_value;

    label_text.str = _GetDisplayLabel(ctx, display_index);

    if (candidate.source == DisplayCandidate::Source::Rime &&
        candidate.index < static_cast<size_t>(ctx.menu.num_candidates)) {
      candidate_text.str =
          escape_string(u8tow(ctx.menu.candidates[candidate.index].text));
      if (ctx.menu.candidates[candidate.index].comment) {
        comment_value = u8tow(ctx.menu.candidates[candidate.index].comment);
      }
      if (!m_has_display_highlight_override &&
          ctx.menu.highlighted_candidate_index ==
              static_cast<int>(candidate.index)) {
        cinfo.highlighted = static_cast<int>(display_index);
      }
    } else if (candidate.index < llm_candidates.size()) {
      candidate_text.str = llm_candidates[candidate.index];
    } else {
      continue;
    }

    if (m_llm_developer_mode) {
      std::wstring source_comment;
      if (candidate.source == DisplayCandidate::Source::LLM) {
        source_comment = L"来源: LLM";
        if (m_llm_provider) {
          source_comment += L"/" + u8tow(m_llm_provider->GetProviderName());
        }
      } else if (candidate.matched_by_llm) {
        source_comment = L"来源: Rime + LLM重排";
      } else {
        source_comment = L"来源: Rime";
      }

      if (!comment_value.empty()) {
        comment_value += L" · ";
      }
      comment_value += source_comment;
    }

    comment_text.str = escape_string(comment_value);

    cinfo.candies.push_back(std::move(candidate_text));
    cinfo.comments.push_back(std::move(comment_text));
    cinfo.labels.push_back(std::move(label_text));
  }

  if (m_has_display_highlight_override &&
      m_display_highlight_override < cinfo.candies.size()) {
    cinfo.highlighted = static_cast<int>(m_display_highlight_override);
  } else if (!llm_mode && !cinfo.candies.empty()) {
    cinfo.highlighted =
        std::min<int>(ctx.menu.highlighted_candidate_index,
                      static_cast<int>(cinfo.candies.size()) - 1);
  }

  if (m_dev_console && m_dev_console->IsEnabled()) {
    std::wstringstream ss;
    ss << L"[DEBUG] _GetCandidateInfo: Rime候选词数=" << ctx.menu.num_candidates
       << L", LLM候选词数=" << llm_candidates.size()
       << L", 展示候选词数=" << cinfo.candies.size();
    m_dev_console->WriteLine(ss.str());
  }
}

void RimeWithWeaselHandler::StartMaintenance() {
  m_session_status_map.clear();
  Finalize();
  _UpdateUI(0);
}

void RimeWithWeaselHandler::EndMaintenance() {
  if (m_disabled) {
    Initialize();
    _UpdateUI(0);
  }
  m_session_status_map.clear();
}

void RimeWithWeaselHandler::SetOption(WeaselSessionId ipc_id,
                                      const std::string& opt,
                                      bool val) {
  // from no-session client, not actual typing session
  if (!ipc_id) {
    if (m_global_ascii_mode && opt == "ascii_mode") {
      for (auto& pair : m_session_status_map)
        rime_api->set_option(to_session_id(pair.first), "ascii_mode", val);
    } else {
      rime_api->set_option(to_session_id(m_active_session), opt.c_str(), val);
    }
  } else {
    rime_api->set_option(to_session_id(ipc_id), opt.c_str(), val);
  }
}

void RimeWithWeaselHandler::OnUpdateUI(std::function<void()> const& cb) {
  _UpdateUICallback = cb;
}

bool RimeWithWeaselHandler::_IsDeployerRunning() {
  HANDLE hMutex = CreateMutex(NULL, TRUE, L"WeaselDeployerMutex");
  bool deployer_detected = hMutex && GetLastError() == ERROR_ALREADY_EXISTS;
  if (hMutex) {
    CloseHandle(hMutex);
  }
  return deployer_detected;
}

void RimeWithWeaselHandler::_UpdateUI(WeaselSessionId ipc_id) {
  // 快速检查：如果UI对象不存在，直接返回
  if (!m_ui) {
    if (m_dev_console && m_dev_console->IsEnabled()) {
      m_dev_console->WriteLine(L"[_UpdateUI] 错误: m_ui 为 nullptr，退出");
    }
    return;
  }

  // 获取会话信息
  RimeSessionId session_id = to_session_id(ipc_id);
  bool is_tsf = _IsSessionTSF(session_id);

  // 准备状态和上下文
  Status& weasel_status = m_ui->status();
  Context weasel_context;
  
  if (ipc_id == 0) {
    weasel_status.disabled = m_disabled;
  }

  // 获取状态信息
  _GetStatus(weasel_status, ipc_id, weasel_context);

  // 判断是否需要获取上下文
  // - 非TSF模式：总是获取
  // - 有LLM候选词时：无论是否TSF，都需要获取，以便在_UI中合并Rime+LLM候选
  const bool has_llm_candidates =
      m_llm_prediction_mode && !_SnapshotLLMCandidates().candidates.empty();

  bool need_context = !is_tsf || has_llm_candidates;
  
  if (need_context) {
    _GetContext(weasel_context, session_id);
  }

  // 更新会话样式设置
  SessionStatus& session_status = get_session_status(ipc_id);
  if (rime_api->get_option(session_id, "inline_preedit")) {
    session_status.style.client_caps |= INLINE_PREEDIT_CAPABLE;
  } else {
    session_status.style.client_caps &= ~INLINE_PREEDIT_CAPABLE;
  }

  // 判断是否应该显示UI
  // 条件1：正在输入且非TSF模式
  // 条件2：LLM预测模式且有候选词
  bool should_show_ui =
      (weasel_status.composing && !is_tsf) || !weasel_context.cinfo.empty();
  
  if (should_show_ui) {
    // 显示UI
    m_ui->Update(weasel_context, weasel_status);
    m_ui->Show();
  } else {
    // 检查是否有消息需要显示
    bool has_message = _ShowMessage(weasel_context, weasel_status);
    
    // 如果没有消息且非TSF模式，隐藏UI
    if (!has_message && !is_tsf) {
    m_ui->Hide();
    m_ui->Update(weasel_context, weasel_status);
    }
  }

  // 刷新托盘图标
  _RefreshTrayIcon(session_id, _UpdateUICallback);

  // 清空消息缓存
  m_message_type.clear();
  m_message_value.clear();
  m_message_label.clear();
  m_option_name.clear();
}


// void RimeWithWeaselHandler::_UpdateUI(WeaselSessionId ipc_id) {
//   // if m_ui nullptr, _UpdateUI meaningless
//   if (!m_ui)
//     return;

//   Status& weasel_status = m_ui->status();
//   Context weasel_context;

//   RimeSessionId session_id = to_session_id(ipc_id);
//   bool is_tsf = _IsSessionTSF(session_id);

//   if (ipc_id == 0)
//     weasel_status.disabled = m_disabled;

//   _GetStatus(weasel_status, ipc_id, weasel_context);

//   if (!is_tsf) {
//     _GetContext(weasel_context, session_id);
//   }

//   SessionStatus& session_status = get_session_status(ipc_id);
//   if (rime_api->get_option(session_id, "inline_preedit"))
//     session_status.style.client_caps |= INLINE_PREEDIT_CAPABLE;
//   else
//     session_status.style.client_caps &= ~INLINE_PREEDIT_CAPABLE;

//   if (weasel_status.composing && !is_tsf) {
//     m_ui->Update(weasel_context, weasel_status);
//     m_ui->Show();
//   } else if (!_ShowMessage(weasel_context, weasel_status) && !is_tsf) {
//     m_ui->Hide();
//     m_ui->Update(weasel_context, weasel_status);
//   }

//   _RefreshTrayIcon(session_id, _UpdateUICallback);

//   m_message_type.clear();
//   m_message_value.clear();
//   m_message_label.clear();
//   m_option_name.clear();
// }

void RimeWithWeaselHandler::_LoadSchemaSpecificSettings(
    WeaselSessionId ipc_id,
    const std::string& schema_id) {
  if (!m_ui)
    return;
  RimeConfig config;
  if (!rime_api->schema_open(schema_id.c_str(), &config))
    return;
  _UpdateShowNotifications(&config);
  m_ui->style() = m_base_style;
  _UpdateUIStyle(&config, m_ui, false);
  SessionStatus& session_status = get_session_status(ipc_id);
  session_status.style = m_ui->style();
  UIStyle& style = session_status.style;
  if (m_llm_developer_mode && style.comment_font_point <= 0) {
    style.comment_font_point = style.font_point > 0 ? style.font_point : 12;
    if (style.comment_font_face.empty()) {
      style.comment_font_face = style.font_face;
    }
  }
  // load schema color style config
  const int BUF_SIZE = 255;
  char buffer[BUF_SIZE + 1] = {0};
  const auto update_color_scheme = [&]() {
    std::string color_name(buffer);
    RimeConfigIterator preset = {0};
    if (rime_api->config_begin_map(
            &preset, &config, ("preset_color_schemes/" + color_name).c_str())) {
      _UpdateUIStyleColor(&config, style, color_name);
      rime_api->config_end(&preset);
    } else {
      RimeConfig weaselconfig;
      if (rime_api->config_open("weasel", &weaselconfig)) {
        _UpdateUIStyleColor(&weaselconfig, style, color_name);
        rime_api->config_close(&weaselconfig);
      }
    }
  };
  const char* key =
      m_current_dark_mode ? "style/color_scheme_dark" : "style/color_scheme";
  if (rime_api->config_get_string(&config, key, buffer, BUF_SIZE))
    update_color_scheme();
  // load schema icon start
  {
    const auto load_icon = [](RimeConfig& config, const char* key1,
                              const char* key2) {
      const auto user_dir = WeaselUserDataPath();
      const auto shared_dir = WeaselSharedDataPath();
      const int BUF_SIZE = 255;
      char buffer[BUF_SIZE + 1] = {0};
      if (rime_api->config_get_string(&config, key1, buffer, BUF_SIZE) ||
          (key2 != NULL &&
           rime_api->config_get_string(&config, key2, buffer, BUF_SIZE))) {
        auto resource = u8tow(buffer);
        if (fs::is_regular_file(user_dir / resource))
          return (user_dir / resource).wstring();
        else if (fs::is_regular_file(shared_dir / resource))
          return (shared_dir / resource).wstring();
      }
      return std::wstring();
    };
    style.current_zhung_icon =
        load_icon(config, "schema/icon", "schema/zhung_icon");
    style.current_ascii_icon = load_icon(config, "schema/ascii_icon", NULL);
    style.current_full_icon = load_icon(config, "schema/full_icon", NULL);
    style.current_half_icon = load_icon(config, "schema/half_icon", NULL);
  }
  // load schema icon end
  rime_api->config_close(&config);
}

void RimeWithWeaselHandler::_LoadAppInlinePreeditSet(WeaselSessionId ipc_id,
                                                     bool ignore_app_name) {
  SessionStatus& session_status = get_session_status(ipc_id);
  RimeSessionId session_id = session_status.session_id;
  static char _app_name[50];
  rime_api->get_property(session_id, "client_app", _app_name,
                         sizeof(_app_name) - 1);
  std::string app_name(_app_name);
  if (!ignore_app_name && m_last_app_name == app_name)
    return;
  m_last_app_name = app_name;
  bool inline_preedit = session_status.style.inline_preedit;
  bool found = false;
  if (!app_name.empty()) {
    auto it = m_app_options.find(app_name);
    if (it != m_app_options.end()) {
      AppOptions& options(m_app_options[it->first]);
      for (const auto& pair : options) {
        if (pair.first == "inline_preedit") {
          rime_api->set_option(session_id, pair.first.c_str(),
                               Bool(pair.second));
          session_status.style.inline_preedit = Bool(pair.second);
          found = true;
          break;
        }
      }
    }
  }
  if (!found) {
    session_status.style.inline_preedit = m_base_style.inline_preedit;
    // load from schema.
    RIME_STRUCT(RimeStatus, status);
    if (rime_api->get_status(session_id, &status)) {
      std::string schema_id = status.schema_id;
      RimeConfig config;
      if (rime_api->schema_open(schema_id.c_str(), &config)) {
        Bool value = False;
        if (rime_api->config_get_bool(&config, "style/inline_preedit",
                                      &value)) {
          session_status.style.inline_preedit = value;
        }
        rime_api->config_close(&config);
      }
      rime_api->free_status(&status);
    }
  }
  if (session_status.style.inline_preedit != inline_preedit)
    _UpdateInlinePreeditStatus(ipc_id);
}

std::wstring RimeWithWeaselHandler::_TrimPredictionContext(
    const std::wstring& context) const {
  if (m_llm_context_max_chars == 0 || context.size() <= m_llm_context_max_chars) {
    return context;
  }

  size_t start = context.size() - m_llm_context_max_chars;
  size_t boundary = context.find(L' ', start);
  if (boundary != std::wstring::npos && boundary + 1 < context.size()) {
    start = boundary + 1;
  }
  return context.substr(start);
}

bool RimeWithWeaselHandler::_ShowMessage(Context& ctx, Status& status) {
  // show as auxiliary string
  std::wstring& tips(ctx.aux.str);
  bool show_icon = false;
  if (m_message_type == "deploy") {
    if (m_message_value == "start")
      if (GetThreadUILanguage() == MAKELANGID(LANG_ENGLISH, SUBLANG_ENGLISH_US))
        tips = L"Deploying RIME";
      else
        tips = L"正在部署 RIME";
    else if (m_message_value == "success")
      if (GetThreadUILanguage() == MAKELANGID(LANG_ENGLISH, SUBLANG_ENGLISH_US))
        tips = L"Deployed";
      else
        tips = L"部署完成";
    else if (m_message_value == "failure") {
      if (GetThreadUILanguage() ==
          MAKELANGID(LANG_CHINESE, SUBLANG_CHINESE_TRADITIONAL))
        tips = L"有錯誤，請查看日誌 %TEMP%\\rime.weasel\\rime.weasel.*.INFO";
      else if (GetThreadUILanguage() ==
               MAKELANGID(LANG_CHINESE, SUBLANG_CHINESE_SIMPLIFIED))
        tips = L"有错误，请查看日志 %TEMP%\\rime.weasel\\rime.weasel.*.INFO";
      else
        tips =
            L"There is an error, please check the logs "
            L"%TEMP%\\rime.weasel\\rime.weasel.*.INFO";
    }
  } else if (m_message_type == "schema") {
    tips = /*L"【" + */ status.schema_name /* + L"】"*/;
  } else if (m_message_type == "option") {
    status.type = SCHEMA;
    if (m_message_value == "!ascii_mode") {
      show_icon = true;
    } else if (m_message_value == "ascii_mode") {
      show_icon = true;
    } else
      tips = u8tow(m_message_label);

    if (m_message_value == "full_shape" || m_message_value == "!full_shape")
      status.type = FULL_SHAPE;
  }
  if (tips.empty() && !show_icon)
    return m_ui->IsCountingDown();
  auto foption = m_show_notifications.find(m_option_name);
  auto falways = m_show_notifications.find("always");
  if ((!add_session && (foption != m_show_notifications.end() ||
                        falways != m_show_notifications.end())) ||
      m_message_type == "deploy") {
    m_ui->Update(ctx, status);
    if (m_show_notifications_time)
      m_ui->ShowWithTimeout(m_show_notifications_time);
    return true;
  } else {
    return m_ui->IsCountingDown();
  }
}
inline std::string _GetLabelText(const std::vector<Text>& labels,
                                 int id,
                                 const wchar_t* format) {
  wchar_t buffer[128];
  swprintf_s<128>(buffer, format, labels.at(id).str.c_str());
  return wtou8(std::wstring(buffer));
}

bool RimeWithWeaselHandler::_Respond(WeaselSessionId ipc_id, EatLine eat) {
  std::set<std::string> actions;
  std::list<std::string> messages;
  bool should_refresh_llm_prediction = false;

  const auto handle_committed_text = [&](const std::wstring& committed_text) {
    if (committed_text.empty()) {
      return;
    }
    if (m_context_history) {
      m_context_history->AddText(committed_text, m_dev_console);
    }
    if (!m_llm_provider) {
      LOG(WARNING) << "[LLM] LLM provider is not available";
      return;
    }
    if (!m_llm_provider->IsAvailable()) {
      LOG(WARNING) << "[LLM] LLM provider is not enabled";
      return;
    }
    if (CommitHasMeaningfulContent(committed_text)) {
      should_refresh_llm_prediction = true;
    }
  };

  SessionStatus& session_status = get_session_status(ipc_id);
  RimeSessionId session_id = session_status.session_id;

  if (!m_pending_llm_commit.empty()) {
    actions.insert("commit");
    const std::wstring pending_commit = m_pending_llm_commit;
    messages.push_back(std::string("commit=") +
                       escape_string<char>(wtou8(pending_commit)) + '\n');
    handle_committed_text(pending_commit);
    m_pending_llm_commit.clear();
  }

  RIME_STRUCT(RimeCommit, commit);
  if (rime_api->get_commit(session_id, &commit)) {
    actions.insert("commit");
    messages.push_back(std::string("commit=") +
                       escape_string<char>(commit.text) + '\n');

    if (commit.text && strlen(commit.text) > 0) {
      std::wstring commit_text_w = u8tow(commit.text);
      if (!commit_text_w.empty()) {
        LOG(INFO) << "[LLM] User committed text: " << commit.text;
        handle_committed_text(commit_text_w);
      }
    }

    rime_api->free_commit(&commit);
  }

  if (should_refresh_llm_prediction) {
    if (m_dev_console && m_dev_console->IsEnabled()) {
      m_dev_console->WriteLine(L"[LLM] 检测到新的有效提交，刷新排序/预测候选");
    }
    m_llm_prediction_mode = true;
    m_has_display_highlight_override = false;
    _TriggerLLMPrediction(ipc_id);
  }

  bool is_composing = false;
  RIME_STRUCT(RimeStatus, status);
  if (rime_api->get_status(session_id, &status)) {
    is_composing = !!status.is_composing;
    actions.insert("status");
    messages.push_back(std::string("status.ascii_mode=") +
                       std::to_string(status.is_ascii_mode) + '\n');
    messages.push_back(std::string("status.composing=") +
                       std::to_string(status.is_composing) + '\n');
    messages.push_back(std::string("status.disabled=") +
                       std::to_string(status.is_disabled) + '\n');
    messages.push_back(std::string("status.full_shape=") +
                       std::to_string(status.is_full_shape) + '\n');
    messages.push_back(std::string("status.schema_id=") +
                       std::string(status.schema_id) + '\n');
    if (m_global_ascii_mode &&
        (session_status.status.is_ascii_mode != status.is_ascii_mode)) {
      for (auto& pair : m_session_status_map) {
        if (pair.first != ipc_id) {
          rime_api->set_option(to_session_id(pair.first), "ascii_mode",
                               !!status.is_ascii_mode);
        }
      }
    }
    session_status.status = status;
    rime_api->free_status(&status);
  }

  const bool has_llm_candidates =
      m_llm_prediction_mode && !_SnapshotLLMCandidates().candidates.empty();
  RIME_STRUCT(RimeContext, ctx);
  if (rime_api->get_context(session_id, &ctx)) {
    if (is_composing) {
      actions.insert("ctx");
      switch (session_status.style.preedit_type) {
        case UIStyle::PREVIEW:
          if (ctx.commit_text_preview != NULL) {
            std::string first = ctx.commit_text_preview;
            messages.push_back(std::string("ctx.preedit=") +
                               escape_string<char>(first) + '\n');
            messages.push_back(
                std::string("ctx.preedit.cursor=") +
                std::to_string(utf8towcslen(first.c_str(), 0)) + ',' +
                std::to_string(utf8towcslen(first.c_str(), (int)first.size())) +
                ',' +
                std::to_string(utf8towcslen(first.c_str(), (int)first.size())) +
                '\n');
            break;
          }
          // no preview, fall back to composition
        case UIStyle::COMPOSITION:
          messages.push_back(std::string("ctx.preedit=") +
                             escape_string<char>(ctx.composition.preedit) +
                             '\n');
          if (ctx.composition.sel_start <= ctx.composition.sel_end) {
            messages.push_back(
                std::string("ctx.preedit.cursor=") +
                std::to_string(utf8towcslen(ctx.composition.preedit,
                                            ctx.composition.sel_start)) +
                ',' +
                std::to_string(utf8towcslen(ctx.composition.preedit,
                                            ctx.composition.sel_end)) +
                ',' +
                std::to_string(utf8towcslen(ctx.composition.preedit,
                                            ctx.composition.cursor_pos)) +
                '\n');
          }
          break;
        case UIStyle::PREVIEW_ALL: {
          CandidateInfo cinfo;
          _GetCandidateInfo(cinfo, ctx);
          std::string topush = std::string("ctx.preedit=") +
                               escape_string<char>(ctx.composition.preedit) +
                               "  [";
          for (size_t i = 0; i < cinfo.candies.size(); ++i) {
            std::string label =
                session_status.style.label_font_point > 0
                    ? _GetLabelText(
                          cinfo.labels, static_cast<int>(i),
                          session_status.style.label_text_format.c_str())
                    : "";
            std::string comment = session_status.style.comment_font_point > 0
                                      ? wtou8(cinfo.comments.at(i).str)
                                      : "";
            std::string mark_text = session_status.style.mark_text.empty()
                                        ? "*"
                                        : wtou8(session_status.style.mark_text);
            std::string prefix =
                (static_cast<int>(i) != cinfo.highlighted) ? "" : mark_text;
            topush += " " + prefix + escape_string(label) +
                      escape_string<char>(wtou8(cinfo.candies.at(i).str)) +
                      " " + escape_string(comment);
          }
          messages.push_back(topush + " ]\n");
          if (ctx.composition.sel_start <= ctx.composition.sel_end) {
            messages.push_back(
                std::string("ctx.preedit.cursor=") +
                std::to_string(utf8towcslen(ctx.composition.preedit,
                                            ctx.composition.sel_start)) +
                ',' +
                std::to_string(utf8towcslen(ctx.composition.preedit,
                                            ctx.composition.sel_end)) +
                ',' +
                std::to_string(utf8towcslen(ctx.composition.preedit,
                                            ctx.composition.cursor_pos)) +
                '\n');
          }
          break;
        }
      }
    }

    if (ctx.menu.num_candidates || has_llm_candidates) {
      CandidateInfo cinfo;
      std::wstringstream ss;
      boost::archive::text_woarchive oa(ss);
      _GetCandidateInfo(cinfo, ctx);
      oa << cinfo;
      messages.push_back(std::string("ctx.cand=") + wtou8(ss.str()) + '\n');
    }
    rime_api->free_context(&ctx);
  } else if (has_llm_candidates) {
    CandidateInfo cinfo;
    std::wstringstream ss;
    boost::archive::text_woarchive oa(ss);
    RimeContext empty_ctx = {0};
    _GetCandidateInfo(cinfo, empty_ctx);
    oa << cinfo;
    messages.push_back(std::string("ctx.cand=") + wtou8(ss.str()) + '\n');
  }

  actions.insert("config");
  messages.push_back(std::string("config.inline_preedit=") +
                     std::to_string((int)session_status.style.inline_preedit) +
                     '\n');

  if (!session_status.__synced) {
    messages.push_back(std::string("config.hide_ime_mode_icon=") +
                       std::to_string((int)hide_ime_mode_icon) + "\n");
    std::wstringstream ss;
    boost::archive::text_woarchive oa(ss);
    oa << session_status.style;

    actions.insert("style");
    messages.push_back(std::string("style=") + wtou8(ss.str().c_str()) + '\n');
    session_status.__synced = true;
  }

  if (actions.empty()) {
    messages.insert(messages.begin(), std::string("action=noop\n"));
  } else {
    messages.insert(messages.begin(),
                    std::string("action=") + join(actions, ",") + '\n');
  }

  messages.push_back(std::string(".\n"));

  if (!eat) {
    return true;
  }

  return std::all_of(messages.begin(), messages.end(),
                     [&eat](std::string& msg) {
                       auto wmsg = u8tow(msg);
                       return eat(wmsg);
                     });
}

static inline COLORREF blend_colors(COLORREF fcolor, COLORREF bcolor) {
  // 提取各通道的值
  BYTE fA = (fcolor >> 24) & 0xFF;  // 获取前景的 alpha 通道
  BYTE fB = (fcolor >> 16) & 0xFF;  // 获取前景的 blue 通道
  BYTE fG = (fcolor >> 8) & 0xFF;   // 获取前景的 green 通道
  BYTE fR = fcolor & 0xFF;          // 获取前景的 red 通道
  BYTE bA = (bcolor >> 24) & 0xFF;  // 获取背景的 alpha 通道
  BYTE bB = (bcolor >> 16) & 0xFF;  // 获取背景的 blue 通道
  BYTE bG = (bcolor >> 8) & 0xFF;   // 获取背景的 green 通道
  BYTE bR = bcolor & 0xFF;          // 获取背景的 red 通道
  // 将 alpha 通道转换为 [0, 1] 的浮动值
  float fAlpha = fA / 255.0f;
  float bAlpha = bA / 255.0f;
  // 计算每个通道的加权平均值
  float retAlpha = fAlpha + (1 - fAlpha) * bAlpha;
  // 混合红、绿、蓝通道
  BYTE retR = (BYTE)((fR * fAlpha + bR * bAlpha * (1 - fAlpha)) / retAlpha);
  BYTE retG = (BYTE)((fG * fAlpha + bG * bAlpha * (1 - fAlpha)) / retAlpha);
  BYTE retB = (BYTE)((fB * fAlpha + bB * bAlpha * (1 - fAlpha)) / retAlpha);
  // 返回合成后的颜色
  return (BYTE)(retAlpha * 255) << 24 | retB << 16 | retG << 8 | retR;
}
// parse color value, with fallback value
static Bool _RimeGetColor(RimeConfig* config,
                          const std::string key,
                          int& value,
                          const ColorFormat& fmt,
                          const unsigned int& fallback) {
  RimeApi* rime_api = rime_get_api();
  char color[256] = {0};
  if (!rime_api->config_get_string(config, key.c_str(), color, 256)) {
    value = fallback;
    return False;
  }
  const auto color_str = std::string(color);
  const auto make_opaque = [&](int& value) {
    value = (fmt != COLOR_RGBA) ? (value | 0xff000000)
                                : ((value << 8) | 0x000000ff);
  };
  const auto ConvertColorToAbgr = [](int color, ColorFormat fmt = COLOR_ABGR) {
    if (fmt == COLOR_ABGR)
      return color & 0xffffffff;
    else if (fmt == COLOR_ARGB)
      return ARGB2ABGR(color) & 0xffffffff;
    else
      return RGBA2ABGR(color) & 0xffffffff;
  };
  if (std::regex_match(color_str, HEX_REGEX)) {
    auto tmp = std::regex_replace(color_str, TRIMHEAD_REGEX, "").substr(0, 8);
    switch (tmp.length()) {
      case 6:  // color code without alpha, xxyyzz add alpha ff
        value = std::stoul(tmp, 0, 16);
        make_opaque(value);
        break;
      case 3:  // color hex code xyz => xxyyzz and alpha ff
        tmp = std::string(2, tmp[0]) + std::string(2, tmp[1]) +
              std::string(2, tmp[2]);
        value = std::stoul(tmp, 0, 16);
        make_opaque(value);
        break;
      case 4:  // color hex code vxyz => vvxxyyzz
        tmp = std::string(2, tmp[0]) + std::string(2, tmp[1]) +
              std::string(2, tmp[2]) + std::string(2, tmp[3]);
        value = std::stoul(tmp, 0, 16);
        break;
      case 7:
      case 8:  // color code with alpha
        value = std::stoul(tmp, 0, 16);
        break;
      default:  // invalid length
        value = fallback;
        return False;
    }
  } else {
    int tmp = 0;
    if (!rime_api->config_get_int(config, key.c_str(), &tmp)) {
      value = fallback;
      return False;
    } else
      value = tmp;
    make_opaque(value);
  }
  value = ConvertColorToAbgr(value, fmt);
  return True;
}
// parset bool type configuration to T type value trueValue / falseValue
template <typename T>
void _RimeGetBool(RimeConfig* config,
                  const char* key,
                  bool cond,
                  T& value,
                  const T& trueValue = true,
                  const T& falseValue = false) {
  RimeApi* rime_api = rime_get_api();
  Bool tempb = False;
  if (rime_api->config_get_bool(config, key, &tempb) || cond)
    value = (!!tempb) ? trueValue : falseValue;
}
//	parse string option to T type value, with fallback
template <typename T>
void _RimeParseStringOptWithFallback(RimeConfig* config,
                                     const std::string& key,
                                     T& value,
                                     const std::map<std::string, T>& amap,
                                     const T& fallback) {
  RimeApi* rime_api = rime_get_api();
  char str_buff[256] = {0};
  if (rime_api->config_get_string(config, key.c_str(), str_buff, 255)) {
    auto it = amap.find(std::string(str_buff));
    value = (it != amap.end()) ? it->second : fallback;
  } else
    value = fallback;
}

template <typename T>
void _RimeGetIntStr(RimeConfig* config,
                    const char* key,
                    T& value,
                    const char* fb_key = nullptr,
                    const void* fb_value = nullptr,
                    const std::function<void(T&)>& func = nullptr) {
  RimeApi* rime_api = rime_get_api();
  if constexpr (std::is_same<T, int>::value) {
    if (!rime_api->config_get_int(config, key, &value) && fb_key != 0)
      rime_api->config_get_int(config, fb_key, &value);
  } else if constexpr (std::is_same<T, std::wstring>::value) {
    const int BUF_SIZE = 2047;
    char buffer[BUF_SIZE + 1] = {0};
    if (rime_api->config_get_string(config, key, buffer, BUF_SIZE) ||
        rime_api->config_get_string(config, fb_key, buffer, BUF_SIZE)) {
      value = u8tow(buffer);
    } else if (fb_value) {
      value = *(T*)fb_value;
    }
  }
  if (func)
    func(value);
}

void RimeWithWeaselHandler::_UpdateShowNotifications(RimeConfig* config,
                                                     bool initialize) {
  Bool show_notifications = true;
  RimeConfigIterator iter;
  if (initialize)
    m_show_notifications_base.clear();
  m_show_notifications.clear();

  if (rime_api->config_get_bool(config, "show_notifications",
                                &show_notifications)) {
    // config read as bool, for gloal all on or off
    if (show_notifications)
      m_show_notifications["always"] = true;
    if (initialize)
      m_show_notifications_base = m_show_notifications;
  } else if (rime_api->config_begin_list(&iter, config, "show_notifications")) {
    // config read as list, list item should be option name in schema
    // or key word 'schema' for schema switching tip
    while (rime_api->config_next(&iter)) {
      char buffer[256] = {0};
      if (rime_api->config_get_string(config, iter.path, buffer, 256))
        m_show_notifications[std::string(buffer)] = true;
    }
    if (initialize)
      m_show_notifications_base = m_show_notifications;
    rime_api->config_end(&iter);
  } else {
    // not configured, or incorrect type
    if (initialize)
      m_show_notifications_base["always"] = true;
    m_show_notifications = m_show_notifications_base;
  }
}

// update ui's style parameters, ui has been check before referenced
static void _UpdateUIStyle(RimeConfig* config, UI* ui, bool initialize) {
  UIStyle& style(ui->style());
  const std::function<void(std::wstring&)> rmspace = [](std::wstring& str) {
    str = std::regex_replace(str, std::wregex(L"\\s*(,|:|^|$)\\s*"), L"$1");
  };
  const std::function<void(int&)> _abs = [](int& value) { value = abs(value); };
  // get font faces
  _RimeGetIntStr(config, "style/font_face", style.font_face, 0, 0, rmspace);
  std::wstring* const pFallbackFontFace = initialize ? &style.font_face : NULL;
  _RimeGetIntStr(config, "style/label_font_face", style.label_font_face, 0,
                 pFallbackFontFace, rmspace);
  _RimeGetIntStr(config, "style/comment_font_face", style.comment_font_face, 0,
                 pFallbackFontFace, rmspace);
  // able to set label font/comment font empty, force fallback to font face.
  if (style.label_font_face.empty())
    style.label_font_face = style.font_face;
  if (style.comment_font_face.empty())
    style.comment_font_face = style.font_face;
  // get font points
  _RimeGetIntStr(config, "style/font_point", style.font_point);
  if (style.font_point <= 0)
    style.font_point = 12;
  _RimeGetBool(config, "hide_ime_mode_icon", initialize, hide_ime_mode_icon);
  _RimeGetIntStr(config, "style/label_font_point", style.label_font_point,
                 "style/font_point", 0, _abs);
  _RimeGetIntStr(config, "style/comment_font_point", style.comment_font_point,
                 "style/font_point", 0, _abs);
  _RimeGetIntStr(config, "style/candidate_abbreviate_length",
                 style.candidate_abbreviate_length, 0, 0, _abs);
  _RimeGetBool(config, "style/inline_preedit", initialize,
               style.inline_preedit);
  _RimeGetBool(config, "style/vertical_auto_reverse", initialize,
               style.vertical_auto_reverse);
  const std::map<std::string, UIStyle::PreeditType> _preeditMap = {
      {std::string("composition"), UIStyle::COMPOSITION},
      {std::string("preview"), UIStyle::PREVIEW},
      {std::string("preview_all"), UIStyle::PREVIEW_ALL}};
  _RimeParseStringOptWithFallback(config, "style/preedit_type",
                                  style.preedit_type, _preeditMap,
                                  style.preedit_type);
  const std::map<std::string, UIStyle::AntiAliasMode> _aliasModeMap = {
      {std::string("force_dword"), UIStyle::FORCE_DWORD},
      {std::string("cleartype"), UIStyle::CLEARTYPE},
      {std::string("grayscale"), UIStyle::GRAYSCALE},
      {std::string("aliased"), UIStyle::ALIASED},
      {std::string("default"), UIStyle::DEFAULT}};
  _RimeParseStringOptWithFallback(config, "style/antialias_mode",
                                  style.antialias_mode, _aliasModeMap,
                                  style.antialias_mode);
  const std::map<std::string, UIStyle::HoverType> _hoverTypeMap = {
      {std::string("none"), UIStyle::HoverType::NONE},
      {std::string("semi_hilite"), UIStyle::HoverType::SEMI_HILITE},
      {std::string("hilite"), UIStyle::HoverType::HILITE}};
  _RimeParseStringOptWithFallback(config, "style/hover_type", style.hover_type,
                                  _hoverTypeMap, style.hover_type);
  const std::map<std::string, UIStyle::LayoutAlignType> _alignType = {
      {std::string("top"), UIStyle::ALIGN_TOP},
      {std::string("center"), UIStyle::ALIGN_CENTER},
      {std::string("bottom"), UIStyle::ALIGN_BOTTOM}};
  _RimeParseStringOptWithFallback(config, "style/layout/align_type",
                                  style.align_type, _alignType,
                                  style.align_type);
  _RimeGetBool(config, "style/display_tray_icon", initialize,
               style.display_tray_icon);
  _RimeGetBool(config, "style/ascii_tip_follow_cursor", initialize,
               style.ascii_tip_follow_cursor);
  _RimeGetBool(config, "style/horizontal", initialize, style.layout_type,
               UIStyle::LAYOUT_HORIZONTAL, UIStyle::LAYOUT_VERTICAL);
  _RimeGetBool(config, "style/paging_on_scroll", initialize,
               style.paging_on_scroll);
  _RimeGetBool(config, "style/click_to_capture", initialize,
               style.click_to_capture, true, false);
  _RimeGetBool(config, "style/fullscreen", false, style.layout_type,
               ((style.layout_type == UIStyle::LAYOUT_HORIZONTAL)
                    ? UIStyle::LAYOUT_HORIZONTAL_FULLSCREEN
                    : UIStyle::LAYOUT_VERTICAL_FULLSCREEN),
               style.layout_type);
  _RimeGetBool(config, "style/vertical_text", false, style.layout_type,
               UIStyle::LAYOUT_VERTICAL_TEXT, style.layout_type);
  _RimeGetBool(config, "style/vertical_text_left_to_right", false,
               style.vertical_text_left_to_right);
  _RimeGetBool(config, "style/vertical_text_with_wrap", false,
               style.vertical_text_with_wrap);
  const std::map<std::string, bool> _text_orientation = {
      {std::string("horizontal"), false}, {std::string("vertical"), true}};
  bool _text_orientation_bool = false;
  _RimeParseStringOptWithFallback(config, "style/text_orientation",
                                  _text_orientation_bool, _text_orientation,
                                  _text_orientation_bool);
  if (_text_orientation_bool)
    style.layout_type = UIStyle::LAYOUT_VERTICAL_TEXT;
  _RimeGetIntStr(config, "style/label_format", style.label_text_format);
  _RimeGetIntStr(config, "style/mark_text", style.mark_text);
  _RimeGetIntStr(config, "style/layout/baseline", style.baseline, 0, 0, _abs);
  _RimeGetIntStr(config, "style/layout/linespacing", style.linespacing, 0, 0,
                 _abs);
  _RimeGetIntStr(config, "style/layout/min_width", style.min_width, 0, 0, _abs);
  _RimeGetIntStr(config, "style/layout/max_width", style.max_width, 0, 0, _abs);
  _RimeGetIntStr(config, "style/layout/min_height", style.min_height, 0, 0,
                 _abs);
  _RimeGetIntStr(config, "style/layout/max_height", style.max_height, 0, 0,
                 _abs);
  // layout (alternative to style/horizontal)
  const std::map<std::string, UIStyle::LayoutType> _layoutMap = {
      {std::string("vertical"), UIStyle::LAYOUT_VERTICAL},
      {std::string("horizontal"), UIStyle::LAYOUT_HORIZONTAL},
      {std::string("vertical_text"), UIStyle::LAYOUT_VERTICAL_TEXT},
      {std::string("vertical+fullscreen"), UIStyle::LAYOUT_VERTICAL_FULLSCREEN},
      {std::string("horizontal+fullscreen"),
       UIStyle::LAYOUT_HORIZONTAL_FULLSCREEN}};
  _RimeParseStringOptWithFallback(config, "style/layout/type",
                                  style.layout_type, _layoutMap,
                                  style.layout_type);
  // disable max_width when full screen
  if (style.layout_type == UIStyle::LAYOUT_HORIZONTAL_FULLSCREEN ||
      style.layout_type == UIStyle::LAYOUT_VERTICAL_FULLSCREEN) {
    style.max_width = 0;
    style.inline_preedit = false;
  }
  _RimeGetIntStr(config, "style/layout/border", style.border,
                 "style/layout/border_width", 0, _abs);
  _RimeGetIntStr(config, "style/layout/margin_x", style.margin_x);
  _RimeGetIntStr(config, "style/layout/margin_y", style.margin_y);
  _RimeGetIntStr(config, "style/layout/spacing", style.spacing, 0, 0, _abs);
  _RimeGetIntStr(config, "style/layout/candidate_spacing",
                 style.candidate_spacing, 0, 0, _abs);
  _RimeGetIntStr(config, "style/layout/hilite_spacing", style.hilite_spacing, 0,
                 0, _abs);
  _RimeGetIntStr(config, "style/layout/hilite_padding_x",
                 style.hilite_padding_x, "style/layout/hilite_padding", 0,
                 _abs);
  _RimeGetIntStr(config, "style/layout/hilite_padding_y",
                 style.hilite_padding_y, "style/layout/hilite_padding", 0,
                 _abs);
  _RimeGetIntStr(config, "style/layout/shadow_radius", style.shadow_radius, 0,
                 0, _abs);
  // disable shadow for fullscreen layout
  style.shadow_radius *=
      (!(style.layout_type == UIStyle::LAYOUT_HORIZONTAL_FULLSCREEN ||
         style.layout_type == UIStyle::LAYOUT_VERTICAL_FULLSCREEN));
  _RimeGetIntStr(config, "style/layout/shadow_offset_x", style.shadow_offset_x);
  _RimeGetIntStr(config, "style/layout/shadow_offset_y", style.shadow_offset_y);
  // round_corner as alias of hilited_corner_radius
  _RimeGetIntStr(config, "style/layout/hilited_corner_radius",
                 style.round_corner, "style/layout/round_corner", 0, _abs);
  // corner_radius not set, fallback to round_corner
  _RimeGetIntStr(config, "style/layout/corner_radius", style.round_corner_ex,
                 "style/layout/round_corner", 0, _abs);
  // fix padding and spacing settings
  if (style.layout_type != UIStyle::LAYOUT_VERTICAL_TEXT) {
    // hilite_padding vs spacing
    // if hilite_padding over spacing, increase spacing
    style.spacing = max(style.spacing, style.hilite_padding_y * 2);
    // hilite_padding vs candidate_spacing
    if (style.layout_type == UIStyle::LAYOUT_VERTICAL_FULLSCREEN ||
        style.layout_type == UIStyle::LAYOUT_VERTICAL) {
      // vertical, if hilite_padding_y over candidate spacing,
      // increase candidate spacing
      style.candidate_spacing =
          max(style.candidate_spacing, style.hilite_padding_y * 2);
    } else {
      // horizontal, if hilite_padding_x over candidate
      // spacing, increase candidate spacing
      style.candidate_spacing =
          max(style.candidate_spacing, style.hilite_padding_x * 2);
    }
    // hilite_padding_x vs hilite_spacing
    if (!style.inline_preedit)
      style.hilite_spacing = max(style.hilite_spacing, style.hilite_padding_x);
  } else  // LAYOUT_VERTICAL_TEXT
  {
    // hilite_padding_x vs spacing
    // if hilite_padding over spacing, increase spacing
    style.spacing = max(style.spacing, style.hilite_padding_x * 2);
    // hilite_padding vs candidate_spacing
    // if hilite_padding_x over candidate
    // spacing, increase candidate spacing
    style.candidate_spacing =
        max(style.candidate_spacing, style.hilite_padding_x * 2);
    // vertical_text_with_wrap and hilite_padding_y over candidate_spacing
    if (style.vertical_text_with_wrap)
      style.candidate_spacing =
          max(style.candidate_spacing, style.hilite_padding_y * 2);
    // hilite_padding_y vs hilite_spacing
    if (!style.inline_preedit)
      style.hilite_spacing = max(style.hilite_spacing, style.hilite_padding_y);
  }
  // fix padding and margin settings
  int scale = style.margin_x < 0 ? -1 : 1;
  style.margin_x = scale * max(style.hilite_padding_x, abs(style.margin_x));
  scale = style.margin_y < 0 ? -1 : 1;
  style.margin_y = scale * max(style.hilite_padding_y, abs(style.margin_y));
  // get enhanced_position
  _RimeGetBool(config, "style/enhanced_position", initialize,
               style.enhanced_position, true, false);
  // get color scheme
  const int BUF_SIZE = 255;
  char buffer[BUF_SIZE + 1] = {0};
  if (initialize && rime_api->config_get_string(config, "style/color_scheme",
                                                buffer, BUF_SIZE))
    _UpdateUIStyleColor(config, style);
}
// load color configs to style, by "style/color_scheme" or specific scheme name
// "color" which is default empty
static bool _UpdateUIStyleColor(RimeConfig* config,
                                UIStyle& style,
                                std::string color) {
  const int BUF_SIZE = 255;
  char buffer[BUF_SIZE + 1] = {0};
  std::string color_mark = "style/color_scheme";
  // color scheme
  if (rime_api->config_get_string(config, color_mark.c_str(), buffer,
                                  BUF_SIZE) ||
      !color.empty()) {
    std::string prefix("preset_color_schemes/");
    prefix += (color.empty()) ? buffer : color;
    // define color format, default abgr if not set
    ColorFormat fmt = COLOR_ABGR;
    const std::map<std::string, ColorFormat> _colorFmt = {
        {std::string("argb"), COLOR_ARGB},
        {std::string("rgba"), COLOR_RGBA},
        {std::string("abgr"), COLOR_ABGR}};
    _RimeParseStringOptWithFallback(config, (prefix + "/color_format"), fmt,
                                    _colorFmt, COLOR_ABGR);
#define COLOR(key, value, fallback) \
  _RimeGetColor(config, (prefix + "/" + key), value, fmt, fallback)
    COLOR("back_color", style.back_color, 0xffffffff);
    COLOR("shadow_color", style.shadow_color, 0);
    COLOR("prevpage_color", style.prevpage_color, 0);
    COLOR("nextpage_color", style.nextpage_color, 0);
    COLOR("text_color", style.text_color, 0xff000000);
    COLOR("candidate_text_color", style.candidate_text_color, style.text_color);
    COLOR("candidate_back_color", style.candidate_back_color, 0);
    COLOR("border_color", style.border_color, style.text_color);
    COLOR("hilited_text_color", style.hilited_text_color, style.text_color);
    COLOR("hilited_back_color", style.hilited_back_color, style.back_color);
    COLOR("hilited_candidate_text_color", style.hilited_candidate_text_color,
          style.hilited_text_color);
    COLOR("hilited_candidate_back_color", style.hilited_candidate_back_color,
          style.hilited_back_color);
    COLOR("hilited_candidate_shadow_color",
          style.hilited_candidate_shadow_color, 0);
    COLOR("hilited_shadow_color", style.hilited_shadow_color, 0);
    COLOR("candidate_shadow_color", style.candidate_shadow_color, 0);
    COLOR("candidate_border_color", style.candidate_border_color, 0);
    COLOR("hilited_candidate_border_color",
          style.hilited_candidate_border_color, 0);
    COLOR("label_color", style.label_text_color,
          blend_colors(style.candidate_text_color, style.candidate_back_color));
    COLOR("hilited_label_color", style.hilited_label_text_color,
          blend_colors(style.hilited_candidate_text_color,
                       style.hilited_candidate_back_color));
    COLOR("comment_text_color", style.comment_text_color,
          style.label_text_color);
    COLOR("hilited_comment_text_color", style.hilited_comment_text_color,
          style.hilited_label_text_color);
    COLOR("hilited_mark_color", style.hilited_mark_color, 0);
#undef COLOR
    return true;
  }
  return false;
}

static void _LoadAppOptions(RimeConfig* config,
                            AppOptionsByAppName& app_options) {
  app_options.clear();
  RimeConfigIterator app_iter;
  RimeConfigIterator option_iter;
  rime_api->config_begin_map(&app_iter, config, "app_options");
  while (rime_api->config_next(&app_iter)) {
    AppOptions& options(app_options[app_iter.key]);
    rime_api->config_begin_map(&option_iter, config, app_iter.path);
    while (rime_api->config_next(&option_iter)) {
      Bool value = False;
      if (rime_api->config_get_bool(config, option_iter.path, &value)) {
        options[option_iter.key] = !!value;
      }
    }
    rime_api->config_end(&option_iter);
  }
  rime_api->config_end(&app_iter);
}

void RimeWithWeaselHandler::_GetStatus(Status& stat,
                                       WeaselSessionId ipc_id,
                                       Context& ctx) {
  const bool has_llm_candidates =
      m_llm_prediction_mode && !_SnapshotLLMCandidates().candidates.empty();
  SessionStatus& session_status = get_session_status(ipc_id);
  RimeSessionId session_id = session_status.session_id;
  RIME_STRUCT(RimeStatus, status);
  if (rime_api->get_status(session_id, &status)) {
    std::string schema_id = "";
    if (status.schema_id)
      schema_id = status.schema_id;
    stat.schema_name = u8tow(status.schema_name);
    stat.schema_id = u8tow(status.schema_id);
    stat.ascii_mode = !!status.is_ascii_mode;
    stat.composing = !!status.is_composing;
    
    // 如果处于LLM预测模式，强制设置composing为true以显示候选栏
    if (has_llm_candidates) {
      stat.composing = true;
      if (m_dev_console && m_dev_console->IsEnabled()) {
        m_dev_console->WriteLine(L"[_GetStatus] LLM预测模式激活，强制设置 composing=true");
      }
    }
    
    stat.disabled = !!status.is_disabled;
    stat.full_shape = !!status.is_full_shape;
    if (schema_id != m_last_schema_id) {
      session_status.__synced = false;
      m_last_schema_id = schema_id;
      if (schema_id != ".default") {  // don't load for schema select menu
        bool inline_preedit = session_status.style.inline_preedit;
        _LoadSchemaSpecificSettings(ipc_id, schema_id);
        _LoadAppInlinePreeditSet(ipc_id, true);
        if (session_status.style.inline_preedit != inline_preedit)
          // in case of inline_preedit set in schema
          _UpdateInlinePreeditStatus(ipc_id);
        // refresh icon after schema changed
        _RefreshTrayIcon(session_id, _UpdateUICallback);
        m_ui->style() = session_status.style;
        if (m_show_notifications.find("schema") != m_show_notifications.end() &&
            m_show_notifications_time > 0) {
          ctx.aux.str = stat.schema_name;
          m_ui->Update(ctx, stat);
          m_ui->ShowWithTimeout(m_show_notifications_time);
        }
      }
    }
    rime_api->free_status(&status);
  }
}

void RimeWithWeaselHandler::_GetContext(Context& weasel_context,
                                        RimeSessionId session_id) {
  const bool has_llm_candidates =
      m_llm_prediction_mode && !_SnapshotLLMCandidates().candidates.empty();
  RIME_STRUCT(RimeContext, ctx);
  if (rime_api->get_context(session_id, &ctx)) {
    if (ctx.composition.length > 0) {
      weasel_context.preedit.str = u8tow(ctx.composition.preedit);
      if (ctx.composition.sel_start < ctx.composition.sel_end) {
        TextAttribute attr;
        attr.type = HIGHLIGHTED;
        attr.range.start =
            utf8towcslen(ctx.composition.preedit, ctx.composition.sel_start);
        attr.range.end =
            utf8towcslen(ctx.composition.preedit, ctx.composition.sel_end);

        weasel_context.preedit.attributes.push_back(attr);
      }
    }
    
    // 获取候选词信息（包括Rime和LLM候选词）
      CandidateInfo& cinfo(weasel_context.cinfo);
    if (ctx.menu.num_candidates > 0) {
      // 有Rime候选词，调用_GetCandidateInfo会同时添加Rime和LLM候选词
      _GetCandidateInfo(cinfo, ctx);
      if (m_dev_console && m_dev_console->IsEnabled()) {
        std::wstringstream ss;
        ss << L"[DEBUG] _GetContext: 有Rime候选词(" << ctx.menu.num_candidates 
           << L"个)，添加后总候选词数=" << cinfo.candies.size();
        m_dev_console->WriteLine(ss.str());
      }
    } else if (has_llm_candidates) {
      // 如果处于LLM预测模式但没有Rime候选词，只显示LLM候选词
      cinfo.clear();
      _GetCandidateInfo(cinfo, ctx);  // 这会添加LLM候选词
      if (m_dev_console && m_dev_console->IsEnabled()) {
        std::wstringstream ss;
        ss << L"[DEBUG] _GetContext: 无Rime候选词，添加LLM候选词后 cinfo.candies.size()=" 
           << cinfo.candies.size();
        m_dev_console->WriteLine(ss.str());
      }
    } else {
      // 既没有Rime候选词，也没有LLM候选词，清空候选词信息
      cinfo.clear();
    }
    
    rime_api->free_context(&ctx);
  } else if (has_llm_candidates) {
    // 如果没有Rime上下文但处于LLM预测模式，创建空的候选词信息并添加LLM候选词
    CandidateInfo& cinfo(weasel_context.cinfo);
    cinfo.clear();
    RimeContext empty_ctx = {0};
    _GetCandidateInfo(cinfo, empty_ctx);  // 这会添加LLM候选词
    if (m_dev_console && m_dev_console->IsEnabled()) {
      std::wstringstream ss;
      ss << L"[DEBUG] _GetContext: 无Rime上下文，添加LLM候选词后 cinfo.candies.size()=" 
         << cinfo.candies.size();
      m_dev_console->WriteLine(ss.str());
    }
  }
}

bool RimeWithWeaselHandler::_IsSessionTSF(RimeSessionId session_id) {
  static char client_type[20] = {0};
  rime_api->get_property(session_id, "client_type", client_type,
                         sizeof(client_type) - 1);
  return std::string(client_type) == "tsf";
}

void RimeWithWeaselHandler::_UpdateInlinePreeditStatus(WeaselSessionId ipc_id) {
  if (!m_ui)
    return;
  SessionStatus& session_status = get_session_status(ipc_id);
  RimeSessionId session_id = session_status.session_id;
  // set inline_preedit option
  bool inline_preedit =
      session_status.style.inline_preedit && _IsSessionTSF(session_id);
  rime_api->set_option(session_id, "inline_preedit", Bool(inline_preedit));
  // show soft cursor on weasel panel but not inline
  rime_api->set_option(session_id, "soft_cursor", Bool(!inline_preedit));
}

void RimeWithWeaselHandler::SetContextHistory(ContextHistory* context_history) {
  m_context_history = context_history;
}

void RimeWithWeaselHandler::SetDevConsole(DevConsole* dev_console) {
  m_dev_console = dev_console;
  // 设置全局开发终端实例供LLMProvider使用
  extern DevConsole* g_dev_console;
  g_dev_console = dev_console;
  
  // 输出LLM提供者状态
  if (m_dev_console && m_dev_console->IsEnabled()) {
    if (!m_llm_provider) {
      m_dev_console->WriteLine(L"[LLM] LLM提供者未初始化");
      m_dev_console->WriteLine(L"[LLM] 请在weasel.yaml中配置：");
      m_dev_console->WriteLine(L"[LLM]   llm:");
      m_dev_console->WriteLine(L"[LLM]     enabled: true");
      m_dev_console->WriteLine(L"[LLM]     openai:");
      m_dev_console->WriteLine(L"[LLM]       api_key: \"your-api-key\"");
    } else if (!m_llm_provider->IsAvailable()) {
      m_dev_console->WriteLine(L"[LLM] LLM提供者已初始化，但不可用");
      m_dev_console->WriteLine(L"[LLM] 请检查配置：llm/enabled 和 llm/openai/api_key");
    } else {
      std::wstring provider_name = u8tow(m_llm_provider->GetProviderName());
      m_dev_console->WriteLine(L"[LLM] LLM提供者已就绪: " + provider_name);
    }
    m_dev_console->WriteLine(std::wstring(L"[LLM] 开发者模式词源标注: ") +
                             (m_llm_developer_mode ? L"开启" : L"关闭"));
  }
}

void RimeWithWeaselHandler::_TriggerLLMPrediction(
    WeaselSessionId ipc_id,
    LLMRequestType request_type,
    const std::wstring& current_input,
    bool require_rime_candidates,
    DWORD debounce_ms) {
  if (!m_llm_provider || !m_llm_provider->IsAvailable()) {
    LOG(WARNING) << "[LLM] LLM provider is not available or not initialized";
    if (m_dev_console && m_dev_console->IsEnabled()) {
      m_dev_console->WriteLine(L"[LLM] LLM提供者不可用，无法进行预测");
    }
    return;
  }

  LOG(INFO) << "[LLM] Starting LLM prediction, ipc_id=" << ipc_id;

  // 从上下文历史获取最近 50 词作为 LLM 上下文
  std::wstring context;
  std::wstring preference_hint;
  if (m_context_history) {
    context = m_context_history->GetRecentContext(m_llm_context_recent_words);
    context = _TrimPredictionContext(context);
    preference_hint = m_context_history->GetPreferenceHint(3);
    LOG(INFO) << "[LLM] Context from history, length=" << context.length();
  }

  if (current_input.empty()) {
    preference_hint.clear();
  }

  LLMRequest request;
  request.type = request_type;
  request.context = context;
  request.current_input = current_input;
  request.preference_hint = preference_hint;
  request.max_candidates = 5;

  if (request.type == LLMRequestType::RimeReorder) {
    const RimeSessionId session_id = to_session_id(ipc_id);
    RIME_STRUCT(RimeContext, reorder_ctx);
    if (rime_api->get_context(session_id, &reorder_ctx)) {
      const size_t rime_candidate_count =
          static_cast<size_t>(reorder_ctx.menu.num_candidates);
      request.rime_candidates.reserve(rime_candidate_count);
      for (size_t i = 0; i < rime_candidate_count; ++i) {
        if (reorder_ctx.menu.candidates[i].text) {
          request.rime_candidates.push_back(
              u8tow(reorder_ctx.menu.candidates[i].text));
        }
      }
      rime_api->free_context(&reorder_ctx);
    }

    if (request.rime_candidates.empty()) {
      request.type = current_input.empty()
                         ? LLMRequestType::NoInputPrediction
                         : LLMRequestType::PinyinConstrainedPrediction;
      if (m_dev_console && m_dev_console->IsEnabled()) {
        m_dev_console->WriteLine(
            L"[LLM] 未捕获到 Rime 候选词，降级为非重排请求");
      }
    }
  }

  const bool enable_rime_reorder = request.type == LLMRequestType::RimeReorder;

  // 如果上下文仍然为空，记录警告但尝试继续预测
  if (context.empty()) {
    LOG(WARNING) << "[LLM] Context is empty, attempting prediction with empty context";
    if (m_dev_console && m_dev_console->IsEnabled()) {
      m_dev_console->WriteLine(L"[LLM] 警告：上下文为空，将使用空上下文进行预测");
    }
    // 不返回，继续尝试预测以支持冷启动
  }

  if (m_dev_console && m_dev_console->IsEnabled()) {
    m_dev_console->WriteLine(L"[LLM] ========== 开始LLM预测 ==========");
    m_dev_console->WriteLine(
        L"[LLM] 请求类型: " + std::wstring(GetLLMRequestTypeName(request.type)));
    if (!context.empty()) {
      m_dev_console->WriteLine(L"[LLM] 上下文长度: " + std::to_wstring(context.length()));
      m_dev_console->WriteLine(L"[LLM] 上下文内容: " + context);
    } else {
      m_dev_console->WriteLine(L"[LLM] 使用空上下文进行预测");
    }
    if (!current_input.empty()) {
      m_dev_console->WriteLine(L"[LLM] 当前输入（拼音）: " + current_input);
    }
    if (!preference_hint.empty()) {
      m_dev_console->WriteLine(L"[LLM] 用户偏好: " + preference_hint);
    }
    if (!request.rime_candidates.empty()) {
      m_dev_console->WriteLine(
          L"[LLM] Rime 候选数: " + std::to_wstring(request.rime_candidates.size()));
    }
  }

  LOG(INFO) << "[LLM] Context length: " << context.length();
  LOG(INFO) << "[LLM] Current input: " << wtou8(current_input);
  LOG(INFO) << "[LLM] Scheduling async unified LLM request";

  m_has_display_highlight_override = false;
  // 生成新的请求序号，用于标记“最新一次”预测请求
  const uint64_t request_seq = ++m_llm_request_seq;

  // 拷贝必要参数到后台线程
  LLMRequest request_copy = request;

  std::thread([this, ipc_id, request_seq, request_copy, require_rime_candidates,
               enable_rime_reorder, debounce_ms]() {
    if (debounce_ms > 0) {
      ::Sleep(debounce_ms);
      if (request_seq != m_llm_request_seq.load()) {
        LOG(INFO) << "[LLM] Debounced request became stale before execution, seq="
                  << request_seq;
        return;
      }
    }
    // 后台线程中执行统一请求，不阻塞用户输入线程
    LOG(INFO) << "[LLM] Async thread executing unified LLM request, seq="
              << request_seq;
    auto candidates = m_llm_provider->ExecuteRequest(request_copy);

    // 如果有更新的请求已经发起，则丢弃本次结果
    if (request_seq != m_llm_request_seq.load()) {
      LOG(INFO) << "[LLM] Discarding stale LLM result, seq=" << request_seq
                << ", latest_seq=" << m_llm_request_seq.load();
      return;
    }

    // 将结果写入共享状态
    size_t candidate_count = 0;
    {
      std::lock_guard<std::mutex> lock(m_llm_mutex);
      m_current_llm_candidates = std::move(candidates);
      m_current_llm_candidates_require_rime = require_rime_candidates;
      m_current_llm_candidates_enable_rime_reorder = enable_rime_reorder;
      candidate_count = m_current_llm_candidates.size();
    }

    LOG(INFO) << "[LLM] Async LLMProvider returned " << candidate_count << " candidates, seq=" << request_seq;

    if (m_dev_console && m_dev_console->IsEnabled()) {
      std::wstringstream ss;
      ss << L"[LLM] 异步预测完成，获得 " << candidate_count << L" 个候选词";
      m_dev_console->WriteLine(ss.str());
      m_dev_console->WriteLine(L"[LLM] ========== 预测结束 ==========");
    }

    // 更新UI显示LLM候选词
    _UpdateUI(ipc_id);
  }).detach();
}

void RimeWithWeaselHandler::_ExitLLMPredictionMode(WeaselSessionId ipc_id) {
  ++m_llm_request_seq;
  m_llm_prediction_mode = false;
  {
    std::lock_guard<std::mutex> lock(m_llm_mutex);
    m_current_llm_candidates.clear();
    m_current_llm_candidates_require_rime = false;
    m_current_llm_candidates_enable_rime_reorder = false;
  }
  m_has_display_highlight_override = false;
  
  // 强制隐藏候选栏
  if (m_ui) {
    m_ui->Hide();
    if (m_dev_console && m_dev_console->IsEnabled()) {
      m_dev_console->WriteLine(L"[LLM] 强制隐藏候选栏");
    }
  }
  
  _UpdateUI(ipc_id);
  
  if (m_dev_console && m_dev_console->IsEnabled()) {
    m_dev_console->WriteLine(L"[LLM] 退出LLM预测模式");
  }
}
