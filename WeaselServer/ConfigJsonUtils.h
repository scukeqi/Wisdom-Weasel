#pragma once

#include <rime_api.h>

#include <iomanip>
#include <locale>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace weasel {
namespace config_json {

inline std::string EscapeJsonString(const std::string& value) {
  std::ostringstream escaped;
  for (unsigned char ch : value) {
    switch (ch) {
      case '\"':
        escaped << "\\\"";
        break;
      case '\\':
        escaped << "\\\\";
        break;
      case '\b':
        escaped << "\\b";
        break;
      case '\f':
        escaped << "\\f";
        break;
      case '\n':
        escaped << "\\n";
        break;
      case '\r':
        escaped << "\\r";
        break;
      case '\t':
        escaped << "\\t";
        break;
      default:
        if (ch < 0x20) {
          escaped << "\\u" << std::uppercase << std::hex << std::setw(4)
                  << std::setfill('0') << static_cast<int>(ch)
                  << std::nouppercase << std::dec;
        } else {
          escaped << static_cast<char>(ch);
        }
        break;
    }
  }
  return escaped.str();
}

inline bool TryGetConfigStringValue(RimeApi* rime_api,
                                    RimeConfig* config,
                                    const std::string& path,
                                    std::string& value) {
  if (!rime_api || !config) {
    return false;
  }

  if (rime_api->config_get_cstring) {
    const char* cstr = rime_api->config_get_cstring(config, path.c_str());
    if (cstr) {
      value = cstr;
      return true;
    }
  }

  char buffer[1024] = {0};
  if (rime_api->config_get_string &&
      rime_api->config_get_string(config, path.c_str(), buffer,
                                  sizeof(buffer) - 1)) {
    value = buffer;
    return true;
  }

  return false;
}

inline bool SerializeConfigValueToJson(RimeApi* rime_api,
                                       RimeConfig* config,
                                       const std::string& path,
                                       std::string& json_output) {
  if (!rime_api || !config || path.empty()) {
    return false;
  }

  RimeConfigIterator iter = {0};
  if (rime_api->config_begin_map &&
      rime_api->config_begin_map(&iter, config, path.c_str())) {
    std::ostringstream json;
    json << "{";
    bool first = true;
    while (rime_api->config_next(&iter)) {
      std::string child_json;
      if (!iter.path || !SerializeConfigValueToJson(rime_api, config, iter.path,
                                                    child_json)) {
        continue;
      }
      if (!first) {
        json << ",";
      }
      first = false;
      json << "\"" << EscapeJsonString(iter.key ? iter.key : "")
           << "\":" << child_json;
    }
    rime_api->config_end(&iter);
    json << "}";
    json_output = json.str();
    return true;
  }

  if (rime_api->config_begin_list &&
      rime_api->config_begin_list(&iter, config, path.c_str())) {
    std::ostringstream json;
    json << "[";
    bool first = true;
    while (rime_api->config_next(&iter)) {
      std::string child_json;
      if (!iter.path || !SerializeConfigValueToJson(rime_api, config, iter.path,
                                                    child_json)) {
        continue;
      }
      if (!first) {
        json << ",";
      }
      first = false;
      json << child_json;
    }
    rime_api->config_end(&iter);
    json << "]";
    json_output = json.str();
    return true;
  }

  Bool bool_value = false;
  if (rime_api->config_get_bool &&
      rime_api->config_get_bool(config, path.c_str(), &bool_value)) {
    json_output = bool_value ? "true" : "false";
    return true;
  }

  int int_value = 0;
  if (rime_api->config_get_int &&
      rime_api->config_get_int(config, path.c_str(), &int_value)) {
    json_output = std::to_string(int_value);
    return true;
  }

  if (rime_api->config_get_double) {
    double double_value = 0.0;
    if (rime_api->config_get_double(config, path.c_str(), &double_value)) {
      std::ostringstream json;
      json.imbue(std::locale::classic());
      json << std::setprecision(15) << double_value;
      json_output = json.str();
      return true;
    }
  }

  std::string string_value;
  if (TryGetConfigStringValue(rime_api, config, path, string_value)) {
    json_output = "\"" + EscapeJsonString(string_value) + "\"";
    return true;
  }

  return false;
}

inline bool SerializeConfigMapToJsonObject(RimeApi* rime_api,
                                           RimeConfig* config,
                                           const std::string& path,
                                           std::string& json_output) {
  if (!rime_api || !config || path.empty()) {
    return false;
  }

  RimeConfigIterator iter = {0};
  if (!rime_api->config_begin_map ||
      !rime_api->config_begin_map(&iter, config, path.c_str())) {
    return false;
  }
  rime_api->config_end(&iter);

  return SerializeConfigValueToJson(rime_api, config, path, json_output);
}

inline bool LoadConfigStringMap(
    RimeApi* rime_api,
    RimeConfig* config,
    const std::string& path,
    std::vector<std::pair<std::string, std::string>>& key_values) {
  key_values.clear();
  if (!rime_api || !config || path.empty() || !rime_api->config_begin_map) {
    return false;
  }

  RimeConfigIterator iter = {0};
  if (!rime_api->config_begin_map(&iter, config, path.c_str())) {
    return false;
  }

  while (rime_api->config_next(&iter)) {
    if (!iter.path || !iter.key) {
      continue;
    }
    std::string value;
    if (!TryGetConfigStringValue(rime_api, config, iter.path, value)) {
      Bool bool_value = false;
      if (rime_api->config_get_bool &&
          rime_api->config_get_bool(config, iter.path, &bool_value)) {
        value = bool_value ? "true" : "false";
      } else {
        int int_value = 0;
        if (rime_api->config_get_int &&
            rime_api->config_get_int(config, iter.path, &int_value)) {
          value = std::to_string(int_value);
        } else if (rime_api->config_get_double) {
          double double_value = 0.0;
          if (rime_api->config_get_double(config, iter.path, &double_value)) {
            std::ostringstream ss;
            ss.imbue(std::locale::classic());
            ss << std::setprecision(15) << double_value;
            value = ss.str();
          } else {
            continue;
          }
        } else {
          continue;
        }
      }
    }

    key_values.emplace_back(iter.key, value);
  }

  rime_api->config_end(&iter);
  return true;
}

}  // namespace config_json
}  // namespace weasel
